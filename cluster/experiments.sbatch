#!/usr/bin/env bash
#SBATCH --job-name=torann-exp
#SBATCH --partition=cpuPartition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
# Standalone non-A/B experiment runner on a compute node with /tmp tmpfs staging.
#
# Runs long ablation, crossover, or multi-round ESS studies using a compiled
# arm in $BENCH_ROOT/arms/$ARM (default: main) while leaving the laptop free.
# Can be submitted as a single job (EXP_TASK=recall-ablation) or a 7-task array
# (--array=0-6, where each task picks one workload from the table below).
set -euo pipefail

ROOT="${BENCH_ROOT:?BENCH_ROOT not set}"
ARM="${ARM:-main}"
TASKS=(recall-ablation crossover refine-rounds exploration lp-recall tune-model tune-cost)
if [[ -n "${EXP_TASK:-}" ]]; then
    TASK="$EXP_TASK"
else
    IDX="${SLURM_ARRAY_TASK_ID:-0}"
    TASK="${TASKS[$IDX]}"
fi

ARM_DIR="$ROOT/arms/$ARM"
if [[ ! -d "$ARM_DIR/torann" ]]; then
    echo "error: arm '$ARM' not found in $ARM_DIR (run build_arms.sbatch first)" >&2
    exit 1
fi

export PATH="$ROOT/venv/bin:$PATH"
export PYTHONPATH="$ARM_DIR:$ROOT/torann:${PYTHONPATH:-}"
export RAYON_NUM_THREADS="${RAYON_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

SCRATCH="/tmp/${USER}-${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
OUT_FINAL="$ROOT/torann/examples/out/exp_${TASK}_${ARM}"
mkdir -p "$SCRATCH/out" "$OUT_FINAL"

cleanup() {
    rsync -rlt "$SCRATCH/out/" "$OUT_FINAL/" || true
    rm -rf "$SCRATCH"
}
trap cleanup EXIT

# Stage examples/exploration and link the compiled arm into /tmp so both
# `sys.path[0]` (cwd) and `../` inside scripts resolve to `$ARM_DIR/torann`
# rather than the unbuilt source tree in `$ROOT/torann`.
ln -snf "$ARM_DIR/torann" "$SCRATCH/torann"
cp -r "$ROOT/torann/examples" "$ROOT/torann/exploration" "$SCRATCH/"
cd "$SCRATCH"
echo "=== host=$(hostname) job=${SLURM_JOB_ID} task=${TASK} arm=${ARM} cpus=${RAYON_NUM_THREADS} start=$(date -Is) ==="
grep -m1 'model name' /proc/cpuinfo || true

case "$TASK" in
    recall-ablation)
        # Issue 1: Recall floor study (empty + filled starts, d=2,8,32)
        python examples/bench_recall_ablation.py \
            --json "$SCRATCH/out/recall_ablation.json" \
            | tee "$SCRATCH/out/recall_ablation.log"
        python examples/report_recall_ablation.py \
            --in "$SCRATCH/out/recall_ablation.json" \
            --out "$SCRATCH/out/recall_ablation_report.md"
        ;;
    crossover)
        # S3: Exact brute vs LSH crossover n*(d, mode, p) across d=2..100
        python examples/crossover.py --dims 2,4,8,16,32,64,100 --backends rust --modes knn,radius --ps 1.0,0.75,0.5,0.25 \
            | tee "$SCRATCH/out/crossover.log"
        [[ -f examples/out/crossover.json ]] && cp -f examples/out/crossover.json "$SCRATCH/out/crossover.json"
        ;;
    refine-rounds)
        # Iterative static-tier accumulation vs one-shot ESS (6 rounds, d=2,8,32)
        python examples/bench_refine_rounds.py --rounds 6 \
            --json "$SCRATCH/out/refine_rounds.json" \
            | tee "$SCRATCH/out/refine_rounds.log"
        ;;
    exploration)
        # Math/tuner validation scripts (1D/2D seam laws, cell size, L^p recall, updates)
        for s in exp_1d.py exp_2d.py exp_cellsize.py exp_lp_recall.py exp_metric_contrast.py exp_update.py; do
            echo "--- running exploration/$s ---"
            python "exploration/$s" 2>&1 | tee "$SCRATCH/out/${s%.py}.log"
        done
        ;;
    tune-model)
        # Predicted vs measured per-table hit rate, and _tune stability
        python exploration/exp_tune_model.py --pairs 500 --tables 500 \
            --json "$SCRATCH/out/tune_model.json" | tee "$SCRATCH/out/tune_model.log"
        python exploration/exp_tune_stability.py | tee "$SCRATCH/out/tune_stability.log"
        ;;
    tune-cost)
        # Recall against real query, update and fit time, per probes and K
        python exploration/exp_tune_cost.py --json "$SCRATCH/out/tune_cost.json" \
            | tee "$SCRATCH/out/tune_cost.log"
        ;;
    lp-recall)
        # Large-dataset evaluation of `recall` tuning and toroidal L^p (p <= 1) LSH
        python - "$SCRATCH/out/lp_recall.json" <<'PY' | tee "$SCRATCH/out/lp_recall.log"
import json, sys, time
import numpy as np
from torann import ToroidalNN, rust, tune_thresholds
from torann.brute import exact_knn

out_path = sys.argv[1]
rng = np.random.default_rng(42)
rows = []
print("=== 1. Recall knob sweep (n=30000, m=1000, k=16, p=1.0) ===")
for d in (8, 16, 32):
    pts = rng.random((30000, d))
    q = rng.random((1000, d))
    ref_idx, _ = rust.brute_knn(pts, q, 16, None, 1.0)
    for rec in (0.30, 0.50, 0.75, 0.90, 0.98, "auto"):
        t0 = time.perf_counter()
        nn = ToroidalNN(seed=7, brute_threshold=0, recall=rec).fit(pts[:25000], pts[25000:], k=16)
        assert nn.backend_name == "rust", f"expected rust, got {nn.backend_name}"
        t_fit = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        idx, _ = nn.query(k=16, queries=q)
        t_q = (time.perf_counter() - t1) * 1e3
        emp = sum(len(np.intersect1d(idx[i], ref_idx[i])) for i in range(len(q))) / ref_idx.size
        row = {"exp": "recall", "d": d, "recall_arg": rec, "B": nn._B, "K": nn._K, "L": nn._L,
               "emp_recall": round(float(emp), 4), "fit_ms": round(t_fit, 2), "query_ms": round(t_q, 2)}
        rows.append(row)
        print(f"d={d:2d} recall={str(rec):>5s} -> B={nn._B} K={nn._K:2d} L={nn._L:2d} | emp_recall={emp:.4f} | query={t_q:6.2f} ms")

print("\n=== 2. Toroidal L^p (p <= 1) sweep (n=30000, m=1000, k=16) ===")
for d in (8, 16, 32):
    pts = rng.random((30000, d))
    q = rng.random((1000, d))
    for p in (0.25, 0.50, 0.75, 1.00):
        t_b0 = time.perf_counter()
        ref_idx, _ = rust.brute_knn(pts, q, 16, None, p)
        t_brute = (time.perf_counter() - t_b0) * 1e3
        t0 = time.perf_counter()
        nn = ToroidalNN(seed=7, brute_threshold=0, p=p).fit(pts[:25000], pts[25000:], k=16)
        t_fit = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        idx, _ = nn.query(k=16, queries=q)
        t_q = (time.perf_counter() - t1) * 1e3
        emp = sum(len(np.intersect1d(idx[i], ref_idx[i])) for i in range(len(q))) / ref_idx.size
        row = {"exp": "lp", "d": d, "p": p, "B": nn._B, "K": nn._K, "L": nn._L,
               "emp_recall": round(float(emp), 4), "fit_ms": round(t_fit, 2),
               "query_ms": round(t_q, 2), "rust_brute_ms": round(t_brute, 2),
               "speedup_vs_rust_brute": round(t_brute / max(t_q, 1e-6), 2)}
        rows.append(row)
        print(f"d={d:2d} p={p:.2f} -> B={nn._B} K={nn._K:2d} L={nn._L:2d} | emp_recall={emp:.4f} | lsh={t_q:6.2f} ms vs rust_brute={t_brute:6.2f} ms ({t_brute/t_q:.2f}x)")

print("\n=== 3. S3 crossover band & tune_thresholds (n in 1024..4096) ===")
for d in (8, 16, 32):
    for n in (1024, 2048, 3072, 4096):
        pts = rng.random((n, d))
        q = rng.random((512, d))
        for p in (1.0, 0.5):
            cfg = tune_thresholds(pts, q, k=2*d, p=p, repeats=2, seed=7)
            nn = ToroidalNN(seed=7, p=p).fit(pts[:n//2], pts[n//2:], k=2*d)
            row = {"exp": "s3_band", "d": d, "n": n, "p": p,
                   "is_approx_knn": nn.is_approximate_knn,
                   "is_approx_radius": nn.is_approximate_radius,
                   "tuned_thr_knn": cfg["brute_threshold_knn"],
                   "tuned_thr_radius": cfg["brute_threshold_radius"]}
            rows.append(row)
            print(f"d={d:2d} n={n:4d} p={p:.2f} -> default(knn={nn.is_approximate_knn}, rad={nn.is_approximate_radius}) | tune_thresholds={cfg}")

with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)
PY
        ;;
    *)
        echo "unknown EXP_TASK: $TASK (expected: ${TASKS[*]})" >&2
        exit 2
        ;;
esac

echo "=== done $(date -Is) -> $OUT_FINAL ==="
