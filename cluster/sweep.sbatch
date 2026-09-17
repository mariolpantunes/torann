#!/usr/bin/env bash
#SBATCH --job-name=torann-sweep
#SBATCH --nodes=1
#SBATCH --output=%x-%A_%a.out
# One array task = one shard of the case list = one whole node.
#
# The shard is the unit of parallelism, and the node is the unit of
# comparison: both arms of every case in this shard are measured here,
# interleaved ABBA, by this task. Splitting the *arms* across tasks instead
# would be faster and worthless -- it would reintroduce exactly the
# between-machine drift the paired design exists to remove.
#
# This Slurm refuses --exclusive (nodes advertise 127 of 128 CPUs usable), so
# the job asks for most of an *idle* node's cores instead -- see config.sh --
# and pins RAYON_NUM_THREADS, so the thread count is a property of the
# experiment rather than of whatever else happened to be running.
set -euo pipefail

ROOT="${BENCH_ROOT:?BENCH_ROOT not set}"
ARM_A="${ARM_A:?}" ARM_B="${ARM_B:?}"
SUITES="${SUITES:-W1 W2 W3 W4 W5}"
PAIRS="${PAIRS:-7}"
SHARDS="${SHARDS:-1}"
OUT_PREFIX="${OUT_PREFIX:?}"
CASES="${CASES:-}"

export PATH="$ROOT/venv/bin:$PATH"
export TORANN_ARMS="$ROOT/arms"
export RAYON_NUM_THREADS="${RAYON_THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc)}}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

TASK="${SLURM_ARRAY_TASK_ID:-0}"
OUT="${OUT_PREFIX}.shard${TASK}.json"

cd "$ROOT/torann"
echo "node $(hostname) | cpus=$RAYON_NUM_THREADS | shard $TASK/$SHARDS | $ARM_A vs $ARM_B"
grep -m1 'model name' /proc/cpuinfo || true

exec python examples/bench_ab.py run \
    --arms "$ARM_A" "$ARM_B" \
    --suites $SUITES \
    ${CASES:+--cases "$CASES"} \
    --pairs "$PAIRS" \
    --threads "$RAYON_NUM_THREADS" \
    --shard "$TASK/$SHARDS" \
    --out "$OUT"
