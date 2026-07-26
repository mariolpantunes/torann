"""Is the index's recall good enough for ESS? Answer it in ESS's own metrics.

`examples/bench_ess_suite.py` measures speed on the six representative
shapes and reports the recall the tuned LSH parameters deliver. At d=32
that recall is **0.69** (from an empty anchor tier) and **0.86** (with
anchors) — well short of 1.0. Recall is a property of the index; it is not
what ESS is for. This script asks the question that matters instead:

1. **Does the result still disperse?** Every shape is scored with the
   metrics ESS is benchmarked on — toroidal Clark-Evans (1.0 = Poisson,
   higher = more regular), minimum pairwise distance, and grid coverage
   where the dimension allows it — against two baselines on the same
   points: Latin-hypercube and uniform random. If ESS at recall 0.69 does
   not beat LHS, the index is not delivering exploration, whatever its
   timings say.
2. **Is recall the thing holding quality back?** The d=32 shapes are re-run
   with the index forced to a higher-recall configuration (more tables,
   more probes). If the metrics do not move, recall 0.69 is *sufficient*
   for this workload and buying more of it is wasted work; if they do move,
   the tuner is under-provisioning at d=32 and that is a correctness-shaped
   bug, not an optimization.

Both questions are asked with anchors and without, because they are
different jobs: from an empty tier ESS is a space-filling sampler competing
with LHS, while with anchors it has to place points in the gaps left by
points it cannot move.

Run from the repository root (needs `ess` on the path)::

    python examples/bench_ess_quality.py
    python examples/bench_ess_quality.py --cases 5 6 --seeds 3
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torann import ToroidalNN  # noqa: E402

try:
    import ess
    import ess.samplers
    import ess.utils
except ImportError:  # pragma: no cover - benchmark-only dependency
    sys.exit("ess not importable: add ~/git/ess/src to PYTHONPATH")

OUT = os.path.join(os.path.dirname(__file__), "out")

# Same shapes as bench_ess_suite.py: (dim, anchors, candidates).
CASES = (
    (2, 0, 256),
    (2, 256, 512),
    (8, 0, 1024),
    (8, 1024, 2048),
    (32, 0, 10000),
    (32, 10000, 20000),
)

# A higher-recall index for question 2: the tuner's L and probes are what
# set recall, so both are raised and nothing else changes.
HIGH_RECALL = {"num_tables": 48, "probes": 8}

# Points above which the O(n^2) discrepancy is skipped.
DISCREPANCY_MAX = 4096


def metrics(points, dim):
    """ESS's dispersion metrics for one point set on the unit torus.

    Args:
        points (np.ndarray): (n, d) points in [0, 1).
        dim (int): Dimensionality.

    Returns:
        dict: Clark-Evans (toroidal), minimum pairwise distance, grid
        coverage where the cell count is tractable, and wrap-around
        discrepancy for small sets.
    """
    n = points.shape[0]
    out = {
        "clark_evans": float(ess.utils.toroidal_clark_evans(points)),
        "min_dist": float(ess.utils.calculate_min_pairwise_distance(points)),
    }
    # Coverage needs a grid; 2 cells per dimension is already 2^d cells, so
    # only ask where that is countable.
    grid = 16 if dim <= 2 else (4 if dim <= 4 else 2)
    if grid ** dim <= 1 << 20:
        bounds = np.array([[0.0, 1.0]] * dim)
        out["coverage"] = float(
            ess.utils.calculate_grid_coverage(points, bounds, grid))
    if n <= DISCREPANCY_MAX:
        out["discrepancy"] = float(ess.utils.wrap_around_discrepancy(points))
        out["discrepancy_expected"] = float(
            ess.utils.expected_discrepancy(n, dim))
    return out


def anchors_of(anchors, dim, seed):
    rng = np.random.default_rng(seed)
    return rng.random((anchors, dim)) if anchors else np.empty((0, dim))


def run_ess(dim, anchors, candidates, seed, index_kwargs=None):
    """ESS's own output for one shape, plus the recall its index delivered."""
    A = anchors_of(anchors, dim, seed)
    bounds = np.array([[0.0, 1.0]] * dim)
    index = ToroidalNN(seed=seed, **(index_kwargs or {}))
    stats: dict = {}
    pts = ess.esa(A, bounds, n=candidates, index=index, seed=seed, stats=stats)
    from bench_ess_suite import recall  # same measurement, self excluded
    return (np.vstack([A, pts]) if anchors else pts,
            {"recall": recall(index, 5), "tables": index.n_tables,
             "epochs": stats.get("epochs_total"),
             "mode": "lsh" if index.is_approximate else "brute"})


def run_baseline(kind, dim, anchors, candidates, seed):
    """The same number of points from a sampler, with the anchors kept."""
    A = anchors_of(anchors, dim, seed)
    rng = np.random.default_rng(seed + 1)
    sampler = (ess.samplers.LHCSampler() if kind == "lhs"
               else ess.samplers.UniformSampler())
    new = np.mod(sampler.sample(candidates, dim, rng).astype(np.float64), 1.0)
    return np.vstack([A, new]) if anchors else new


def report(rows):
    keys = ["clark_evans", "min_dist", "coverage", "discrepancy"]
    head = ["d", "anchors", "cands", "variant", "recall", "CE", "min_dist",
            "coverage", "discrepancy"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        cells = [str(r["dim"]), str(r["anchors"]), str(r["candidates"]),
                 r["variant"],
                 "-" if r.get("recall") is None or np.isnan(r["recall"])
                 else f"{r['recall']:.3f}"]
        for key in keys:
            v = r["metrics"].get(key)
            cells.append("-" if v is None else f"{v:.4f}")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="+")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--json", default=os.path.join(OUT, "quality.json"))
    args = ap.parse_args()
    picked = [(i + 1, CASES[i]) for i in range(len(CASES))]
    if args.cases:
        picked = [(i, CASES[i - 1]) for i in args.cases]

    rows = []
    for idx, (dim, anchors, cands) in picked:
        for seed in range(args.seeds):
            pts, info = run_ess(dim, anchors, cands, seed)
            rows.append({"dim": dim, "anchors": anchors, "candidates": cands,
                         "variant": "ess", "seed": seed, **info,
                         "metrics": metrics(pts, dim)})
            for kind in ("lhs", "random"):
                b = run_baseline(kind, dim, anchors, cands, seed)
                rows.append({"dim": dim, "anchors": anchors,
                             "candidates": cands, "variant": kind,
                             "seed": seed, "recall": None,
                             "metrics": metrics(b, dim)})
            if dim >= 32:  # question 2: does more recall buy quality?
                pts, info = run_ess(dim, anchors, cands, seed, HIGH_RECALL)
                rows.append({"dim": dim, "anchors": anchors,
                             "candidates": cands,
                             "variant": "ess-high-recall", "seed": seed,
                             **info, "metrics": metrics(pts, dim)})
            print(f"  [shape {idx}: d={dim} {anchors}+{cands} seed={seed} done]",
                  flush=True)

    print()
    report(rows)
    os.makedirs(OUT, exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n[saved {args.json}]")
