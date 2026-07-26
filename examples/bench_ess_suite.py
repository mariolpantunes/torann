"""The representative ESS benchmark suite: six shapes, driven through `ess`.

torann exists for the Empty-Space-Search loop, so the benchmark that decides
whether an optimization is worth keeping is the *loop*, not a query
microbenchmark: initialize a batch, then per epoch query every candidate
against the whole index, step, `update`, until convergence.

The six shapes span the regimes ESS actually runs, with and without an
anchor tier (an empty one is the from-scratch first batch, which skips
`_smart_init`; a populated one pays for it):

    d=2   anchors 0      candidates 256
    d=2   anchors 256    candidates 512
    d=8   anchors 0      candidates 1024
    d=8   anchors 1024   candidates 2048
    d=32  anchors 0      candidates 10 000
    d=32  anchors 10 000 candidates 20 000

They deliberately cross three thresholds: the brute/LSH crossover (512
points for the native backend, so the smallest shape is *exact*), the
batched-join minimum (64 queries), and `d = 8`, below which the eight-lane
distance kernel has no full vector to work on.

Each shape is reported on its own and the suite is reported as a whole, so a
change cannot look good by trading a big shape against a small one. Every
shape is run twice in the same build — `query_block_size=1` for the
per-query path, the default for the batched join — which makes the
comparison an A/B rather than a claim about history, and lets the harness
assert that both paths return the same neighbours.

Run from the repository root (needs `ess` on the path)::

    python examples/bench_ess_suite.py
    python examples/bench_ess_suite.py --cases 5 6 --repeat 3
    python examples/bench_ess_suite.py --json out/suite.json
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torann import ToroidalNN  # noqa: E402
from torann.brute import exact_knn  # noqa: E402

try:
    import ess
    import ess.utils
except ImportError:  # pragma: no cover - benchmark-only dependency
    sys.exit("ess not importable: add ~/git/ess/src to PYTHONPATH")

OUT = os.path.join(os.path.dirname(__file__), "out")

# (dim, anchors, candidates) — see the module docstring.
CASES = (
    (2, 0, 256),
    (2, 256, 512),
    (8, 0, 1024),
    (8, 1024, 2048),
    (32, 0, 10000),
    (32, 10000, 20000),
)

PHASES = ("setup_s", "query_s", "force_s", "step_s", "update_s")

RECALL_QUERIES = 128


def recall(index, k, seed=0):
    """Mean fraction of the true k nearest neighbours the index returns.

    Measured on the final configuration, against an exact toroidal scan of
    every indexed point. Recall is the product ESS buys from the index, so a
    speed-up that moves it is not a speed-up.

    Args:
        index (ToroidalNN): Index as left by the run.
        k (int): Neighbours per query.
        seed (int): Sub-sample seed.

    Returns:
        float: Recall in [0, 1], or nan when the index is exact anyway.
    """
    if not index.is_approximate:
        return float("nan")
    arena = index._arena
    rng = np.random.default_rng(seed)
    take = min(RECALL_QUERIES, arena.shape[0])
    rows = rng.choice(arena.shape[0], take, replace=False)
    q = np.ascontiguousarray(arena[rows])
    # The sampled queries are themselves indexed, so both sides must exclude
    # the query's own id — otherwise every row scores one free hit and the
    # recall reads (k-1)/k neighbours as k/k.
    ids = np.ascontiguousarray(rows, dtype=np.int64)
    approx, _ = index.query(k=k, queries=q, exclude_ids=ids)
    exact, _ = exact_knn(arena, q, k, exclude_ids=ids)
    hits = sum(len(set(approx[i].tolist()) & set(exact[i].tolist()))
               for i in range(take))
    return hits / float(take * k)


def run(dim, anchors, candidates, block, seed=0):
    """One shape, one query path, through the real ESS loop.

    Args:
        dim (int): Dimensionality.
        anchors (int): Static points; 0 selects the from-scratch path.
        candidates (int): Points ESS generates.
        block (int | None): ``query_block_size`` — 1 for the per-query path,
            None for the batched join.
        seed (int): Seed for the sample and for ESS.

    Returns:
        dict: Timings, phase shares, convergence, quality and index state.
    """
    rng = np.random.default_rng(seed)
    samples = rng.random((anchors, dim)) if anchors else np.empty((0, dim))
    bounds = np.array([[0.0, 1.0]] * dim)
    index = ToroidalNN(seed=seed, query_block_size=block)
    stats: dict = {}

    t0 = time.perf_counter()
    pts = ess.esa(samples, bounds, n=candidates, index=index, seed=seed,
                  stats=stats)
    total = time.perf_counter() - t0

    impl = getattr(index, "_impl", None)
    native = impl.stats() if hasattr(impl, "stats") else {}
    queries = max(1, native.get("queries", 0))
    row = {
        "dim": dim, "anchors": anchors, "candidates": candidates,
        "total_s": total,
        **{p: stats.get(p, 0.0) for p in PHASES},
        "epochs": stats.get("epochs_total"),
        "clark_evans": ess.utils.toroidal_clark_evans(pts),
        "mode": "lsh" if index.is_approximate else "brute",
        "backend": index.backend_name or "brute",
        "tables": index.n_tables,
        "B": getattr(index, "_B", None), "K": getattr(index, "_K", None),
        "recall": recall(index, stats.get("k", 5)),
        "pairs_per_query": native.get("pairs", 0) / queries,
        "path": ("per_query" if native.get("cands") else "batched")
        if native else "brute",
        "fingerprint": int(np.asarray(pts * 1e9, dtype=np.int64).sum()),
    }
    return row


def table(rows, ref=None):
    """Print one row per shape; with `ref`, add the A/B comparison."""
    head = ["d", "anchors", "cands", "mode", "L", "total", "query",
            "setup", "epochs", "CE", "recall"]
    if ref:
        head += ["speed-up", "query x", "same?"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for i, r in enumerate(rows):
        cells = [
            str(r["dim"]), str(r["anchors"]), str(r["candidates"]), r["mode"],
            str(r["tables"]), f"{r['total_s']:.2f}s",
            f"{r['query_s']:.2f}s ({100 * r['query_s'] / r['total_s']:.0f}%)",
            f"{100 * r['setup_s'] / r['total_s']:.0f}%",
            str(r["epochs"]), f"{r['clark_evans']:.4f}",
            "exact" if np.isnan(r["recall"]) else f"{r['recall']:.3f}",
        ]
        if ref:
            b = ref[i]
            # The exact path has no query_block_size to vary, so the two
            # runs are the same code: report n/a rather than dress the
            # run-to-run spread (~15% at these sizes) up as a speed-up.
            comp = ["n/a", "n/a"] if r["mode"] == "brute" else [
                f"{b['total_s'] / r['total_s']:.2f}x",
                f"{b['query_s'] / max(r['query_s'], 1e-9):.2f}x",
            ]
            cells += comp + [
                "yes" if b["fingerprint"] == r["fingerprint"] else "NO",
            ]
        print("| " + " | ".join(cells) + " |")


def group(rows, ref=None, label="suite"):
    """Aggregate the suite: a change has to win on the whole, not on one
    shape. Reports the summed wall time and the geometric mean of the
    per-shape speed-ups, which weights every regime equally."""
    tot = sum(r["total_s"] for r in rows)
    qry = sum(r["query_s"] for r in rows)
    line = (f"{label}: total {tot:.2f}s, query {qry:.2f}s "
            f"({100 * qry / tot:.0f}% of wall)")
    if ref:
        rtot = sum(r["total_s"] for r in ref)
        rqry = sum(r["query_s"] for r in ref)
        geo = statistics.geometric_mean(
            [b["total_s"] / r["total_s"] for b, r in zip(ref, rows)])
        same = all(b["fingerprint"] == r["fingerprint"]
                   for b, r in zip(ref, rows))
        line += (f"  |  vs per-query: {rtot / tot:.2f}x total, "
                 f"{rqry / qry:.2f}x query, geometric mean per shape "
                 f"{geo:.2f}x, results {'identical' if same else 'DIFFER'}")
    print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="+",
                    help="1-based shape indices (default: all)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per shape; the fastest is kept")
    ap.add_argument("--json", help="write the raw rows here")
    args = ap.parse_args()

    picked = [CASES[i - 1] for i in args.cases] if args.cases else list(CASES)

    def best(dim, anchors, cands, block):
        runs = [run(dim, anchors, cands, block) for _ in range(args.repeat)]
        return min(runs, key=lambda r: r["total_s"])

    old = [best(*c, 1) for c in picked]
    new = [best(*c, None) for c in picked]

    print("\n=== per-query path (query_block_size=1) ===")
    table(old)
    group(old, label="suite (per-query)")
    print("\n=== batched join (default) ===")
    table(new, ref=old)
    group(new, ref=old, label="suite (batched) ")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or OUT, exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"per_query": old, "batched": new}, fh, indent=1)
        print(f"\n[saved {args.json}]")
