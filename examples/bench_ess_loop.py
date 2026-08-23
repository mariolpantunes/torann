"""The ESS inner loop with the trajectory pinned — an A/B that means something.

`bench_ess_suite.py` drives the real `ess.esa` to convergence, which is the
right benchmark for "is the library good" and the wrong one for "is this
change faster": the f32 tie-break decides which neighbour a candidate sees,
ESS then walks a slightly different path, and the two builds converge in
different numbers of epochs. Wall-clock totals stop being comparable —
measured across the Rust 1.98 kernel change, the anchored d=32 shape went
from 105 epochs to 126, so the *faster* build finished the suite slower.

This runs the same call sequence ESS does — `fit(anchors, candidates)`, then
per epoch a self-join `query` and an `update` — for a **fixed** epoch count
with a **fixed** pseudo-random step. Both builds therefore perform exactly
the same work, and the difference is speed and nothing else. It is not a
substitute for the suite (it does not converge, and it does not measure
quality); it is the tool for attributing a speed change.

Shapes are the suite's, minus the one that runs in brute mode.

Run from the repository root::

    python examples/bench_ess_loop.py
    python examples/bench_ess_loop.py --epochs 50 --repeat 5 --json out/loop.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torann import ToroidalNN  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")

# (dim, anchors, candidates) — bench_ess_suite.CASES, minus the brute-mode shape.
CASES = (
    (2, 256, 512),
    (8, 0, 1024),
    (8, 1024, 2048),
    (32, 0, 10000),
    (32, 10000, 20000),
)

K = 5  # ess.K_LOCAL


def run(dim, anchors, candidates, epochs, seed=0):
    """One shape, `epochs` fixed epochs of the ESS loop.

    Returns:
        dict: per-epoch query and update milliseconds, plus a checksum of
        every neighbour returned along the way — two builds that agree on
        it did identical work, and a build that does not is reordering
        ties, not returning different neighbours.
    """
    rng = np.random.default_rng(seed)
    anchor_pts = rng.random((anchors, dim)) if anchors else np.empty((0, dim))
    cands = rng.random((candidates, dim))
    steps = rng.normal(0, 0.01, (epochs, candidates, dim))

    index = ToroidalNN(seed=seed).fit(anchor_pts, cands, k=K)
    t_query = t_update = 0.0
    checksum = 0
    for e in range(epochs):
        t0 = time.perf_counter()
        idx, _ = index.query(k=K)
        t_query += time.perf_counter() - t0
        checksum = (checksum + int(idx.sum())) & 0xFFFFFFFFFFFF

        t0 = time.perf_counter()
        index.update(np.mod(index.candidates + steps[e], 1.0))
        t_update += time.perf_counter() - t0

    return {
        "dim": dim, "anchors": anchors, "candidates": candidates,
        "epochs": epochs,
        "query_ms": t_query / epochs * 1e3,
        "update_ms": t_update / epochs * 1e3,
        "checksum": checksum,
        "backend": index.backend_name or "brute",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20,
                    help="epochs per shape (fixed, not to convergence)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="runs per shape; the fastest is reported")
    ap.add_argument("--cases", type=int, nargs="+",
                    help="1-based case numbers to run (default: all)")
    ap.add_argument("--json", help="write the raw rows here")
    args = ap.parse_args()

    picked = ([CASES[i - 1] for i in args.cases] if args.cases else list(CASES))
    rows = []
    print("| d | anchors | cands | query ms/epoch | update ms/epoch | checksum |")
    print("|" + "---|" * 6)
    for dim, anchors, cands in picked:
        best = min((run(dim, anchors, cands, args.epochs)
                    for _ in range(args.repeat)),
                   key=lambda r: r["query_ms"])
        rows.append(best)
        print(f"| {dim} | {anchors} | {cands} | {best['query_ms']:.2f} "
              f"| {best['update_ms']:.2f} | {best['checksum']} |")

    total = sum(r["query_ms"] for r in rows)
    print(f"\nquery total across shapes: {total:.2f} ms/epoch")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or OUT,
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
