"""Brute-force vs LSH crossover: where does the index start paying?

For each dimension and dataset size, measures per-query time of the exact
blocked-NumPy brute path against each LSH backend (forced on via
brute_threshold), plus LSH recall and the break-even query count
q* = fit_time / (brute - lsh per-query saving). The crossover n*(d, backend)
is the smallest n where LSH answers faster than brute *and* recall >= 0.95.

Run from the repository root:
  python examples/crossover.py [--dims 8,16,32] [--backends python,rust]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torann import (
    ToroidalNN,
    available_backends,
)

N_LIST = [500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000]
M_Q = 1_000
REPEAT = 3


def bench_queries(nn, Q, k):
    t0 = time.perf_counter()
    idx, dst = nn.query(k=k, queries=Q)
    best = time.perf_counter() - t0
    for _ in range(REPEAT - 1):
        t0 = time.perf_counter()
        idx, dst = nn.query(k=k, queries=Q)
        best = min(best, time.perf_counter() - t0)
    return idx, dst, 1e6 * best / Q.shape[0]


def bench_radius(nn, Q, radius):
    t0 = time.perf_counter()
    res = nn.query_radius(radius, queries=Q, pad=True)
    best = time.perf_counter() - t0
    for _ in range(REPEAT - 1):
        t0 = time.perf_counter()
        res = nn.query_radius(radius, queries=Q, pad=True)
        best = min(best, time.perf_counter() - t0)
    return res.ids, 1e6 * best / Q.shape[0]


def radius_recall(got_ids, exact_ids):
    hits = 0
    total = 0
    for g, e in zip(got_ids, exact_ids):
        ev = e[e >= 0]
        if ev.size:
            gv = g[g >= 0]
            hits += len(np.intersect1d(gv, ev))
            total += ev.size
    return hits / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="8,16,32")
    ap.add_argument("--backends", default=",".join(available_backends()))
    ap.add_argument("--modes", default="knn,radius")
    ap.add_argument("--ps", default="1.0,0.5")
    args = ap.parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    backends = args.backends.split(",")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    ps = [float(x) for x in args.ps.split(",") if x.strip()]

    header = f"{'mode':>6} {'p':>4} {'d':>3} {'n':>7} {'brute us/q':>11} " + " ".join(
        f"{b + ' us/q':>12} {b + ' rec':>8}" for b in backends
    )
    print(header)
    print("-" * len(header))
    results = []
    for mode in modes:
        for p in ps:
            for d in dims:
                k = 2 * d
                for n in N_LIST:
                    rng = np.random.default_rng(0)
                    pts = rng.random((n, d))
                    Q = rng.random((M_Q, d))

                    brute = ToroidalNN(seed=1, brute_threshold=10**12, p=p).fit(pts, k=k)
                    exact_idx, exact_dst, brute_knn_us = bench_queries(brute, Q, k)
                    r_cut = float(np.median(exact_dst[:, min(k - 1, exact_dst.shape[1] - 1)]))
                    if mode == "knn":
                        brute_us = brute_knn_us
                    else:
                        exact_r_ids, brute_us = bench_radius(brute, Q, r_cut)

                    row = {"mode": mode, "p": p, "d": d, "n": n, "k": k, "brute_us": round(brute_us, 1)}
                    cells = []
                    for backend in backends:
                        nn = ToroidalNN(seed=1, brute_threshold=0, backend=backend, p=p)
                        t0 = time.perf_counter()
                        if mode == "knn":
                            nn.fit(pts, k=k)
                            fit_s = time.perf_counter() - t0
                            idx, _, lsh_us = bench_queries(nn, Q, k)
                            hits = sum(len(np.intersect1d(idx[i], exact_idx[i])) for i in range(M_Q))
                            rec = hits / exact_idx.size
                        else:
                            nn.fit(pts, k=k, radius=r_cut)
                            fit_s = time.perf_counter() - t0
                            r_ids, lsh_us = bench_radius(nn, Q, r_cut)
                            rec = radius_recall(r_ids, exact_r_ids)
                        saving = brute_us - lsh_us
                        row[backend] = {
                            "fit_s": round(fit_s, 4),
                            "us_per_q": round(lsh_us, 1),
                            "speedup": round(brute_us / max(lsh_us, 1e-6), 2),
                            "recall": round(rec, 3),
                            "breakeven_queries": (int(np.ceil(1e6 * fit_s / saving)) if saving > 0 else None),
                        }
                        cells.append(f"{lsh_us:>12.1f} {rec:>8.3f}")
                    results.append(row)
                    print(f"{mode:>6} {p:>4.2f} {d:>3} {n:>7} {brute_us:>11.1f} " + " ".join(cells))

    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "crossover.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults saved to {out_dir}/crossover.json")


if __name__ == "__main__":
    main()
