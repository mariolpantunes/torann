"""What recall costs: tables, probes and hash depth against real timings.

For each cell, try probes x hash depth with the table cap lifted, at the
default recall target, and record (K, L, m), real recall and the query, update
and fit times. The result decides the cap and the `probes` default.

Run on an idle node: python exploration/exp_tune_cost.py --json out.json
"""

import argparse
import json
import statistics
import time

import numpy as np

import torann.wrapper as w
from torann import ToroidalNN
from torann.brute import exact_knn


def workload(d, n, data, rng):
    if data == "uniform":
        return rng.random((n, d))
    centres = rng.random((16, d))
    return np.mod(centres[rng.integers(0, 16, n)] + rng.normal(0.0, 0.02, (n, d)), 1.0)


def run(pts, ids, k, radius, mode, p, m, depth, repeats):
    n = pts.shape[0]
    half = n // 2
    base = ToroidalNN(seed=1, brute_threshold=0, p=p, probes=m)
    base.fit(pts[:half], pts[half:], k=k, radius=None if mode == "knn" else radius)
    nn = ToroidalNN(seed=1, brute_threshold=0, p=p, probes=m, dims_per_table=max(1, base._K + depth))
    t0 = time.perf_counter()
    nn.fit(pts[:half], pts[half:], k=k, radius=None if mode == "knn" else radius)
    fit_s = time.perf_counter() - t0
    Q = pts[half:]
    query_s, update_s = [], []
    step = np.random.default_rng(3).normal(0.0, 0.01, Q.shape)
    for r in range(repeats):
        t0 = time.perf_counter()
        if mode == "knn":
            nn.query(k=k)
        else:
            nn.query_radius(radius, pad=True)
        query_s.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        nn.update(np.mod(Q + (r % 2 == 0) * step, 1.0))
        update_s.append(time.perf_counter() - t0)
    q = pts[ids]
    if mode == "knn":
        got, _ = nn.query(k=k, queries=q, exclude_ids=ids)
        want, _ = exact_knn(nn._arena, q, k, exclude_ids=ids, p=p)
        rec = sum(len(set(a) & set(b)) for a, b in zip(got.tolist(), want.tolist(), strict=True)) / want.size
    else:
        got = nn.query_radius(radius, queries=q)
        want = nn.query_radius(radius, queries=q, exact=True)
        hits = sum(len(set(a.ids.tolist()) & set(b.ids.tolist())) for a, b in zip(got, want, strict=True))
        rec = hits / max(1, sum(b.ids.size for b in want))
    return {
        "K": nn._K,
        "L": nn._L,
        "B": nn._B,
        "m": m,
        "depth": depth,
        "recall": rec,
        "expected": nn.expected_recall,
        "query_ms": 1e3 * statistics.median(query_s),
        "update_ms": 1e3 * statistics.median(update_s),
        "fit_ms": 1e3 * fit_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    w._MAX_TABLES = 96
    rows = []
    for data in ("uniform", "clustered"):
        for mode in ("knn", "radius"):
            for d in (4, 8, 16, 32, 64, 100):
                for n in (4096, 16384, 65536):
                    for p in (1.0, 0.5):
                        rng = np.random.default_rng(d + n)
                        pts = workload(d, n, data, rng)
                        k = min(2 * d, 64)
                        ids = rng.choice(n, 256, replace=False)
                        _, dst = exact_knn(pts, pts[ids[:64]], k, exclude_ids=ids[:64], p=p)
                        radius = float(np.median(dst[:, -1]))
                        for m in (0, 1, 2, 4, 8):
                            for depth in (0, -1, -2):
                                r = run(pts, ids, k, radius, mode, p, m, depth, a.repeats)
                                r.update(data=data, mode=mode, d=d, n=n, p=p)
                                rows.append(r)
                                print(json.dumps(r), flush=True)
                        with open(a.json, "w") as f:
                            json.dump(rows, f)


if __name__ == "__main__":
    main()
