"""Default (B, K, L), real recall and query time on a workload grid.

Run once per build and compare the two JSON files:
    python exploration/exp_tune_grid.py --json before.json
    python exploration/exp_tune_grid.py --json after.json --compare before.json
"""

import argparse
import json
import statistics
import time

import numpy as np

from torann import ToroidalNN
from torann.brute import exact_knn


def workload(d, n, data, rng):
    if data == "uniform":
        return rng.random((n, d))
    centres = rng.random((16, d))
    return np.mod(centres[rng.integers(0, 16, n)] + rng.normal(0.0, 0.02, (n, d)), 1.0)


def cell(d, n, p, data, mode, seed, m=500, **kw):
    rng = np.random.default_rng(seed)
    pts = workload(d, n, data, rng)
    k = min(2 * d, 64)
    ids = rng.choice(n, m, replace=False)
    Q = pts[ids]
    want, dst = exact_knn(pts, Q, k, exclude_ids=ids, p=p)
    radius = float(np.median(dst[:, k - 1]))
    nn = ToroidalNN(seed=seed, brute_threshold=0, p=p, **kw)
    t0 = time.perf_counter()
    if mode == "knn":
        nn.fit(pts, k=k)
    else:
        nn.fit(pts, k=k, radius=radius)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if mode == "knn":
        got, _ = nn.query(k=k, queries=Q, exclude_ids=ids)
        query_s = time.perf_counter() - t0
        recall = np.mean([len(set(a) & set(b)) / k for a, b in zip(got, want, strict=True)])
    else:
        got = nn.query_radius(radius, queries=Q)
        query_s = time.perf_counter() - t0
        ref = nn.query_radius(radius, queries=Q, exact=True)
        hits = sum(len(set(g.ids.tolist()) & set(r.ids.tolist())) for g, r in zip(got, ref, strict=True))
        recall = hits / max(1, sum(r.ids.size for r in ref))
    return nn._B, nn._K, nn._L, nn.probes, float(recall), 1e3 * query_s, 1e3 * fit_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    rows = {}
    for data in ("uniform", "clustered"):
        for mode in ("knn", "radius"):
            for d in (4, 8, 16, 32, 64):
                for n in (4096, 16384):
                    for p in (1.0, 0.75, 0.5, 0.25):
                        runs = [cell(d, n, p, data, mode, s) for s in range(a.seeds)]
                        rows[f"{data}/{mode}/d{d}/n{n}/p{p}"] = {
                            "B": runs[0][0],
                            "K": runs[0][1],
                            "L": statistics.median(r[2] for r in runs),
                            "probes": runs[0][3],
                            "recall": statistics.mean(r[4] for r in runs),
                            "query_ms": statistics.median(r[5] for r in runs),
                            "fit_ms": statistics.median(r[6] for r in runs),
                        }
    with open(a.json, "w") as f:
        json.dump(rows, f, indent=1)
    if a.compare:
        with open(a.compare) as f:
            old = json.load(f)
        drops = []
        for key, r in rows.items():
            o = old[key]
            drops.append((r["recall"] - o["recall"], key, o, r))
        drops.sort(key=lambda x: x[0])
        print("worst recall changes")
        for dr, key, o, r in drops[:15]:
            print(
                f"{key:34s} L {o['L']:4.0f}->{r['L']:4.0f} rec {o['recall']:.3f}->{r['recall']:.3f}"
                f" q {o['query_ms']:7.1f}->{r['query_ms']:7.1f}"
            )
        ratio = sorted(r["query_ms"] / o["query_ms"] for _, _, o, r in drops)
        lratio = sorted(r["L"] / o["L"] for _, _, o, r in drops)
        print(f"cells {len(drops)} recall delta min {drops[0][0]:+.3f} median {drops[len(drops) // 2][0]:+.3f}")
        print(f"query ratio median {ratio[len(ratio) // 2]:.2f}, L ratio median {lratio[len(lratio) // 2]:.2f}")


if __name__ == "__main__":
    main()
