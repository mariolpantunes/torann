"""Stability probes for `_tune` and the p <= 1 kernels.

  1. continuity of (B, K, L) as p -> 1
  2. seed stability of (B, K, L) per cell
  3. discontinuity of L and cost across the B boundary
  4. extreme inputs: tiny and huge radii, duplicates, d=1, K > d
  5. native vs python on awkward data for p in a grid

Run: python exploration/exp_tune_stability.py
"""

import collections
import warnings

import numpy as np

from torann import ToroidalNN


def tuned(pts, k=None, radius=None, seed=0, **kw):
    nn = ToroidalNN(seed=seed, brute_threshold=0, backend="python", **kw).fit(pts, k=k, radius=radius)
    return nn._B, nn._K, nn._L


def continuity():
    print("== 1. continuity as p -> 1 (d, n: (B,K,L) per p)")
    for d, n in ((8, 4096), (32, 4096), (32, 32768)):
        pts = np.random.default_rng(d).random((n, d))
        print(d, n, "knn   ", [(p, tuned(pts, k=2 * d, p=p)) for p in (1.0, 1 - 1e-9, 0.99, 0.9)])
        r = d * 0.2
        print(
            d,
            n,
            "radius",
            [(p, tuned(pts, k=2 * d, radius=r * d ** (1 / p - 1), p=p)) for p in (1.0, 1 - 1e-9, 0.99, 0.9)],
        )


def seeds():
    print("== 2. distinct (B,K,L) over 20 data+hash seeds")
    for d in (2, 4, 8, 16, 32, 64):
        for n in (2048, 4096, 16384):
            for p in (1.0, 0.5):
                c = collections.Counter()
                for s in range(20):
                    pts = np.random.default_rng(s).random((n, d))
                    c[tuned(pts, k=min(2 * d, 64), seed=s, p=p)] += 1
                flag = " <-- B flips" if len({x[0] for x in c}) > 1 else ""
                print(f"d={d:3d} n={n:6d} p={p} {dict(c)}{flag}")


def boundary():
    print("== 3. sweep k across the B boundary (d=8, n=4096, p=1)")
    pts = np.random.default_rng(1).random((4096, 8))
    for k in (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128):
        print(k, tuned(pts, k=k))


def extremes():
    print("== 4. extreme inputs")
    warnings.simplefilter("error")
    rng = np.random.default_rng(3)
    cases = {
        "radius=0": (rng.random((512, 8)), {"radius": 0.0}),
        "radius=1e-300": (rng.random((512, 8)), {"radius": 1e-300}),
        "radius=1e3": (rng.random((512, 8)), {"radius": 1e3}),
        "radius=1e308": (rng.random((512, 8)), {"radius": 1e308}),
        "radius=inf": (rng.random((512, 8)), {"radius": float("inf")}),
        "all duplicates": (np.tile(rng.random((1, 8)), (512, 1)), {"k": 16}),
        "two sites": (rng.random((2, 8))[rng.integers(0, 2, 512)], {"k": 16}),
        "d=1": (rng.random((512, 1)), {"k": 2}),
        "d=2 K>d": (rng.random((65536, 2)), {"k": 4}),
        "n=2": (rng.random((2, 8)), {"k": 1}),
        "k>=n": (rng.random((16, 8)), {"k": 64}),
    }
    for name, (pts, kw) in cases.items():
        for p in (1.0, 0.5, 0.25):
            try:
                print(f"{name:16s} p={p} ->", tuned(pts, p=p, **kw))
            except Exception as e:  # noqa: BLE001
                print(f"{name:16s} p={p} -> {type(e).__name__}: {e}")
    warnings.simplefilter("default")


def native():
    print("== 5. native vs python, exact and LSH")
    rng = np.random.default_rng(5)
    d = 16
    sets = {
        "uniform": rng.random((2048, d)),
        "seam": np.mod(rng.normal(0, 1e-3, (2048, d)), 1.0),
        "dyadic": rng.integers(0, 16, (2048, d)) / 16.0,
        "tiny": 0.5 + rng.normal(0, 1e-7, (2048, d)),
    }
    for name, pts in sets.items():
        for p in (1.0, 0.75, 0.5, 0.3, 0.25, 1 - 1e-9):
            out = {}
            for be in ("python", "rust"):
                for thr, tag in ((10**9, "exact"), (0, "lsh")):
                    nn = ToroidalNN(seed=9, brute_threshold=thr, backend=be, p=p).fit(pts, k=8)
                    i, s = nn.query(k=8, queries=pts[:128])
                    r = float(np.median(s[:, -1]))
                    ri, rd = nn.query_radius(r, queries=pts[:128], pad=True)
                    out[be, tag] = (i, s, ri, rd)
            for tag in ("exact", "lsh"):
                a, b = out["python", tag], out["rust", tag]
                ids = np.mean(a[0] != b[0])
                dst = float(np.max(np.abs(a[1] - b[1]) / np.maximum(a[1], 1e-300)))
                rad = a[2].shape == b[2].shape and np.array_equal(a[2], b[2])
                print(f"{name:8s} p={p:<12} {tag:5s} ids_diff={ids:.4f} rel_dist={dst:.2e} radius_same={rad}")


if __name__ == "__main__":
    continuity()
    boundary()
    extremes()
    native()
    seeds()
