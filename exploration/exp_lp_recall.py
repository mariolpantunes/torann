"""Can the L1 hash family retrieve L^p (p<1) neighbours? Measured, not argued.

The grid hash collides with probability ``1 - B*delta`` per sampled
dimension, so over ``K`` dims ``log P(collide) ~ -B * sum_j delta_j`` — a
function of the **L1** distance alone. Two points with the same L1 and very
different ``L^0.5`` are therefore retrieved with the same probability. That
makes "rank the candidates by ``L^p``" a rerank of a set that was never
selected for ``L^p``, and the question is how much that costs.

The diagnostic is rank agreement: take the true ``k`` nearest under one
metric and report where they sit in the *other* metric's ordering. If the
``L^0.5``-nearest sit at L1 rank 5000, no L1-driven candidate set of a few
thousand will contain them and recall against ``L^0.5`` truth is near zero.

Also reported: what an L1 k-NN query of increasing width recovers of the
``L^0.5`` top-5 — the retrieval curve an index would actually have to pay.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import importlib

    import ess
    importlib.import_module("ess.ess")
except ImportError:  # pragma: no cover - exploration-only dependency
    sys.exit("ess not importable: add ~/git/ess/src to PYTHONPATH")

N = 2000
QUERIES = 200
K = 5
WIDTHS = (5, 20, 50, 200, 1000)


def lp_matrix(Q, pts, p):
    """Toroidal ``(sum u^p)`` — the monotone part; the 1/p root does not
    change any ranking, and omitting it keeps the numbers finite at p=0.25.
    """
    acc = np.zeros((Q.shape[0], pts.shape[0]))
    for j in range(Q.shape[1]):
        u = np.abs(Q[:, j, None] - pts[None, :, j])
        np.minimum(u, 1.0 - u, out=u)
        acc += u ** p
    return acc


def cross_ranks(points, p, seed=0):
    """Where each metric's top-K sits in the other metric's ordering."""
    rng = np.random.default_rng(seed)
    qi = rng.choice(points.shape[0], QUERIES, replace=False)
    Q = points[qi]

    D1 = lp_matrix(Q, points, 1.0)
    Dp = lp_matrix(Q, points, p)
    D1[np.arange(len(qi)), qi] = np.inf
    Dp[np.arange(len(qi)), qi] = np.inf

    r1 = np.argsort(np.argsort(D1, axis=1), axis=1)   # L1 rank of every point
    rp = np.argsort(np.argsort(Dp, axis=1), axis=1)   # L^p rank

    top_p = np.argpartition(Dp, K, axis=1)[:, :K]     # true L^p nearest
    top_1 = np.argpartition(D1, K, axis=1)[:, :K]     # true L1 nearest

    l1_rank_of_lp = np.take_along_axis(r1, top_p, axis=1).ravel()
    lp_rank_of_l1 = np.take_along_axis(rp, top_1, axis=1).ravel()

    # What an L1 query of width w recovers of the L^p top-K.
    curve = []
    for w in WIDTHS:
        hit = (np.take_along_axis(r1, top_p, axis=1) < w).mean()
        curve.append((w, float(hit)))
    return l1_rank_of_lp, lp_rank_of_l1, curve


if __name__ == "__main__":
    for dim in (8, 32):
        bounds = np.array([[0.0, 1.0]] * dim)
        pts = ess.esa(np.empty((0, dim)), bounds, n=N, seed=0)
        print(f"\n=== d={dim}, n={N}, ESS-converged, k={K} ===")
        for p in (0.5, 0.25):
            a, b, curve = cross_ranks(pts, p)
            print(f"\np={p}")
            print(f"  L1 rank of the true L^p top-{K}:  "
                  f"median {np.median(a):.0f}, "
                  f"90th pct {np.percentile(a, 90):.0f}, max {a.max():.0f}")
            print(f"  L^p rank of the true L1 top-{K}:  "
                  f"median {np.median(b):.0f}, "
                  f"90th pct {np.percentile(b, 90):.0f}, max {b.max():.0f}")
            print("  recall of L^p top-5 from an L1 query of width w:")
            for w, hit in curve:
                print(f"    w={w:5d}  ->  {100 * hit:5.1f}%  "
                      f"({100 * w / N:.0f}% of the set scanned)")
