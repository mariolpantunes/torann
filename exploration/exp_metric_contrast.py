"""Two ways out of the high-d force collapse: a sharper law, or a lower p.

At d=32 the 64 nearest neighbours of a converged ESS point all sit within
3% of each other in toroidal L1, so the Gaussian force evaluates to nearly
the same weight for all of them (rank 64 pushes 86% as hard as rank 1).
The force law stops discriminating, which means a wrongly-returned far
neighbour votes at nearly full strength. Two independent fixes:

1. **A sharper force.** The spread is small but not zero, so a steep
   enough law still separates rank 1 from rank 64. This computes what
   steepness is required — as a `sigma/R` for the Gaussian and as an
   exponent for a power law — and what it costs in dynamic range.
2. **A lower p.** Toroidal `L^p` for `p < 1` is known to keep more
   relative contrast in high dimension. `requirements.md` §7 settled on
   L1 "deliberately", noting the grid-LSH family provably covers
   `p in (0, 1]` — so the hash needs no change. What p<1 gives up is the
   triangle inequality, which this index only ever used for the pruning
   bound that was already measured useless at d >= 4.

Contrast is reported as `(r_k - r_1) / r_1` over the k nearest: the
relative spread the force law has to work with. Bigger is better.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import importlib

    import ess
    essmod = importlib.import_module("ess.ess")
except ImportError:  # pragma: no cover - exploration-only dependency
    sys.exit("ess not importable: add ~/git/ess/src to PYTHONPATH")

DIMS = (8, 16, 32, 64)
POWERS = (1.0, 0.5, 0.25)
N = 2000
QUERIES = 400
KMAX = 64


def toroidal_lp(Q, pts, p):
    """``(sum_i u_i^p)^(1/p)`` with ``u`` the per-axis toroidal gap.

    For ``p = 1`` this is the index's own metric. For ``p < 1`` it is a
    quasi-norm — the ranking it induces is still well defined, which is all
    a k-NN index and a force magnitude need.
    """
    out = np.empty((Q.shape[0], pts.shape[0]))
    acc = np.zeros_like(out)
    for j in range(Q.shape[1]):
        u = np.abs(Q[:, j, None] - pts[None, :, j])
        np.minimum(u, 1.0 - u, out=u)
        acc += u ** p
    np.power(acc, 1.0 / p, out=out)
    return out


def contrast(points, p, kmax=KMAX, seed=0):
    """Relative spread over the kmax nearest, averaged over queries."""
    rng = np.random.default_rng(seed)
    qi = rng.choice(points.shape[0], min(QUERIES, points.shape[0]),
                    replace=False)
    D = toroidal_lp(points[qi], points, p)
    D[np.arange(len(qi)), qi] = np.inf
    part = np.partition(D, kmax, axis=1)[:, :kmax]
    part.sort(axis=1)
    r1, rk = part[:, 0], part[:, kmax - 1]
    return float(np.mean((rk - r1) / r1)), float(r1.mean()), float(rk.mean())


def force_sharpness(r1, rk, ratio=10.0):
    """Steepness needed for the nearest neighbour to outvote rank k by `ratio`.

    ``F(r_1) / F(r_k) = ratio`` is the discrimination the force law has to
    supply. Returned as the Gaussian ``sigma`` in units of ``r_1`` (ESS
    expresses it in units of R, so compare against its default 0.5) and as
    the exponent of an equivalent power law.
    """
    # Gaussian: F ~ exp(-r^2 / 2s^2)  ->  s^2 = (rk^2 - r1^2) / (2 ln ratio)
    s2 = (rk * rk - r1 * r1) / (2.0 * math.log(ratio))
    sigma = math.sqrt(s2) if s2 > 0 else float("nan")
    # Power law: F ~ r^-q  ->  q = ln ratio / ln(rk / r1)
    q = math.log(ratio) / math.log(rk / r1) if rk > r1 else float("inf")
    return sigma / r1, q


def run(points, label):
    print(f"\n=== {label} ===")
    print("| d | p | r_1 | r_64 | contrast (r64-r1)/r1 | sigma/r_1 for "
          "10:1 | power-law exponent for 10:1 |")
    print("|" + "---|" * 7)
    for dim in DIMS:
        pts = points(dim)
        for p in POWERS:
            c, r1, rk = contrast(pts, p)
            sig, q = force_sharpness(r1, rk)
            print(f"| {dim} | {p} | {r1:.4f} | {rk:.4f} | {100 * c:.1f}% "
                  f"| {sig:.3f} | {q:.0f} |")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    run(lambda d: rng.random((N, d)), "uniform random points")

    cache: dict = {}

    def converged(d):
        if d not in cache:
            bounds = np.array([[0.0, 1.0]] * d)
            cache[d] = ess.esa(np.empty((0, d)), bounds, n=N, seed=0)
        return cache[d]

    run(converged, "ESS-converged points (the actual regime)")
