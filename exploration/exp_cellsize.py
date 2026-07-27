"""Can a grid cell be sized so that the query only ever touches 1 + 2d cells?

The design being tested: size cells from `n` and the force law, look in the
query's own cell plus the face-adjacent ones, and rely on the force law to
make everything outside that neighbourhood irrelevant. For toroidal L1 the
neighbourhood is the von Neumann one (1 + 2d cells: 5 at d=2), not the
Moore one — the diagonal cells are further in L1 than the faces and carry
no weight the faces do not already have.

Two things have to hold at once:

1. **Capture.** Everything within the interaction radius R must fall inside
   the +/-1 neighbourhood. A point within toroidal-L1 distance R differs
   from the query by at most R in any single coordinate (L-inf <= L1), so a
   cell width w >= R is sufficient — and, for the worst-case single-axis
   point, necessary.
2. **Selectivity.** The cells must hold few enough points to be worth
   visiting: occupancy c = n / B^d for B = 1/w cells per dimension.

These pull in opposite directions, and this script reports where they cross
for the shapes ESS actually runs. It also measures the premise directly:
what share of the net force a point receives comes from its j-th nearest
neighbour, so "outside the range means almost no impact" is a number rather
than an assumption.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torann.brute import exact_knn  # noqa: E402

try:
    import importlib

    import ess
    # `ess.ess` the attribute is the public *function*, not the module, so
    # the private helpers have to be reached through importlib.
    essmod = importlib.import_module("ess.ess")
except ImportError:  # pragma: no cover - exploration-only dependency
    sys.exit("ess not importable: add ~/git/ess/src to PYTHONPATH")

# The suite's shapes, as (dim, total points).
SHAPES = ((2, 512), (2, 768), (4, 2048), (8, 2048), (8, 3072),
          (16, 2048), (32, 4000), (32, 10000), (32, 30000), (64, 2048))

OCCUPANCY = 5.0  # points per cell we would want to visit


def capture_vs_selectivity():
    """Where the two requirements cross, per shape."""
    print("R          = ESS's interaction radius (toroidal L1, its own "
          "heuristic)")
    print("B_capture  = cells/dim allowed if +/-1 must cover R   (1/R)")
    print("B_occupy   = cells/dim needed for ~5 points per cell  ((n/c)^1/d)")
    print("cells      = 1 + 2d, the von Neumann neighbourhood\n")
    head = ("d", "n", "R", "R/d", "B_capture", "B_occupy", "cells",
            "occupancy at B_capture", "verdict")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for dim, n in SHAPES:
        R = essmod._l1_radius_heuristic(dim, n)
        b_cap = 1.0 / R
        b_occ = (n / OCCUPANCY) ** (1.0 / dim)
        # Points per cell if we take the largest grid that still captures.
        b_use = max(int(math.floor(b_cap)), 1)
        occ = n / (b_use ** dim) if b_use ** dim < 1e300 else 0.0
        ok = b_cap >= b_occ
        verdict = "works" if ok else ("marginal" if b_cap >= 0.8 * b_occ
                                      else "impossible")
        print(f"| {dim} | {n} | {R:.4f} | {R / dim:.4f} | {b_cap:.2f} "
              f"| {b_occ:.2f} | {1 + 2 * dim} | {occ:.3g} | {verdict} |")


def force_share_by_rank(dim, n, kmax=64, seed=0):
    """Share of the net force weight contributed by each neighbour rank.

    Runs ESS to convergence, then re-derives the force weights its own
    kernel would apply — `exp(log f(d/R))`, the same `metric_fn` — for the
    `kmax` nearest neighbours of every point. Reports the cumulative share,
    which is what "points outside the range barely matter" has to mean
    quantitatively.
    """
    bounds = np.array([[0.0, 1.0]] * dim)
    stats: dict = {}
    pts = ess.esa(np.empty((0, dim)), bounds, n=n, seed=seed, stats=stats)
    R = stats["radius"]

    _, dst = exact_knn(np.ascontiguousarray(pts), np.ascontiguousarray(pts),
                       kmax + 1)
    dst = dst[:, 1:]  # drop self
    w = np.exp(essmod.gaussian_force(dst / R))
    share = w.mean(axis=0)
    cum = np.cumsum(share) / share.sum()

    inside = float((dst <= R).sum(axis=1).mean())
    print(f"\nd={dim}, n={n}, R={R:.4f}: mean neighbours inside R = "
          f"{inside:.1f} (of {kmax} examined)")
    print("| rank | distance | d/R | force weight | cumulative share |")
    print("|---|---|---|---|---|")
    for j in list(range(5)) + [7, 9, 14, 19, 29, 49, kmax - 1]:
        if j >= kmax:
            continue
        print(f"| {j + 1} | {dst[:, j].mean():.4f} "
              f"| {dst[:, j].mean() / R:.2f} | {share[j]:.4g} "
              f"| {100 * cum[j]:.2f}% |")


if __name__ == "__main__":
    capture_vs_selectivity()
    for dim, n in ((2, 512), (8, 2048), (32, 4000)):
        force_share_by_rank(dim, n)
