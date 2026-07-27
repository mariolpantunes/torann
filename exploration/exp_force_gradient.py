"""Is ESS's force the gradient of ESS's metric? No — and it costs quality.

`ess._compute_forces` takes each neighbour's push *direction* from the
L2-normalised toroidal displacement, `delta / ||delta||_2`, whatever metric
supplied the *magnitudes*. For the L1 potential the gradient is

    d(d_1)/d(delta_j)  =  sign(delta_j)

— every coordinate pushing equally — not `delta_j / ||delta||_2`, which
concentrates the push on the coordinates that already differ most. So the
force ESS applies is not the gradient of any potential it defines, and the
mismatch grows with dimension, where the two directions diverge most.

This measures the size of the error the only way that settles it: hold the
magnitudes, the neighbour lists, the radius and the step rule identical, and
change **only** the direction. `exp_lp_quality.relax` at `p=1` is exactly the
corrected version, since `|delta|^(p-1) * sign(delta)` reduces to
`sign(delta)` there, so the comparison is against `ess.esa` itself.

Scope, because this is easy to overstate. torann's own tests cover
retrieval, byte-identical hash tables and prefix relaxation, and never
evaluate a force, so they are untouched. `NEXT.md` §7's force-law analysis
measured *magnitudes* (rank 1 pushes 0.349, rank 64 pushes 0.299), which
carry no direction, so it is untouched. What the result changes is that
every ESS quality figure on record understates ESS by up to ~0.017 CE, and
comparisons between arms that share the defect keep their ordering.
"""

import argparse
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import ess
    importlib.import_module("ess.ess")
except ImportError:  # pragma: no cover - exploration-only dependency
    sys.exit("ess not importable: add ~/git/ess/src to PYTHONPATH")

import exp_lp_quality as lpq  # noqa: E402  (needs the path set above)

DIMS = (8, 16, 32, 40)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    print("Only the force DIRECTION differs: magnitudes, neighbours, radius")
    print("and step rule are identical. Toroidal Clark-Evans, higher better.")
    print(f"{'d':>4}{'start':>8}{'grad sign(d)':>15}{'ess L2 dir':>13}"
          f"{'diff':>9}{'pooled sd':>11}{'verdict':>16}")
    for d in args.dims:
        bounds = np.array([[0.0, 1.0]] * d)
        for filled in (False, True):
            mine, theirs = [], []
            for s in range(args.seeds):
                rng = np.random.default_rng(1000 * d + s)
                static = rng.random((args.n, d)) if filled else np.empty((0, d))
                pts = rng.random((args.n, d))

                a = lpq.relax(static, pts, 1.0, 0.5, k=args.k, seed=s)
                mine.append(lpq.quality(
                    np.vstack([static, a]) if filled else a)[0])

                b = np.mod(ess.esa(static, bounds, n=args.n,
                                   seed=np.random.default_rng(s), k=args.k), 1.0)
                theirs.append(lpq.quality(
                    np.vstack([static, b]) if filled else b)[0])

            m, t = np.mean(mine), np.mean(theirs)
            pooled = float(np.sqrt((np.var(mine, ddof=1)
                                    + np.var(theirs, ddof=1)) / 2))
            v = ("gradient BETTER" if m - t > 2 * pooled
                 else ("gradient worse" if m - t < -2 * pooled else "same"))
            print(f"{d:>4}{('filled' if filled else 'empty'):>8}{m:>15.4f}"
                  f"{t:>13.4f}{m - t:>+9.4f}{pooled:>11.4f}{v:>16}")


if __name__ == "__main__":
    main()
