"""P2b: does toroidal `L^p` for `p < 1` actually improve ESS's point sets?

`NEXT.md` §7.2 measured that lower `p` restores the *contrast* the force law
loses above `d ~ 8` — at d=32 the relative spread over 64 neighbours goes
2.9% at `p=1` to 20.1% at `p=0.25`. Nothing has tested whether that contrast
converts into a better point set, and `RESEARCH.md` §1 has an exact toroidal
`L^p` LSH family ready to build in Rust whose entire value depends on the
answer. This gate costs a day and no index work.

It is a gate rather than a formality because the inference has already been
shown to fail elsewhere. Mirkes, Allohibi & Gorban (*Entropy* 2020) confirm
fractional quasinorms have greater relative contrast and then show it does
**not** translate into kNN classification quality, with `p = 0.5, 1, 2`
differing insignificantly.

**Design.**

* Brute force throughout, `n <= 512`, so no index is involved and the metric
  is the only variable. This also sidesteps §7.3's finding that an L1 hash
  reranked by `L^p` is capped at 93.2% recall — neighbour *selection* here is
  exact under whichever `p` is being tested, so a retrieval artifact cannot
  be mistaken for a metric effect.
* `p` x `sigma` as a full factorial, because §7.2 also shows the required
  force sharpness relaxes as `p` falls (`sigma/r1` 0.113 -> 0.309). Without
  the second factor "lower p helps" and "sharper sigma helps" are
  inseparable, and sharpening sigma is free where a new hash family is not.
* ESS's own force kernel is called directly — `_compute_forces` takes the
  distance array as an argument, so only neighbour selection and `dists` are
  replaced. The relaxation being measured is ESS's, not a reimplementation.
* The interaction radius is re-derived per `p` as the median k-th nearest
  `L^p` distance. ESS's `_l1_radius_heuristic` inverts an exact **L1**
  ball-volume formula that has no closed form here, and an empirical radius
  is metric-correct and assumption-free. It matters: the force sees `d_p/R`,
  so an L1 radius with `L^p` distances would mis-scale the whole kernel.

**The validation gate comes first.** At `p=1` this loop must reproduce
`ess.esa`'s Clark-Evans within between-seed noise. If it does not, the
harness is measuring itself and every number below is void.

**Decision rule, fixed in advance.** `p=0.5` at tuned `sigma` must beat
`p=1` at tuned `sigma` by more than the between-seed sd (0.0037 CE at d=32).
Otherwise the `L^p` family is descoped and the effort moves to the sharper
force law, which is ESS-side and cheap.
"""

import argparse
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import ess
    essmod = importlib.import_module("ess.ess")
    esutils = importlib.import_module("ess.utils")
except ImportError:  # pragma: no cover - exploration-only dependency
    sys.exit("ess not importable: add ~/git/ess/src to PYTHONPATH")

PS = (1.0, 0.5, 0.25)
SIGMAS = (2.0, 1.0, 0.5, 0.25, 0.113)
DIMS = (8, 16, 32)


def lp_pairwise(a, b, p):
    """Toroidal `L^p` quasi-distance, `(sum |wrap(delta)|^p)^(1/p)`.

    The wrap is per axis into [-1/2, 1/2), the same convention the force
    kernel uses for direction, so distance and direction stay consistent.
    """
    d = np.abs(a[:, None, :] - b[None, :, :])
    d = np.minimum(d, 1.0 - d)
    if p == 1.0:
        return d.sum(-1)
    return np.power(np.power(d, p).sum(-1), 1.0 / p)


def knn_lp(active, all_data, k, p, self_offset):
    """Exact `L^p` k-NN of each active point against everything, self excluded."""
    D = lp_pairwise(active, all_data, p)
    D[np.arange(active.shape[0]), np.arange(len(D)) + self_offset] = np.inf
    idx = np.argpartition(D, k, axis=1)[:, :k]
    dst = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dst, axis=1)
    return np.take_along_axis(idx, order, 1), np.take_along_axis(dst, order, 1)


def radius_lp(points, k, p):
    """Empirical interaction radius: median k-th nearest `L^p` distance."""
    D = lp_pairwise(points, points, p)
    np.fill_diagonal(D, np.inf)
    kk = min(k, D.shape[0] - 1)
    return float(np.median(np.partition(D, kk - 1, axis=1)[:, kk - 1]))


def relax(static, active, p, sigma, k=5, epochs=1024, lr=0.5, decay=0.99,
          tol=1e-2, patience=25, seed=0):
    """ESS's relaxation with the metric swapped. Returns the moved points."""
    rng = np.random.default_rng(seed)
    active = np.mod(np.array(active, dtype=float), 1.0)
    n_static = static.shape[0]
    radius = radius_lp(np.vstack([static, active]) if n_static else active, k, p)

    current_lr, ema, best_ema, calm = lr, None, np.inf, 0
    for _ in range(epochs):
        all_data = np.vstack([static, active]) if n_static else active
        ids, dists = knn_lp(active, all_data, k, p, n_static)
        forces = essmod._compute_forces(
            active, all_data, ids, dists, radius,
            essmod.gaussian_force, rng, sigma=sigma)

        step = forces * (current_lr * radius)
        step_norm = np.abs(step).sum(axis=1, keepdims=True)
        step *= np.minimum(
            1.0, essmod.STEP_CAP * radius / np.maximum(step_norm, 1e-12))
        active = np.mod(active + step, 1.0)

        f_max = float(np.max(np.linalg.norm(forces, axis=1)))
        ema = f_max if ema is None else 0.5 * ema + 0.5 * f_max
        if ema < best_ema * 0.99:
            best_ema, calm = ema, 0
        else:
            calm += 1
        if ema < tol or calm >= patience or current_lr * ema < 1e-9:
            break
        current_lr *= decay
    return active


def quality(points):
    return (esutils.toroidal_clark_evans(points),
            esutils.toroidal_separation(points))


def validate(d, n, seeds, k):
    """Gate: at p=1 this loop must match `ess.esa` within between-seed noise."""
    print(f"\n=== validation at p=1, d={d}, n={n} — must match ess.esa ===")
    # `esa(samples, bounds, n)` anchors to `samples` and produces `n` NEW
    # points, so the matching call for a relaxation from empty passes an
    # empty anchor set. Comparing against `esa(pts, ..., n=len(pts))` would
    # be a filled 2n-point run against an empty n-point one -- a different
    # problem, and the first thing this gate caught.
    mine, theirs = [], []
    bounds = np.array([[0.0, 1.0]] * d)
    for s in range(seeds):
        rng = np.random.default_rng(s)
        pts = rng.random((n, d))
        mine.append(quality(relax(np.empty((0, d)), pts, 1.0, 0.5,
                                  k=k, seed=s))[0])
        theirs.append(quality(np.mod(
            ess.esa(np.empty((0, d)), bounds, n=n,
                    seed=np.random.default_rng(s), k=k),
            1.0))[0])
    m, t = np.mean(mine), np.mean(theirs)
    sd = np.std(theirs, ddof=1) if seeds > 1 else float("nan")
    ok = abs(m - t) <= max(3 * sd, 0.02)
    print(f"  this harness CE {m:.4f}   ess.esa CE {t:.4f}   "
          f"diff {m - t:+.4f}   between-seed sd {sd:.4f}")
    print(f"  VALIDATION {'PASSED' if ok else 'FAILED — numbers below are void'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--filled", action="store_true",
                    help="anchor against an equal number of static points, "
                         "the case ESS is actually used in")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    if not args.skip_validation:
        if not validate(args.dims[0], args.n, args.seeds, args.k):
            sys.exit(1)

    for d in args.dims:
        print(f"\n=== d={d}, n={args.n}, "
              f"{'filled' if args.filled else 'empty'} start, "
              f"{args.seeds} seeds — Clark-Evans (higher = more regular) ===")
        print(f"{'p':>6}" + "".join(f"{'sigma=' + str(s):>14}" for s in SIGMAS))
        base = None
        for p in PS:
            row = f"{p:>6}"
            for sigma in SIGMAS:
                ce = []
                for s in range(args.seeds):
                    rng = np.random.default_rng(1000 * d + s)
                    if args.filled:
                        static = rng.random((args.n, d))
                    else:
                        static = np.empty((0, d))
                    pts = rng.random((args.n, d))
                    out = relax(static, pts, p, sigma, k=args.k, seed=s)
                    ce.append(quality(out)[0])
                m = float(np.mean(ce))
                if p == 1.0 and sigma == SIGMAS[0]:
                    base = m
                row += f"{m:>14.4f}"
            print(row)
        print(f"  reference: p=1, sigma=0.5 -> {base:.4f}; "
              f"decision threshold is +0.0037 CE over the best p=1 cell")


if __name__ == "__main__":
    main()
