"""The uniform-grid fallback for `L^p`, kept as the control arm.

`exp_lp_family.py` gets `exp(-c*delta^p)` *exactly* by drawing the cell rate
from a `p`-stable law and scattering Poisson breakpoints on the circle. This
script answers the weaker question `NEXT.md` §7.4 originally posed: can the
same kernel be reached while keeping **uniform** cells, i.e. mixing only over
the torus-legal widths `w = 1/B` with integer `B`, plus an atom for "drop
this dimension" (`f == 1`)?

It matters for two reasons. The uniform-grid hash is one `floor()` per
dimension where the breakpoint hash is a `searchsorted`, so if the mixture
is close enough the cheaper kernel wins on speed; and it is the honest
control for any claim that the Poisson construction was *necessary* rather
than merely elegant.

The representation theory is the same in both cases — `E_w[max(0, 1-delta/w)]`
spans exactly the convex decreasing `f` with `f(0)=1`, and `exp(-c*delta^p)`
is convex for `p <= 1` — so the only question is what the integer-`B`
quantisation and the `w <= 1` ceiling cost. Fitted by non-negative least
squares over `delta in [0, 0.5]`, the toroidal range.

Read `P(drop)` as the price of `p`: the fraction of sampled dimensions that
contribute nothing, so a table needs `1/(1 - P(drop))` times as many for
equal selectivity.
"""

import numpy as np

# Per-coordinate 5-NN gap and target per-dimension collision (NEXT.md §3).
SCALES = ((32, 0.154, 0.69), (8, 0.090, 0.80))
PS = (1.0, 0.5, 0.25)
BMAX = 256


def nnls(a: np.ndarray, b: np.ndarray, iters: int = 20000) -> np.ndarray:
    """Non-negative least squares by projected gradient.

    Small and dependency-free on purpose: scipy is not a dependency of this
    repo, and the design matrix here is a few hundred columns.
    """
    x = np.full(a.shape[1], 1.0 / a.shape[1])
    step = 1.0 / np.linalg.norm(a, 2) ** 2
    for _ in range(iters):
        x = np.maximum(0.0, x - step * (a.T @ (a @ x - b)))
    return x


def main() -> None:
    delta = np.linspace(0.0, 0.5, 2001)
    widths = np.arange(2, BMAX + 1)
    # Columns: the collision law of a uniform B-cell grid with random offset,
    # plus a constant column for the dropped dimension.
    design = np.stack(
        [np.maximum(0.0, 1.0 - b * delta) for b in widths] + [np.ones_like(delta)],
        axis=1,
    )

    print(f"{'d':>4} {'p':>5} {'c':>7} {'max err':>9} {'rms':>8} "
          f"{'P(drop)':>8} {'P(B=2)':>7} {'eff dims':>9}")
    for d, dstar, q0 in SCALES:
        for p in PS:
            c = -np.log(q0) / dstar**p
            target = np.exp(-c * delta**p)
            w = nnls(design, target)
            w /= w.sum()
            err = design @ w - target
            print(f"{d:>4} {p:>5} {c:>7.3f} {np.abs(err).max():>9.4f} "
                  f"{np.sqrt((err**2).mean()):>8.4f} {w[-1]:>8.3f} "
                  f"{w[0]:>7.3f} {1.0 - w[-1]:>9.3f}")


if __name__ == "__main__":
    main()
