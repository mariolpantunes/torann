"""A toroidal LSH family for `L^p`, `0 < p <= 1`, with the stable law in the
cell rate rather than in a projection vector.

`NEXT.md` §7.2 measures that lower `p` restores the contrast the ESS force
law loses above `d ~ 8`, and §7.3 measures that the L1 hash cannot retrieve
`L^p` neighbours — its collision probability is a function of L1 and nothing
else, so reranking by `L^p` is capped at 93.2% recall, structurally. §7.4
therefore asks for a hash family designed natively for `p < 1`.

The textbook answer (Datar, Immorlica, Indyk, Mirrokni, SoCG'04 — the first
provably efficient ANN scheme for `p < 1`) is to draw the projection vector
from a `p`-stable law and grid the projection:
`h(x) = floor((a.x + b) / w)`. **That construction cannot be used here.**
The continuous homomorphisms `T^d -> T^1` are exactly the characters
`x -> sum_j n_j x_j mod 1` with *integer* `n_j` (Pontryagin duality), so a
real-valued `a` has no toroidal meaning at all, and an integer-valued one
wraps many times over and destroys locality. torann's collision law is
exactly `max(0, 1 - B*delta)` with no boundary defect *because* it never
projects — it grids each coordinate independently. Seamlessness and
projection are mutually exclusive on the torus.

The same stable law re-enters one level down. For `p in (0, 1]`,
`exp(-c*delta^p)` is completely monotone, hence a Laplace mixture:

    E_s[ exp(-s*delta) ] = exp(-c*delta^p)   for  s = c^(1/p) * S_p

with `S_p` the one-sided `p`-stable subordinator — sampled by the same
Chambers-Mallows-Stuck method the projection route would have used. And
`exp(-s*delta)` is exactly the void probability of a rate-`s` Poisson
process on a circle. So the construction is:

    per (table, dimension): draw s ~ c^(1/p) * S_p, scatter Poisson(s)
    breakpoints on [0, 1), hash x_j to the index of the arc containing it.

Per-coordinate collision is then `exp(-c*delta_j^p)` *exactly*, seamlessly
toroidal, with no integer-B constraint anywhere. Concatenating `K`
dimensions gives `exp(-c * sum_j delta_j^p) = exp(-c * ||delta||_p^p)`: an
exact `L^p` LSH family on the torus, where the only approximation is the
rate cap below.

Two things this script quantifies, because both decide the design:

1. **The cap.** `E[S_p] = inf` for `p < 1`, so the expected cell count per
   dimension is infinite and capping the rate is mandatory, not a tuning
   choice. All of the family's error lives here.
2. **The drop atom.** `P(no breakpoint) = exp(-c)` in closed form: a
   dimension that does not discriminate at all. Dimension subsampling stops
   being a separate mechanism — it falls out of the family, and it prices
   `p` directly, since a table needs `1/(1 - e^-c)` times the sampled
   dimensions for equal selectivity.

Calibration follows §3: `c` is set so that a true 5-NN, which differs from
the query by `delta*` in a typical coordinate, collides with probability
`q0` in that coordinate.
"""

import numpy as np

# Per-coordinate 5-NN gap and the per-dimension collision it should earn,
# from the measured geometry table in NEXT.md §3.
SCALES = ((32, 0.154, 0.69), (8, 0.090, 0.80), (4, 0.041, 0.90))
PS = (1.0, 0.5, 0.25)
CAPS = (np.inf, 200.0, 50.0)
NSAMP = 2_000_000


def positive_stable(p: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """One-sided `p`-stable subordinator, `p in (0, 1)`, by Chambers-Mallows-Stuck.

    Normalised so that `E[exp(-lam * S)] = exp(-lam**p)`, which is the
    identity the whole construction rests on; `check_sampler` verifies it.
    """
    theta = rng.uniform(0.0, np.pi, size)
    w = rng.exponential(1.0, size)
    return (np.sin(p * theta) / np.sin(theta) ** (1.0 / p)) * (
        np.sin((1.0 - p) * theta) / w
    ) ** ((1.0 - p) / p)


def calibrate(p: float, dstar: float, q0: float) -> float:
    """`c` such that `exp(-c * dstar**p) == q0`."""
    return -np.log(q0) / dstar**p


def check_sampler(rng: np.random.Generator) -> None:
    """The Laplace transform identity, without which nothing below holds."""
    print("CMS sampler: E[exp(-lam*S)] against exp(-lam^p)")
    for p in (0.5, 0.25):
        s = positive_stable(p, NSAMP, rng)
        for lam in (0.25, 1.0, 2.0):
            mc = float(np.mean(np.exp(-lam * s)))
            exact = float(np.exp(-(lam**p)))
            print(f"  p={p:<5} lam={lam:<5} MC {mc:.5f}   exact {exact:.5f}"
                  f"   err {abs(mc - exact):.2e}")


def main() -> None:
    rng = np.random.default_rng(0)
    check_sampler(rng)

    delta = np.linspace(0.0, 0.5, 201)  # toroidal per-coordinate gap
    print(f"\n{'d':>4} {'p':>5} {'c':>7} {'rate cap':>9} {'max err':>9} "
          f"{'E[cells]':>13} {'P(drop)':>8} {'eff dims':>9}")
    for d, dstar, q0 in SCALES:
        for p in PS:
            c = calibrate(p, dstar, q0)
            target = np.exp(-c * delta**p)
            raw = c ** (1.0 / p) * positive_stable(p, NSAMP, rng)
            for cap in CAPS:
                s = np.minimum(raw, cap)
                f = np.array([np.mean(np.exp(-s * dd)) for dd in delta])
                drop = float(np.mean(np.exp(-s)))
                print(f"{d:>4} {p:>5} {c:>7.3f} {cap:>9} "
                      f"{np.abs(f - target).max():>9.5f} {np.mean(s):>13.2f} "
                      f"{drop:>8.3f} {1.0 - drop:>9.3f}")


if __name__ == "__main__":
    main()
