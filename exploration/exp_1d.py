"""1D experiments: collision law, winding defect, seam invariance, stability.

Candidates:
  H1  rotated integer grid   h(x) = floor(B ((x+u) mod 1))        [toroidal]
  H2  integer projection     h(x) = floor(B ((z x + u) mod 1))    [toroidal, suspect]
  H3  naive shifted grid     h(x) = floor((x+u) / (1/B))          [not toroidal]

Every P[collision] is a Monte-Carlo estimate over fresh (x, hash-draw) pairs.
Run from the repository root:  python exploration/exp_1d.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import vizstyle as vs

RNG = np.random.default_rng(0)
TRIALS = 200_000
OUT = os.path.join(os.path.dirname(__file__), "out")


def circ_dist(a):
    """Toroidal distance of a (mod-1) displacement."""
    a = np.mod(a, 1.0)
    return np.minimum(a, 1.0 - a)


def h1(x, u, B):
    return np.floor(B * np.mod(x + u, 1.0)).astype(np.int64)


def h2(x, u, z, B):
    return np.floor(B * np.mod(z * x + u, 1.0)).astype(np.int64)


def h3(x, u, B):
    return np.floor(B * (x + u)).astype(np.int64)  # grid on the line, no wrap


def collision_curve(hash_pair, deltas):
    """Measured P[h(x) = h(y)] for y = (x + delta) mod 1, x ~ U[0,1)."""
    out = np.empty(len(deltas))
    for i, d in enumerate(deltas):
        x = RNG.random(TRIALS)
        y = np.mod(x + d, 1.0)
        out[i] = np.mean(hash_pair(x, y))
    return out


def main():
    vs.apply()
    os.makedirs(OUT, exist_ok=True)
    deltas = np.linspace(0.0, 0.5, 51)

    # ---------------- A + B: collision law and the winding defect ----------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    def pair_h1(B):
        def f(x, y):
            u = RNG.random(x.shape)
            return h1(x, u, B) == h1(y, u, B)
        return f

    def pair_h2(B):
        def f(x, y):
            u = RNG.random(x.shape)
            z = np.round(RNG.normal(0, 2.0, x.shape))
            z[z == 0] = 1.0
            return h2(x, u, z, B) == h2(y, u, z, B)
        return f

    def pair_h3(B):
        def f(x, y):
            u = RNG.random(x.shape)
            return h3(x, u, B) == h3(y, u, B)
        return f

    curves = [
        ("H1 grid, B=2", pair_h1(2), vs.BLUE),
        ("H1 grid, B=4", pair_h1(4), vs.AQUA),
        ("H2 int. projection, B=4", pair_h2(4), vs.RED),
        ("H3 naive grid, B=4", pair_h3(4), vs.YELLOW),
    ]
    for name, f, color in curves:
        axA.plot(deltas, collision_curve(f, deltas), color=color, label=name)
    for B, color in ((2, vs.BLUE), (4, vs.AQUA)):
        axA.plot(deltas, np.maximum(0, 1 - B * deltas), ls="--", lw=1.1,
                 color=color, alpha=0.85)
    axA.set(title="A — collision probability vs toroidal distance",
            xlabel="toroidal distance δ", ylabel="P[same bucket]")
    axA.legend()
    axA.text(0.31, 0.62, "dashed = max(0, 1−Bδ)", fontsize=8.5, color=vs.INK2)

    B = 4
    for z, color in ((1, vs.BLUE), (2, vs.VIOLET), (3, vs.RED)):
        def fz(x, y, z=z):
            u = RNG.random(x.shape)
            return h2(x, u, float(z), B) == h2(y, u, float(z), B)
        axB.plot(deltas, collision_curve(fz, deltas), color=color, label=f"z = {z}")
        axB.plot(deltas, np.maximum(0, 1 - B * circ_dist(z * deltas)),
                 ls="--", lw=1.1, color=color, alpha=0.85)
    axB.set(title="B — H2 winding defect (B=4, fixed weight z)",
            xlabel="toroidal distance δ", ylabel="P[same bucket]")
    axB.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "1d_collision.png"), bbox_inches="tight")

    # ---------------- C + D: seam invariance and update stability ----------
    fig, (axC, axD) = plt.subplots(1, 2, figsize=(11, 4.2))

    delta = 0.05
    mids = np.linspace(0.0, 1.0, 101)
    for name, f, color in curves:
        pc = np.empty(len(mids))
        for i, m in enumerate(mids):
            x = np.full(TRIALS // 4, np.mod(m - delta / 2, 1.0))
            y = np.full(TRIALS // 4, np.mod(m + delta / 2, 1.0))
            pc[i] = np.mean(f(x, y))
        axC.plot(mids, pc, color=color, label=name)
    axC.axvspan(0.0, delta / 2, color=vs.GRID, alpha=0.6)
    axC.axvspan(1.0 - delta / 2, 1.0, color=vs.GRID, alpha=0.6)
    axC.text(0.03, 0.08, "pair straddles seam", fontsize=8.5, color=vs.INK2)
    axC.set(title=f"C — seam invariance (pair distance fixed at δ={delta})",
            xlabel="pair midpoint position on the circle",
            ylabel="P[same bucket]", ylim=(-0.02, 1.02))
    axC.legend(loc="center")

    steps = np.linspace(0.0, 0.6, 61)
    for B, color in ((2, vs.BLUE), (3, vs.VIOLET), (4, vs.AQUA)):
        pc = np.empty(len(steps))
        for i, s in enumerate(steps):
            x = RNG.random(TRIALS // 4)
            u = RNG.random(TRIALS // 4)
            pc[i] = np.mean(h1(x, u, B) != h1(np.mod(x + s, 1.0), u, B))
        axD.plot(steps, pc, color=color, label=f"B = {B}")
        axD.plot(steps, np.minimum(1, B * circ_dist(steps)), ls="--", lw=1.1,
                 color=color, alpha=0.85)
    axD.set(title="D — H1 index churn: P[bucket changes] per step",
            xlabel="movement step s (toroidal)", ylabel="P[rehash]")
    axD.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "1d_invariance.png"), bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
