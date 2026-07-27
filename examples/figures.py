"""Generate the README method figures (2D, transparent background).

Everything is derived from the library itself — the hash formula, the
collision law and the toroidal metric are computed, not drawn by hand:

  assets/method_hash.png       the offset integer grid in 2D: points
                               coloured by their cell key; the same colour
                               touching opposite edges *is* the wrap.
  assets/method_collision.png  measured per-dimension collision frequency
                               vs the exact law max(0, 1 - B*delta).
  assets/method_region.png     the toroidal L1 ball: what "closest region"
                               means when the query sits near a corner.
  assets/region.gif            the ball following a query from the centre
                               of the square into a corner — the region
                               wraps around all four edges.

Run from the repository root:  python examples/figures.py
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib import animation                    # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from torann.brute import exact_knn                  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")

# House palette (matches the logo).
SLATE, LIGHT = "#64748b", "#cbd5e1"
BLUE, TEAL, PURPLE = "#3b82f6", "#14b8a6", "#8b5cf6"

RC = {
    "figure.dpi": 200, "savefig.transparent": True,
    "font.size": 11, "text.color": SLATE,
    "axes.edgecolor": SLATE, "axes.labelcolor": SLATE,
    "xtick.color": SLATE, "ytick.color": SLATE,
    "axes.titlecolor": SLATE, "legend.frameon": False,
}


def cell_keys(X, B, u):
    """The 2D L1 hash: one table, K = 2, S = (0, 1)."""
    codes = np.minimum((np.mod(X + u, 1.0) * B).astype(np.int64), B - 1)
    return codes[:, 0] + B * codes[:, 1]


def fig_hash(B=3, seed=4):
    rng = np.random.default_rng(seed)
    u = rng.random(2)
    X = rng.random((260, 2))

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(X[:, 0], X[:, 1], c=cell_keys(X, B, u), cmap="Set2",
               s=42, edgecolors="white", linewidths=0.8, zorder=3)
    for m in range(B):  # cell boundaries: (m/B - u) mod 1 per axis
        ax.axvline(np.mod(m / B - u[0], 1.0), color=SLATE, ls="--", lw=1.1)
        ax.axhline(np.mod(m / B - u[1], 1.0), color=SLATE, ls="--", lw=1.1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0, 1), yticks=(0, 1))
    ax.set_title(f"c(x) = ⌊B·((x+u) mod 1)⌋   (B = {B}, one table)\n"
                 "same colour on opposite edges: cells wrap — no seam")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "method_hash.png"))
    plt.close(fig)


def fig_collision(n=200_000, seed=1):
    rng = np.random.default_rng(seed)
    # per-dimension toroidal distance lives in [0, 1/2]
    deltas = np.linspace(0.0, 0.5, 51)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for B, color in ((2, BLUE), (4, TEAL), (8, PURPLE)):
        x = rng.random(n)
        u = rng.random(n)  # a fresh offset per pair = expectation over u
        measured = [
            (np.floor(np.mod(x + u, 1.0) * B) ==
             np.floor(np.mod(x + delta + u, 1.0) * B)).mean()
            for delta in deltas
        ]
        ax.plot(deltas, np.maximum(0.0, 1.0 - B * deltas), color=color,
                lw=1.4, label=f"exact  max(0, 1 − {B}δ)")
        ax.plot(deltas, measured, "o", color=color, ms=3.5, alpha=0.75)
    ax.set(xlabel="per-dimension toroidal distance δ",
           ylabel="P[same cell]",
           title="the collision law is exact (dots: measured)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "method_collision.png"))
    plt.close(fig)


def _ball_axes(ax, pts, q, r, k=12):
    """Distance field + L1 ball + true k-NN for query q, on ax."""
    g = np.linspace(0.0, 1.0, 320)
    GX, GY = np.meshgrid(g, g)
    dx = np.abs(GX - q[0])
    dy = np.abs(GY - q[1])
    D = np.minimum(dx, 1.0 - dx) + np.minimum(dy, 1.0 - dy)
    ax.imshow(D, origin="lower", extent=(0, 1, 0, 1), cmap="Purples_r",
              vmin=0.0, vmax=1.0, alpha=0.9)
    ax.contour(GX, GY, D, levels=[r], colors=[TEAL], linewidths=2.0)
    idx, _ = exact_knn(pts, q[None, :], k)
    nn = pts[idx[0]]
    others = np.ones(len(pts), dtype=bool)
    others[idx[0]] = False
    ax.scatter(pts[others, 0], pts[others, 1], s=14, c=LIGHT, zorder=3)
    ax.scatter(nn[:, 0], nn[:, 1], s=42, c=TEAL, edgecolors="white",
               linewidths=0.8, zorder=4)
    ax.scatter([q[0]], [q[1]], s=130, c=BLUE, marker="*",
               edgecolors="white", linewidths=0.8, zorder=5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0, 1), yticks=(0, 1))
    ax.set_aspect("equal")


def fig_region(seed=2, n=400, r=0.22):
    pts = np.random.default_rng(seed).random((n, 2))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    _ball_axes(axes[0], pts, np.array([0.5, 0.5]), r)
    axes[0].set_title("query at the centre: an L1 ball (diamond)")
    _ball_axes(axes[1], pts, np.array([0.96, 0.94]), r)
    axes[1].set_title("query at the corner: the same ball, wrapped\n"
                      "(teal = its true 12-NN — across all four edges)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "method_region.png"))
    plt.close(fig)


def gif_region(seed=2, n=400, r=0.22, frames=36):
    """The query slides from the centre into a corner; its nearest
    region follows it around the torus. (GIF: solid background.)"""
    pts = np.random.default_rng(seed).random((n, 2))
    path = np.linspace([0.5, 0.5], [0.99, 0.97], frames)
    path = np.vstack([path, path[::-1]])  # ping-pong loop

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=110)
    fig.patch.set_facecolor("white")

    def draw(i):
        ax.clear()
        ax.set_facecolor("white")
        _ball_axes(ax, pts, path[i], r)
        ax.set_title("toroidal L1: the nearest region wraps")

    anim = animation.FuncAnimation(fig, draw, frames=len(path))
    anim.save(os.path.join(OUT, "region.gif"),
              writer=animation.PillowWriter(fps=14))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(RC)
    fig_hash()
    print("method_hash.png")
    fig_collision()
    print("method_collision.png")
    fig_region()
    print("method_region.png")
    gif_region()
    print("region.gif")
