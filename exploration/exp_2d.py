"""2D experiments: concatenation law, bucket geometry, mini-ANN recall.

Run from the repository root:  python exploration/exp_2d.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import vizstyle as vs

RNG = np.random.default_rng(1)
OUT = os.path.join(os.path.dirname(__file__), "out")


def h1_codes(X, U, B):
    """H1 per-dimension cells for points X (n,2) with offsets U (2,)."""
    return np.floor(B * np.mod(X + U, 1.0)).astype(np.int64)


def torus_l1(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d).sum(-1)


# ---------------------------------------------------------------- E: law --
def concat_law(ax):
    B, trials = 3, 400_000
    d1, d2 = RNG.random(trials) * 0.5, RNG.random(trials) * 0.5
    x = RNG.random((trials, 2))
    y = np.mod(x + np.column_stack([d1, d2]), 1.0)
    U = RNG.random((trials, 2))
    hit = (h1_codes(x, U, B) == h1_codes(y, U, B)).all(axis=1)
    theory = np.maximum(0, 1 - B * d1) * np.maximum(0, 1 - B * d2)

    bins = np.linspace(0, 1, 21)
    which = np.digitize(theory, bins) - 1
    meas = np.array([hit[which == i].mean() if (which == i).any() else np.nan
                     for i in range(len(bins) - 1)])
    centers = (bins[:-1] + bins[1:]) / 2
    ax.plot([0, 1], [0, 1], ls="--", lw=1.1, color=vs.MUTED)
    ax.plot(centers, meas, "o", ms=5, color=vs.BLUE, mec=vs.SURFACE, mew=0.8)
    ax.set(title="E — concatenation law (K=2, B=3)",
           xlabel="predicted  ∏ max(0, 1−Bδᵢ)", ylabel="measured P[same bucket]")
    ax.text(0.05, 0.9, "dashed = y = x", fontsize=8.5, color=vs.INK2)


# ----------------------------------------------------------- F: geometry --
def bucket_maps(fig_path):
    B, res = 3, 420
    g = (np.arange(res) + 0.5) / res
    GX, GY = np.meshgrid(g, g)
    P = np.column_stack([GX.ravel(), GY.ravel()])
    cmap = ListedColormap([c + "55" for c in vs.SERIES] + [c + "AA" for c in vs.SERIES])

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.2))
    (axH1, axH2), (axNet, axNaive) = axes

    u = np.array([0.31, 0.77])
    cells = h1_codes(P, u, B)
    axH1.imshow((cells[:, 0] * B + cells[:, 1]).reshape(res, res), cmap=cmap,
                origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")
    axH1.set_title("H1 — one table's cells (B=3, K=2)")

    z1, z2 = np.array([2.0, -1.0]), np.array([1.0, 2.0])
    c1 = np.floor(B * np.mod(P @ z1 + 0.31, 1.0))
    c2 = np.floor(B * np.mod(P @ z2 + 0.77, 1.0))
    axH2.imshow((c1 * B + c2).reshape(res, res), cmap=cmap,
                origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")
    axH2.set_title("H2 — winding stripes (z=(2,−1),(1,2))")

    # Candidate net: cells sharing a bucket with a corner query, L=6 tables.
    q = np.array([0.04, 0.06])
    L = 6
    net_h1 = np.zeros(res * res, dtype=bool)
    net_naive = np.zeros(res * res, dtype=bool)
    for _ in range(L):
        u = RNG.random(2)
        net_h1 |= (h1_codes(P, u, B) == h1_codes(q[None], u, B)).all(axis=1)
        naive = np.floor(B * (P + u)).astype(np.int64)
        naive_q = np.floor(B * (q[None] + u)).astype(np.int64)
        net_naive |= (naive == naive_q).all(axis=1)
    for ax, net, title in ((axNet, net_h1, "H1 candidate net — wraps the corner"),
                           (axNaive, net_naive, "H3 naive net — clipped at borders")):
        ax.imshow(net.reshape(res, res), cmap=ListedColormap([vs.SURFACE, vs.BLUE + "66"]),
                  origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")
        ax.plot(*q, "o", ms=9, color=vs.RED, mec=vs.SURFACE, mew=1.2)
        ax.set_title(f"{title} (L={L})")

    for ax in axes.ravel():
        ax.grid(False)
        ax.set(xticks=(0, 0.5, 1), yticks=(0, 0.5, 1))
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")


# ----------------------------------------------------------- G: mini-ANN --
def mini_ann(ax_recall, ax_cands):
    n, k, B, K = 4000, 8, 3, 4
    pts = RNG.random((n, 2))
    q_ids = RNG.choice(n, 300, replace=False)
    exact = np.argsort(
        np.array([torus_l1(pts[q][None], pts) for q in q_ids]), axis=1)[:, 1:k + 1]

    Ls = [1, 2, 4, 8, 12]
    for probe, color, label in ((False, vs.BLUE, "buckets only"),
                                (True, vs.AQUA, "+ neighbour-cell probe")):
        recalls, cand_counts = [], []
        for L in Ls:
            S = RNG.integers(0, 2, size=(L, K))
            U = RNG.random((L, K))
            found = np.zeros((len(q_ids), n), dtype=bool)
            for t in range(L):
                frac = B * np.mod(pts[:, S[t]] + U[t], 1.0)
                codes = np.minimum(frac.astype(np.int64), B - 1)
                keysets = [codes]
                if probe:
                    f = frac - codes
                    dirs = np.where(f >= 0.5, 1, -1)
                    for j in range(K):
                        alt = codes.copy()
                        alt[:, j] = (alt[:, j] + dirs[:, j]) % B
                        keysets.append(alt)
                base = codes @ (B ** np.arange(K))
                qsets = [ks[q_ids] @ (B ** np.arange(K)) for ks in keysets]
                for qk in qsets:
                    found |= base[None, :] == qk[:, None]
            hits = found[np.arange(len(q_ids))[:, None], exact].mean()
            recalls.append(hits)
            cand_counts.append(found.sum(1).mean())
        ax_recall.plot(Ls, recalls, "o-", ms=5, color=color, label=label,
                       mec=vs.SURFACE, mew=0.8)
        ax_cands.plot(Ls, cand_counts, "o-", ms=5, color=color, label=label,
                      mec=vs.SURFACE, mew=0.8)
    ax_recall.axhline(0.95, ls="--", lw=1.1, color=vs.MUTED)
    ax_recall.text(6.5, 0.905, "0.95 target", fontsize=8.5, color=vs.INK2)
    ax_recall.set(title=f"G — candidate-set recall of true {k}-NN (n={n})",
                  xlabel="tables L", ylabel="recall", ylim=(0, 1.02))
    ax_recall.legend(loc="lower right")
    ax_cands.axhline(n, ls="--", lw=1.1, color=vs.MUTED)
    ax_cands.text(1.2, n * 0.88, "n (brute force)", fontsize=8.5, color=vs.INK2)
    ax_cands.set(title="G — cost: mean candidates per query",
                 xlabel="tables L", ylabel="candidates", yscale="log")
    ax_cands.legend(loc="lower right")


def main():
    vs.apply()
    os.makedirs(OUT, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    concat_law(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "2d_concat.png"), bbox_inches="tight")

    bucket_maps(os.path.join(OUT, "2d_buckets.png"))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    mini_ann(a, b)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "2d_miniann.png"), bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
