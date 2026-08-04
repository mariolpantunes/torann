"""Render the README benchmark figures from recorded measurements.

The numbers below are the phase-8 measurements
(v3) — AMD Ryzen AI 7 PRO 350, 16 threads, d=16, k=32, best of 3. They can
be re-measured with `examples/bench_backends.py`, `compare_faiss_flat.py`
and `ess_sim.py`; this script only draws them.

  assets/bench_query.png   µs/query vs n: torus data (the real workload)
                           and wrap-free box data (FAISS's best case).
  assets/bench_ess.png     the simulated ESS main loop, ms per epoch,
                           with the recall each system actually delivers.

Run from the repository root:  python examples/plot_benchmarks.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")

SLATE = "#64748b"
BLUE, TEAL, PURPLE, AMBER, ROSE = (
    "#3b82f6", "#14b8a6", "#8b5cf6", "#f59e0b", "#f43f5e")

RC = {
    "figure.dpi": 200, "savefig.transparent": True,
    "font.size": 11, "text.color": SLATE,
    "axes.edgecolor": SLATE, "axes.labelcolor": SLATE,
    "xtick.color": SLATE, "ytick.color": SLATE,
    "axes.titlecolor": SLATE, "legend.frameon": False,
}

N = [20_000, 100_000, 500_000, 1_000_000]

# µs/query, torus data (d=16, k=32): the workload torann exists for.
TORUS = {
    "NumPy brute force (exact)": (AMBER, [3561, 17840, 91723, 185406]),
    "torann [python]":           (PURPLE, [1363, 1908, 4008, 5065]),
    "torann [rust]":             (BLUE, [11.1, 16.0, 93.3, 148.0]),
}

# µs/query, wrap-free box data: toroidal L1 == plain L1, FAISS is exact.
BOX = {
    "NumPy brute force (exact)": (AMBER, [3464, 17570, 104857, 193080]),
    "FAISS Flat L1 (exact)":     (SLATE, [25.9, 131.7, 647.1, 1473.6]),
    "FAISS HNSW (recall .74-.87)": (ROSE, [2.9, 7.3, 19.0, 15.2]),
    "torann [rust]":             (BLUE, [28.4, 139.4, 899.0, 2297.4]),
}

# The ESS main loop, simulated end to end (examples/ess_sim.py): 15k
# anchors + 10 batches x 3k candidates x 32 epochs -> 45k points, torus
# data, d=16, k=32. ms/epoch (query + update) and delivered recall.
ESS = [
    ("torann [rust]", BLUE, 61.5, "recall 0.983–0.995"),
    ("FAISS Flat, rebuilt/epoch", ROSE, 106.2, "recall 0.25–0.28 — seam-blind"),
    ("torann [python]", PURPLE, 3728.9, "recall 0.983"),
]


def fig_query():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, data, title in (
        (axes[0], TORUS, "torus data — the real workload\n(FAISS cannot answer the metric)"),
        (axes[1], BOX, "wrap-free box data — FAISS's best case\n(toroidal L1 == plain L1)"),
    ):
        for label, (color, us) in data.items():
            ax.plot(N, us, "o-", color=color, lw=1.8, ms=5, label=label)
        ax.set(xscale="log", yscale="log", title=title, xlabel="indexed points n")
        ax.grid(True, which="both", color=SLATE, alpha=0.15, lw=0.6)
        ax.legend(fontsize=9, loc="upper left", frameon=True,
                  facecolor="#f8fafc", edgecolor="none", framealpha=0.92)
    axes[0].set_ylabel("µs / query   (d=16, k=32, 16 threads)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bench_query.png"))
    plt.close(fig)


def fig_ess():
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    labels = [r[0] for r in ESS][::-1]
    for i, (_, color, ms, note) in enumerate(reversed(ESS)):
        ax.barh(i, ms, color=color, height=0.62)
        ax.text(ms * 1.12, i, f"{ms:,.1f} ms — {note}",
                va="center", fontsize=10, color=SLATE)
    ax.set(xscale="log", yticks=range(len(ESS)), yticklabels=labels,
           xlabel="ms per ESS epoch (query + update), lower is better",
           title="the simulated ESS main loop, end to end\n"
                 "(45 000 points, d=16, k=32, torus data)")
    ax.set_xlim(right=6e4)
    ax.grid(True, axis="x", color=SLATE, alpha=0.15, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bench_ess.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(RC)
    fig_query()
    print("bench_query.png")
    fig_ess()
    print("bench_ess.png")
