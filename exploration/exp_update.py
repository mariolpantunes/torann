"""Experiment H: how stale can the index get before recall suffers?

Points drift a little every epoch (sigma = 0.01 per dimension). We never
rehash — buckets keep the keys from epoch 0 — and measure candidate-set
recall of the true 8-NN (computed from *current* positions) as staleness
accumulates. Compared against a fresh index rebuilt every epoch, plus the
fraction of keys that actually changed (the work an exact selective update
would do).

Run from the repository root:  python exploration/exp_update.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import vizstyle as vs

RNG = np.random.default_rng(4)
OUT = os.path.join(os.path.dirname(__file__), "out")

N, K, B, L, KNN, EPOCHS, SIGMA = 4000, 5, 3, 3, 8, 12, 0.02


def codes_of(pts, S, U):
    frac = B * np.mod(pts[:, S] + U, 1.0)
    c = np.minimum(frac.astype(np.int64), B - 1)
    return c, frac - c


def keys_of(pts, S, U):
    return codes_of(pts, S, U)[0] @ (B ** np.arange(K))


def candidate_recall(pts, table_keys, S_all, U_all, q_ids, exact, probes=True):
    """Recall of true KNN inside the probed candidate set (query uses
    *current* positions; buckets hold `table_keys`, possibly stale)."""
    found = np.zeros((len(q_ids), N), dtype=bool)
    for t in range(L):
        codes, f = codes_of(pts[q_ids], S_all[t], U_all[t])
        qkeys = [codes @ (B ** np.arange(K))]
        if probes:
            dirs = np.where(f >= 0.5, 1, -1)
            for j in range(K):  # neighbour-cell probes
                alt = codes.copy()
                alt[:, j] = (alt[:, j] + dirs[:, j]) % B
                qkeys.append(alt @ (B ** np.arange(K)))
        for qk in qkeys:
            found |= table_keys[t][None, :] == qk[:, None]
    return found[np.arange(len(q_ids))[:, None], exact].mean()


def torus_knn(pts, q_ids):
    d = np.abs(pts[q_ids][:, None, :] - pts[None, :, :])
    D = np.minimum(d, 1.0 - d).sum(-1)
    D[np.arange(len(q_ids)), q_ids] = np.inf
    return np.argsort(D, axis=1)[:, :KNN]


def main():
    vs.apply()
    os.makedirs(OUT, exist_ok=True)
    pts = RNG.random((N, 2))
    S_all = RNG.integers(0, 2, size=(L, K))
    U_all = RNG.random((L, K))

    stale_keys = [keys_of(pts, S_all[t], U_all[t]) for t in range(L)]
    epochs = np.arange(EPOCHS + 1)
    rec_stale, rec_stale_np, rec_fresh, churn = [], [], [], []
    for _ in epochs:
        q_ids = RNG.choice(N, 300, replace=False)
        exact = torus_knn(pts, q_ids)
        fresh_keys = [keys_of(pts, S_all[t], U_all[t]) for t in range(L)]
        rec_stale.append(candidate_recall(pts, stale_keys, S_all, U_all, q_ids, exact))
        rec_stale_np.append(candidate_recall(pts, stale_keys, S_all, U_all,
                                             q_ids, exact, probes=False))
        rec_fresh.append(candidate_recall(pts, fresh_keys, S_all, U_all, q_ids, exact))
        churn.append(np.mean([np.mean(f != s) for f, s in zip(fresh_keys, stale_keys)]))
        pts = np.mod(pts + RNG.normal(0, SIGMA, (N, 2)), 1.0)

    fig, (axR, axC) = plt.subplots(1, 2, figsize=(11, 4.2))
    axR.plot(epochs, rec_fresh, "o-", ms=5, color=vs.BLUE, mec=vs.SURFACE,
             mew=0.8, label="rebuilt every epoch")
    axR.plot(epochs, rec_stale, "o-", ms=5, color=vs.AQUA, mec=vs.SURFACE,
             mew=0.8, label="stale, with probes")
    axR.plot(epochs, rec_stale_np, "o-", ms=5, color=vs.RED, mec=vs.SURFACE,
             mew=0.8, label="stale, no probes")
    axR.set(title=f"H — recall under staleness (σ={SIGMA}/dim/epoch)",
            xlabel="epochs since last rehash", ylabel="candidate-set recall",
            ylim=(0, 1.03))
    axR.legend(loc="lower left")

    axC.plot(epochs, churn, "o-", ms=5, color=vs.VIOLET, mec=vs.SURFACE, mew=0.8,
             label="measured")
    drift = SIGMA * np.sqrt(2 * epochs / np.pi)  # E|N(0, sigma*sqrt(e))|
    axC.plot(epochs, 1 - (1 - np.minimum(1, B * drift)) ** K, ls="--", lw=1.1,
             color=vs.VIOLET, alpha=0.85, label="theory 1−(1−B·E|drift|)^K")
    axC.set(title="H — stale keys: work an exact update would do",
            xlabel="epochs since last rehash", ylabel="fraction of keys changed",
            ylim=(0, 1.03))
    axC.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "2d_staleness.png"), bbox_inches="tight")

    for e in (0, 1, 2, 4, 8, 12):
        print(f"epoch {e:>2}: stale recall {rec_stale[e]:.3f} "
              f"(fresh {rec_fresh[e]:.3f}), churn {churn[e]:.3f}")


if __name__ == "__main__":
    main()
