"""Does ESS need recall, or only plausible repellers?

`OPTIMIZE.md` §4b measured that recall 0.69 at d=32 costs 1.28% CE against
perfect recall, while dropping tables to *lower* recall costs 4.3%. Those
two look contradictory until you notice that fewer tables degrades recall
**and** locality together, so that experiment confounds them. This one
separates them.

The index is exact throughout — recall is not a property of the index here,
it is imposed. Between the query and the force kernel the neighbour list is
replaced by a controlled corruption of it, and the *true* toroidal-L1
distance of whatever was substituted is passed along, so only the
**selection** changes and never the force magnitudes.

The arms form a locality ladder, all with recall pinned to 0 except the
first two:

===============  ==========================================================
``exact``        true k-NN (control)
``top2k``        k sampled from the true 2k nearest — §6's arm 2
``rank1-4k``     k sampled from ranks [k, 4k) — never a true neighbour,
                 still the nearest thing that is not one
``rank8-16k``    k sampled from ranks [8k, 16k) — local only in the loosest
                 sense
``ratio2x``      k sampled from everything within 2x the k-th distance —
                 §6's arm 3
``uniform``      k uniformly random points — the null; CE should collapse
===============  ==========================================================

Read the result as a ladder, not as pass/fail. If CE holds along it, ESS
wants plausible repellers and the index should be redesigned around cost
per *local* neighbour; if it falls off as soon as the true neighbours go,
the distance ranking is the product and the current design is already the
right shape. `mean_ratio` (delivered distance / true k-th distance) is what
makes the rungs comparable across dimensions.

Run from the repository root (needs `ess` on the path)::

    python examples/bench_recall_ablation.py
    python examples/bench_recall_ablation.py --cases 1 --seeds 5
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torann import ToroidalNN  # noqa: E402
from torann.brute import exact_knn, pairwise_l1  # noqa: E402

try:
    import ess
    import ess.utils
except ImportError:  # pragma: no cover - benchmark-only dependency
    sys.exit("ess not importable: add ~/git/ess/src to PYTHONPATH")

OUT = os.path.join(os.path.dirname(__file__), "out")

# Smaller than the suite's shapes on purpose: every arm runs an exact
# (m, n) join every epoch, so this is sized to keep the whole ladder x
# seeds sweep inside a coffee break. d=32 is the regime the question is
# about; d=8 is there to show whether the answer is dimension-dependent.
CASES = (
    (32, 0, 4000),
    (8, 0, 2048),
)

ARMS = ("exact", "top2k", "rank1-4k", "rank8-16k", "ratio2x", "uniform")

# Rank windows, in multiples of k. `None` marks the arms whose pool is not
# a rank window (they are selected by distance ratio, or over everything).
WINDOWS = {
    "exact": (0, 1),
    "top2k": (0, 2),
    "rank1-4k": (1, 4),
    "rank8-16k": (8, 16),
}

_BUDGET = 1 << 23  # float64 cells per distance block, ~64 MiB


def _sample_masked(rng, mask, k):
    """`k` column ids drawn uniformly without replacement from each row's
    ``True`` entries.

    Uniform random keys plus a partial sort is the vectorised way to do a
    per-row sample from a ragged candidate set: masked-out columns get an
    infinite key, so the `k` smallest keys are exactly a uniform draw from
    what is left. Rows with fewer than `k` candidates keep what they have
    and are padded with ``-1``.

    Args:
        rng (np.random.Generator): Source of the keys.
        mask (np.ndarray): ``(m, n)`` bool, eligible columns.
        k (int): Draws per row.

    Returns:
        np.ndarray: ``(m, k)`` int64 column ids, ``-1`` where a row ran out.
    """
    keys = rng.random(mask.shape)
    keys[~mask] = np.inf
    part = np.argpartition(keys, k - 1, axis=1)[:, :k]
    ok = np.isfinite(np.take_along_axis(keys, part, axis=1))
    return np.where(ok, part, -1)


class CorruptIndex(ToroidalNN):
    """An exact index that degrades its own answers in a controlled way.

    Overrides `query` only: `fit`, `update` and `promote` stay the facade's,
    so ESS drives it exactly as it drives the real thing. The returned
    distances are always the true toroidal L1 distances to the returned
    ids, which is what keeps this an experiment about *selection*.
    """

    def __init__(self, arm, seed=0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.arm = arm
        self._rng = np.random.default_rng(seed + 977)
        self.diag = {"recall": [], "ratio": []}

    def query(self, k=None, queries=None, exclude_ids=None):
        """Exact k-NN, then the arm's corruption. Signature is the facade's.

        Only the default self-join is corrupted — that is the query whose
        result reaches the force kernel. `_smart_init` passes explicit
        `queries` and stays exact, so initialisation is identical across
        arms and cannot be mistaken for the effect being measured.
        """
        self._check_fitted()
        kq = int(k) if k is not None else (self._k_hint or 2 * self._d)
        Q, ex = self._resolve_queries(queries, exclude_ids)
        pts = self._arena
        if queries is not None:
            return exact_knn(pts, np.ascontiguousarray(Q), kq, ex)
        m, n = Q.shape[0], pts.shape[0]

        idx = np.full((m, kq), -1, dtype=np.int64)
        dst = np.full((m, kq), np.inf)
        step = max(1, _BUDGET // max(1, n))
        for s in range(0, m, step):
            e = min(m, s + step)
            D = pairwise_l1(Q[s:e], pts)
            if ex is not None:
                D[np.arange(e - s), ex[s:e]] = np.inf
            idx[s:e], dst[s:e] = self._corrupt(D, kq)
        return idx, dst

    def _corrupt(self, D, k):
        """One block: exact ranking, then the arm's substitution.

        Args:
            D (np.ndarray): ``(m, n)`` true distances, self already ``inf``.
            k (int): Neighbours to deliver.

        Returns:
            tuple: ``(ids, dists)``, each ``(m, k)``, rows sorted ascending.
        """
        m, n = D.shape
        window = WINDOWS.get(self.arm)
        pool = min(n, k * (window[1] if window else 1))

        # The exact answer is needed by every arm: as the control, as the
        # rank window's source, as the radius the ratio arm doubles, and as
        # the denominator of the reported diagnostics.
        part = np.argpartition(D, pool - 1, axis=1)[:, :pool]
        pd = np.take_along_axis(D, part, axis=1)
        order = np.argsort(pd, axis=1, kind="stable")
        ranked = np.take_along_axis(part, order, axis=1)
        ranked_d = np.take_along_axis(pd, order, axis=1)
        truth = ranked[:, :k]
        kth = ranked_d[:, min(k, pool) - 1]

        if self.arm == "exact":
            ids = truth
        elif window is not None:
            lo = window[0] * k
            cols = self._rng.random((m, ranked.shape[1] - lo)).argsort(axis=1)
            ids = np.take_along_axis(ranked[:, lo:], cols[:, :k], axis=1)
        elif self.arm == "ratio2x":
            mask = D <= 2.0 * kth[:, None]
            ids = _sample_masked(self._rng, mask, k)
        elif self.arm == "uniform":
            ids = _sample_masked(self._rng, np.isfinite(D), k)
        else:  # pragma: no cover - guarded by the CLI choices
            raise ValueError(f"unknown arm {self.arm!r}")

        # A row that ran short keeps its true neighbours there; that is the
        # conservative direction (it moves the arm towards the control).
        ids = np.where(ids < 0, truth, ids)
        dists = np.take_along_axis(D, ids, axis=1)
        order = np.argsort(dists, axis=1, kind="stable")
        ids = np.take_along_axis(ids, order, axis=1)
        dists = np.take_along_axis(dists, order, axis=1)

        hit = (ids[:, :, None] == truth[:, None, :]).any(axis=2)
        self.diag["recall"].append(float(hit.mean()))
        self.diag["ratio"].append(
            float(np.mean(dists / np.maximum(kth[:, None], 1e-12))))
        return ids, dists


def run(arm, dim, anchors, candidates, seed):
    """One ESS run under one arm; returns its points and diagnostics."""
    rng = np.random.default_rng(seed)
    A = rng.random((anchors, dim)) if anchors else np.empty((0, dim))
    bounds = np.array([[0.0, 1.0]] * dim)
    index = CorruptIndex(arm, seed=seed)
    stats: dict = {}
    pts = ess.esa(A, bounds, n=candidates, index=index, seed=seed, stats=stats)
    points = np.vstack([A, pts]) if anchors else pts
    return points, {
        "recall": float(np.mean(index.diag["recall"])),
        "mean_ratio": float(np.mean(index.diag["ratio"])),
        "epochs": stats.get("epochs_total"),
    }


def report(rows):
    """Per shape, the ladder with CE and separation relative to the control."""
    shapes = sorted({(r["dim"], r["anchors"], r["candidates"]) for r in rows})
    for dim, anchors, cands in shapes:
        sub = [r for r in rows if (r["dim"], r["anchors"], r["candidates"])
               == (dim, anchors, cands)]
        base = np.mean([r["clark_evans"] for r in sub if r["arm"] == "exact"])
        print(f"\nd={dim}, {anchors}+{cands}, "
              f"{len({r['seed'] for r in sub})} seeds "
              f"(paired: same seed across arms)")
        head = ["arm", "recall", "mean_ratio", "CE", "dCE vs exact",
                "separation", "epochs"]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for arm in ARMS:
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            ce = np.array([r["clark_evans"] for r in a])
            sep = np.mean([r["separation"] for r in a])
            print(f"| {arm} | {np.mean([r['recall'] for r in a]):.3f} "
                  f"| {np.mean([r['mean_ratio'] for r in a]):.2f} "
                  f"| {ce.mean():.4f} +/- {ce.std(ddof=1) if len(ce) > 1 else 0:.4f} "
                  f"| {100 * (ce.mean() - base) / base:+.2f}% "
                  f"| {sep:.4f} "
                  f"| {np.mean([r['epochs'] for r in a]):.0f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="+")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", default=os.path.join(OUT, "recall_ablation.json"))
    args = ap.parse_args()

    import torann
    print(f"[torann: {torann.__file__}]")
    picked = ([CASES[i - 1] for i in args.cases] if args.cases else list(CASES))

    rows = []
    for dim, anchors, cands in picked:
        for arm in args.arms:
            for seed in range(args.seeds):
                pts, info = run(arm, dim, anchors, cands, seed)
                rows.append({
                    "dim": dim, "anchors": anchors, "candidates": cands,
                    "arm": arm, "seed": seed, **info,
                    "clark_evans": float(ess.utils.toroidal_clark_evans(pts)),
                    "separation": float(ess.utils.toroidal_separation(pts)),
                })
                print(f"  [d={dim} {arm} seed={seed}: "
                      f"CE {rows[-1]['clark_evans']:.4f} "
                      f"recall {info['recall']:.3f} "
                      f"ratio {info['mean_ratio']:.2f}]", flush=True)

    report(rows)
    os.makedirs(OUT, exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n[saved {args.json}]")
