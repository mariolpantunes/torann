"""Exact toroidal-L1 search — blocked NumPy, no approximation.

Below the wrapper's ``brute_threshold`` this *is* the index: exact search
wins outright at small n, measured per backend (see
``ToroidalNN._BRUTE_DEFAULTS``). It is also the ground truth the LSH
implementations are validated against, and serves
``query_radius(exact=True)`` at any size via the module functions.
"""

from __future__ import annotations

import numpy as np

from . import rust
from .base import BaseIndex

__all__ = ["BruteIndex", "exact_knn", "exact_radius", "pairwise_l1", "pairwise_lp", "radius_scan"]

# Element budget for the temporary (queries x points) distance blocks.
#
# `pairwise_l1` keeps about ten (m, n) buffers live at once -- eight lane
# accumulators, a temp and the wall -- so the working set is roughly 80 bytes
# per element of the block, not 8. At 1 << 24 that is well past any L3 and the
# loop runs at main-memory speed. Measured over the shapes ESS and OBLESA give
# this path, 1 << 22 is faster at every one of them and by 6-11% at most:
#
#   M/Q/d          1<<20    1<<22    1<<24
#   400/200/100    22.27    20.56    22.12
#   400/2048/32    83.52    91.93   102.15
#   960/512/100   155.66   142.37   154.67
#   200/4096/100  216.15   189.74   214.30
#   3840/512/100  567.06   483.76   515.68
#
# Blocking cannot change the result -- each row's distances are accumulated
# the same way whatever the block -- and that was verified rather than assumed
# at every budget in the table.
_BUDGET = 1 << 22

# NumPy reduces a contiguous axis with *pairwise* summation: eight lane
# accumulators, a fixed tree over them, then the tail — and past a 128-element
# block it recurses, which this reconstruction does not follow. Below that
# limit the order can be reproduced exactly with (m, n) blocks, which is what
# lets the (m, n, d) temporary disappear without moving a single bit.
_PAIRWISE_MAX = 128


def pairwise_l1(Q: np.ndarray, pts: np.ndarray) -> np.ndarray:
    r"""Dense toroidal-L1 distance matrix.

    The metric is

    $$d(a, b) = \sum_{i=1}^{d} \min(|a_i - b_i|,\; 1 - |a_i - b_i|)$$

    Accumulates over the ``d`` dimensions into ``(m, n)`` buffers instead of
    materialising the ``(m, n, d)`` difference block and reducing it. The
    block was the whole cost: profiled on the shapes ESS gives this path
    (256x256 at d=2, 7680x256 at d=2), the reduction over an axis of length
    two alone was ~45% of the call, because a two-element reduction is all
    per-output overhead. Results are unchanged — the accumulation follows
    NumPy's own pairwise order for ``d <= 128`` (verified bit-for-bit), and
    above that the original block form still serves.

    Still call it through a block loop (``exact_knn`` / ``exact_radius`` do):
    the working set is now ~10 ``(m, n)`` buffers rather than ``d`` of them.

    Args:
        Q: ``(m, d)`` float64 queries in ``[0, 1)``.
        pts: ``(n, d)`` float64 points in ``[0, 1)``.

    Returns:
        ``(m, n)`` float64 distance matrix.
    """
    m, d = Q.shape
    n = pts.shape[0]
    if d > _PAIRWISE_MAX:
        diff = np.abs(Q[:, None, :] - pts[None, :, :])
        np.minimum(diff, 1.0 - diff, out=diff)
        return diff.sum(-1)

    wall = np.empty((m, n))

    def fold(j, out):
        """``min(|q_j - p_j|, 1 - |q_j - p_j|)`` into ``out``, no allocation."""
        np.subtract(Q[:, j, None], pts[None, :, j], out=out)
        np.abs(out, out=out)
        np.subtract(1.0, out, out=wall)
        np.minimum(out, wall, out=out)
        return out

    if d < 8:  # NumPy sums fewer than eight elements straight through
        acc = fold(0, np.empty((m, n)))
        tmp = np.empty((m, n))
        for j in range(1, d):
            acc += fold(j, tmp)
        return acc

    lanes = [fold(j, np.empty((m, n))) for j in range(8)]
    tmp = np.empty((m, n))
    j = 8
    while j + 8 <= d:
        for lane in range(8):
            lanes[lane] += fold(j + lane, tmp)
        j += 8
    acc = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
    while j < d:
        acc += fold(j, tmp)
        j += 1
    return acc


def _validate_p(p: float) -> float:
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p) or p <= 0.0:
        raise ValueError(f"p must be a finite positive float, got {p!r}")
    return float(p)


def _root(x: np.ndarray, p: float) -> np.ndarray:
    if p == 1.0:
        return x
    if p == 0.5:
        return np.square(x)
    return np.power(x, 1.0 / p)


def pairwise_lp(Q: np.ndarray, pts: np.ndarray, p: float = 1.0) -> np.ndarray:
    r"""Un-rooted toroidal-``L^p`` sum matrix $\sum_{i=1}^d \delta_i^p$.

    Rank-equivalent to the ``L^p`` distance (for all $p > 0$), so callers
    can select top-$k$ or threshold against $r^p$ before taking the $1/p$
    root only on the surviving entries. Delegates directly to
    :func:`pairwise_l1` when ``p == 1.0`` so the default path is bit-identical.
    """
    p = _validate_p(p)
    if p == 1.0:
        return pairwise_l1(Q, pts)
    m, d = Q.shape
    n = pts.shape[0]
    if d > _PAIRWISE_MAX:
        diff = np.abs(Q[:, None, :] - pts[None, :, :])
        np.minimum(diff, 1.0 - diff, out=diff)
        np.power(diff, p, out=diff)
        return diff.sum(-1)

    wall = np.empty((m, n))

    def fold(j, out):
        np.subtract(Q[:, j, None], pts[None, :, j], out=out)
        np.abs(out, out=out)
        np.subtract(1.0, out, out=wall)
        np.minimum(out, wall, out=out)
        if p == 0.5:
            np.sqrt(out, out=out)
        elif p == 0.25:
            np.sqrt(out, out=out)
            np.sqrt(out, out=out)
        else:
            np.power(out, p, out=out)
        return out

    if d < 8:
        acc = fold(0, np.empty((m, n)))
        tmp = np.empty((m, n))
        for j in range(1, d):
            acc += fold(j, tmp)
        return acc

    lanes = [fold(j, np.empty((m, n))) for j in range(8)]
    tmp = np.empty((m, n))
    j = 8
    while j + 8 <= d:
        for lane in range(8):
            lanes[lane] += fold(j + lane, tmp)
        j += 8
    acc = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
    while j < d:
        acc += fold(j, tmp)
        j += 1
    return acc


def _blocks(m: int, n: int, d: int):
    step = max(1, _BUDGET // max(1, n * d))
    for s in range(0, m, step):
        yield s, min(m, s + step)


def _exclude(D, exclude_ids, s, e):
    """Mask each query row's excluded id in the block ``[s, e)``; ``-1``
    excludes nothing (it must not wrap to the last point)."""
    if exclude_ids is None:
        return
    rows = np.flatnonzero(exclude_ids[s:e] >= 0)
    D[rows, exclude_ids[s:e][rows]] = np.inf


def exact_knn(pts, Q, k, exclude_ids=None, p: float = 1.0):
    """Exact toroidal-``L^p`` k-NN of ``Q`` against ``pts``.

    Args:
        pts: ``(n, d)`` float64 points in ``[0, 1)``.
        Q: ``(m, d)`` float64 queries in ``[0, 1)``.
        k: Neighbours per query.
        exclude_ids: Optional int64 ``(m,)`` — one point id excluded per
            query (the self-join).
        p: Metric exponent ``> 0`` (default ``1.0``, toroidal L1).

    Returns:
        ``(idx, dist)`` of shape ``(m, k)``: int64 ids and float64
        distances, rows sorted ascending, padded with ``-1`` / ``inf``
        where fewer than ``k`` points exist.
    """
    p = _validate_p(p)
    m, (n, d) = Q.shape[0], pts.shape
    idx = np.full((m, k), -1, dtype=np.int64)
    dst = np.full((m, k), np.inf)
    kk = min(k, n)
    for s, e in _blocks(m, n, d):
        D = pairwise_lp(Q[s:e], pts, p)
        _exclude(D, exclude_ids, s, e)
        if kk == 1:
            # ESS asks for exactly this in `_smart_init` (the farthest of a
            # candidate pool), and it is worth splitting out: argpartition
            # builds a full (m, n) index matrix to select one column, which
            # profiled at ~45% of the call. Ties go to the lowest id here
            # rather than to whatever introselect left in place — that is a
            # tighter guarantee, not a looser one.
            part = D.argmin(axis=1)[:, None]
            pd = np.take_along_axis(D, part, axis=1)
        else:
            part = np.argpartition(D, kk - 1, axis=1)[:, :kk]
            pd = np.take_along_axis(D, part, axis=1)
            # argpartition keeps an arbitrary subset of the distance tied at
            # the k-th slot; the contract keeps the lowest ids, so rows with
            # such a tie are re-selected from every point at or below it.
            kth = pd.max(axis=1)
            tied = np.flatnonzero((D == kth[:, None]).sum(axis=1) > (pd == kth[:, None]).sum(axis=1))
            for row in tied:
                pool = np.flatnonzero(D[row] <= kth[row])
                part[row] = pool[np.lexsort((pool, D[row, pool]))][:kk]
                pd[row] = D[row, part[row]]
            order = np.lexsort((part, pd), axis=1)  # (distance, id)
            part = np.take_along_axis(part, order, axis=1)
            pd = np.take_along_axis(pd, order, axis=1)
        finite = np.isfinite(pd)
        idx[s:e, :kk] = np.where(finite, part, -1)
        dst[s:e, :kk] = np.where(finite, _root(pd, p), np.inf)
    return idx, dst


def exact_radius(pts, Q, radius, exclude_ids=None, p: float = 1.0):
    """Exact toroidal-``L^p`` range query.

    Args:
        pts: ``(n, d)`` float64 points in ``[0, 1)``.
        Q: ``(m, d)`` float64 queries in ``[0, 1)``.
        radius: Inclusive distance threshold.
        exclude_ids: Optional int64 ``(m,)`` — one point id excluded per
            query.
        p: Metric exponent ``> 0`` (default ``1.0``, toroidal L1).

    Returns:
        CSR triple ``(indptr, ids, dists)``: row ``i`` is
        ``ids[indptr[i]:indptr[i+1]]``, sorted by distance.
    """
    p = _validate_p(p)
    m, (n, d) = Q.shape[0], pts.shape
    thr = radius if (p == 1.0 or radius <= 0.0) else radius**p
    counts, all_ids, all_dst = [], [], []
    for s, e in _blocks(m, n, d):
        D = pairwise_lp(Q[s:e], pts, p)
        _exclude(D, exclude_ids, s, e)
        for row in D:
            ids = np.flatnonzero(row <= thr)
            order = np.argsort(row[ids], kind="stable")
            counts.append(ids.size)
            all_ids.append(ids[order])
            all_dst.append(_root(row[ids[order]], p))
    indptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(np.asarray(counts, dtype=np.int64), out=indptr[1:])
    return (
        indptr,
        np.concatenate(all_ids) if all_ids else np.empty(0, np.int64),
        np.concatenate(all_dst) if all_dst else np.empty(0),
    )


def radius_scan(pts, Q, radius, exclude_ids=None, p: float = 1.0):
    """The *serving* exact range scan: compiled when the wheel provides it.

    The same split `BruteIndex.query_knn` makes — `exact_radius` above stays
    the reference and stays reachable — with the dispatch factored out here
    because two callers need it: `BruteIndex` below, and the wrapper's
    ``query_radius(exact=True)``, which scans the arena directly and would
    otherwise be the one path the kernel never reached.

    It carries the compiled contract: f64 with the algebraic float methods,
    so a few ulp on the distance. One consequence the k-NN path does not
    have — ``<=`` is a threshold, so a point within an ulp of `radius` can
    fall on either side of it, and the returned *set* can differ by that
    point rather than merely its order.

    50-70x on the shapes ESS and OBLESA give this path.

    Args, Returns:
        As :func:`exact_radius`.
    """
    p = _validate_p(p)
    if rust.brute_radius is None:
        return exact_radius(pts, Q, radius, exclude_ids, p=p)
    q = np.ascontiguousarray(Q, dtype=np.float64)
    ex = None if exclude_ids is None else np.ascontiguousarray(exclude_ids, dtype=np.int64)
    if p == 1.0:
        return rust.brute_radius(pts, q, float(radius), ex)
    return rust.brute_radius(pts, q, float(radius), ex, p)


class BruteIndex(BaseIndex):
    """The exact implementation of the index contract."""

    def __init__(self, p: float = 1.0):
        self.p = _validate_p(p)
        self._pts = np.empty((0, 0))
        self.n_points = 0
        self.n_static = 0

    def build(self, points, n_static):
        """Copy the points; both tiers share one array here."""
        self._pts = np.ascontiguousarray(points, dtype=np.float64).copy()
        self.n_points = self._pts.shape[0]
        self.n_static = int(n_static)

    def update(self, coords):
        """Overwrite the candidate rows (no structure to maintain)."""
        self._pts[self.n_static :] = coords

    def promote(self, new_candidates):
        """Freeze candidates into the static tier; append the new batch."""
        self.n_static = self.n_points
        if new_candidates.size:
            self._pts = np.vstack([self._pts, new_candidates])
            self.n_points = self._pts.shape[0]

    def query_knn(self, queries, k, exclude_ids=None):
        """k-NN over every point — compiled when the wheel provides it.

        `exact_knn` below is the reference and stays reachable: it is what
        `metrics.toroidal_separation` reports with and what the LSH
        implementations are validated against, so it must not move. This is
        the *serving* path, and it carries the same contract the compiled LSH
        backend has had since 1.98 — f32 with the algebraic float methods, a
        few ulp on the distance, ids that can reorder within an ulp.

        It is 11-125x faster than the scan it replaces, most of that from
        never materialising the `(m, n)` distance matrix `argpartition` would
        need: the distance is consumed by the heap as it is computed. Below
        `ToroidalNN`'s crossover this is the only path there is, so until now
        the compiled kernel served exclusively the regime a population-sized
        caller never enters.
        """
        if rust.brute_knn is None:
            return exact_knn(self._pts, queries, k, exclude_ids, p=self.p)
        q = np.ascontiguousarray(queries, dtype=np.float64)
        ex = None if exclude_ids is None else np.ascontiguousarray(exclude_ids, dtype=np.int64)
        if self.p == 1.0:
            return rust.brute_knn(self._pts, q, int(k), ex)
        return rust.brute_knn(self._pts, q, int(k), ex, self.p)

    def query_radius(self, queries, radius, exclude_ids=None):
        """Exact range query — see :func:`radius_scan`."""
        return radius_scan(self._pts, queries, radius, exclude_ids, p=self.p)
