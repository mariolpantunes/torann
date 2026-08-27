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

__all__ = ["BruteIndex", "exact_knn", "exact_radius", "pairwise_l1",
           "radius_scan"]

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
    acc = (((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
           + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7])))
    while j < d:
        acc += fold(j, tmp)
        j += 1
    return acc


def _blocks(m: int, n: int, d: int):
    step = max(1, _BUDGET // max(1, n * d))
    for s in range(0, m, step):
        yield s, min(m, s + step)


def exact_knn(pts, Q, k, exclude_ids=None):
    """Exact toroidal-L1 k-NN of ``Q`` against ``pts``.

    Args:
        pts: ``(n, d)`` float64 points in ``[0, 1)``.
        Q: ``(m, d)`` float64 queries in ``[0, 1)``.
        k: Neighbours per query.
        exclude_ids: Optional int64 ``(m,)`` — one point id excluded per
            query (the self-join).

    Returns:
        ``(idx, dist)`` of shape ``(m, k)``: int64 ids and float64
        distances, rows sorted ascending, padded with ``-1`` / ``inf``
        where fewer than ``k`` points exist.
    """
    m, (n, d) = Q.shape[0], pts.shape
    idx = np.full((m, k), -1, dtype=np.int64)
    dst = np.full((m, k), np.inf)
    kk = min(k, n)
    for s, e in _blocks(m, n, d):
        D = pairwise_l1(Q[s:e], pts)
        if exclude_ids is not None:
            D[np.arange(e - s), exclude_ids[s:e]] = np.inf
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
            order = np.argsort(pd, axis=1, kind="stable")
            part = np.take_along_axis(part, order, axis=1)
            pd = np.take_along_axis(pd, order, axis=1)
        finite = np.isfinite(pd)
        idx[s:e, :kk] = np.where(finite, part, -1)
        dst[s:e, :kk] = np.where(finite, pd, np.inf)
    return idx, dst


def exact_radius(pts, Q, radius, exclude_ids=None):
    """Exact toroidal-L1 range query.

    Args:
        pts: ``(n, d)`` float64 points in ``[0, 1)``.
        Q: ``(m, d)`` float64 queries in ``[0, 1)``.
        radius: Inclusive distance threshold.
        exclude_ids: Optional int64 ``(m,)`` — one point id excluded per
            query.

    Returns:
        CSR triple ``(indptr, ids, dists)``: row ``i`` is
        ``ids[indptr[i]:indptr[i+1]]``, sorted by distance.
    """
    m, (n, d) = Q.shape[0], pts.shape
    counts, all_ids, all_dst = [], [], []
    for s, e in _blocks(m, n, d):
        D = pairwise_l1(Q[s:e], pts)
        if exclude_ids is not None:
            D[np.arange(e - s), exclude_ids[s:e]] = np.inf
        for row in D:
            ids = np.flatnonzero(row <= radius)
            order = np.argsort(row[ids], kind="stable")
            counts.append(ids.size)
            all_ids.append(ids[order])
            all_dst.append(row[ids[order]])
    indptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(np.asarray(counts, dtype=np.int64), out=indptr[1:])
    return (indptr,
            np.concatenate(all_ids) if all_ids else np.empty(0, np.int64),
            np.concatenate(all_dst) if all_dst else np.empty(0))


def radius_scan(pts, Q, radius, exclude_ids=None):
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
    if rust.brute_radius is None:
        return exact_radius(pts, Q, radius, exclude_ids)
    q = np.ascontiguousarray(Q, dtype=np.float64)
    ex = None if exclude_ids is None else np.ascontiguousarray(
        exclude_ids, dtype=np.int64)
    return rust.brute_radius(pts, q, float(radius), ex)


class BruteIndex(BaseIndex):
    """The exact implementation of the index contract."""

    def __init__(self):
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
        self._pts[self.n_static:] = coords

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
            return exact_knn(self._pts, queries, k, exclude_ids)
        q = np.ascontiguousarray(queries, dtype=np.float64)
        ex = None if exclude_ids is None else np.ascontiguousarray(
            exclude_ids, dtype=np.int64)
        return rust.brute_knn(self._pts, q, int(k), ex)

    def query_radius(self, queries, radius, exclude_ids=None):
        """Exact range query — see :func:`radius_scan`."""
        return radius_scan(self._pts, queries, radius, exclude_ids)
