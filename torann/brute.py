"""Exact toroidal-``L^p`` search — blocked NumPy, no approximation.

Below the wrapper's ``brute_threshold`` this *is* the index (crossover
measurements in ANALYSIS.md: exact search wins outright at small n). It is
also the ground truth the LSH implementations are validated against, and
serves ``query_radius(exact=True)`` at any size via the module functions.

``p`` defaults to 1 everywhere, which is the metric the rest of the library
is built on, and that path is bit-identical to the L1-only code it replaced.
Other ``p`` are served here and *only* here: for ``p < 1`` the quasi-norm
has no triangle inequality, so every bound the LSH path prunes with —
prefix relaxation, bucket bounds, the collision guarantee itself — is
invalid. The facade does not expose ``p``; use these functions or
``BruteIndex`` directly.
"""

from __future__ import annotations

import numpy as np

from .base import BaseIndex

__all__ = ["BruteIndex", "pairwise_l1", "pairwise_lp", "exact_knn",
           "exact_radius"]

# Element budget for the temporary (queries x points) distance blocks.
_BUDGET = 1 << 24

# NumPy reduces a contiguous axis with *pairwise* summation: eight lane
# accumulators, a fixed tree over them, then the tail — and past a 128-element
# block it recurses, which this reconstruction does not follow. Below that
# limit the order can be reproduced exactly with (m, n) blocks, which is what
# lets the (m, n, d) temporary disappear without moving a single bit.
_PAIRWISE_MAX = 128


def pairwise_lp(Q: np.ndarray, pts: np.ndarray, p: float = 1.0) -> np.ndarray:
    r"""Dense toroidal ``L^p`` matrix, **un-rooted**.

    The metric is

    $$d_p(a, b) = \Big(\sum_{i=1}^{d} \delta_i^{\,p}\Big)^{1/p},
      \qquad \delta_i = \min(|a_i - b_i|,\; 1 - |a_i - b_i|)$$

    and what this returns is the inner sum, $\sum_i \delta_i^{\,p}$. The two
    are monotonically related, so k-NN *selection* and radius comparison can
    both run on the sum and pay one ``pow`` per selected row instead of one
    per pair; ``exact_knn`` and ``exact_radius`` take the root themselves
    where a calibrated distance is reported. At ``p = 1`` the sum *is* the
    distance and no root is taken anywhere.

    The wrap is per axis and metric-independent — only the aggregation
    changes with ``p``. For ``p < 1`` this is a quasi-norm: it satisfies the
    triangle inequality only up to a factor, which is why nothing that
    prunes may consume it (see the module docstring).

    Accumulates over the ``d`` dimensions into ``(m, n)`` buffers instead of
    materialising the ``(m, n, d)`` difference block and reducing it. The
    block was the whole cost: profiled on the shapes ESS gives this path
    (256x256 at d=2, 7680x256 at d=2), the reduction over an axis of length
    two alone was ~45% of the call, because a two-element reduction is all
    per-output overhead. Results are unchanged — the accumulation follows
    NumPy's own pairwise order for ``d <= 128`` (verified bit-for-bit), and
    above that the original block form still serves. ``p = 1`` skips the
    ``pow`` rather than raising to the first power, so that path is
    bit-identical to the L1-only version of this function and no measurement
    on record shifts.

    Still call it through a block loop (``exact_knn`` / ``exact_radius`` do):
    the working set is now ~10 ``(m, n)`` buffers rather than ``d`` of them.

    Args:
        Q: ``(m, d)`` float64 queries in ``[0, 1)``.
        pts: ``(n, d)`` float64 points in ``[0, 1)``.
        p: Metric exponent, ``> 0``. ``p = 1`` is toroidal L1.

    Returns:
        ``(m, n)`` float64 matrix of ``sum_i delta_i**p``.
    """
    if not p > 0.0:
        raise ValueError(f"p must be positive, got {p}")
    m, d = Q.shape
    n = pts.shape[0]
    unit = p == 1.0
    # `sqrt` is exactly 2x `power(x, 0.5)` at every shape measured, from
    # (60,120) to (256,4000). The obvious generalisation does *not* follow:
    # x**0.75 as sqrt(x)*sqrt(sqrt(x)) is 1.4-2.4x **slower** than one
    # `power` everywhere, because three ufunc calls and their temporaries
    # cost more than the pow they replace, in cache or out of it. Measured
    # and rejected — do not re-try the dyadic chain.
    root = p == 0.5
    if d > _PAIRWISE_MAX:
        diff = np.abs(Q[:, None, :] - pts[None, :, :])
        np.minimum(diff, 1.0 - diff, out=diff)
        if root:
            np.sqrt(diff, out=diff)
        elif not unit:
            np.power(diff, p, out=diff)
        return diff.sum(-1)

    wall = np.empty((m, n))

    def fold(j, out):
        """``min(|q_j - p_j|, 1 - |q_j - p_j|)**p`` into ``out``, no alloc."""
        np.subtract(Q[:, j, None], pts[None, :, j], out=out)
        np.abs(out, out=out)
        np.subtract(1.0, out, out=wall)
        np.minimum(out, wall, out=out)
        if root:
            np.sqrt(out, out=out)
        elif not unit:
            np.power(out, p, out=out)
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


def pairwise_l1(Q: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Dense toroidal-L1 distance matrix — :func:`pairwise_lp` at ``p = 1``,
    where the un-rooted sum is the distance itself."""
    return pairwise_lp(Q, pts, 1.0)


def _root(sums: np.ndarray, p: float) -> np.ndarray:
    """Un-rooted sums to ``L^p`` distances, in place, ``inf`` preserved."""
    if p == 1.0:
        return sums
    return np.power(sums, 1.0 / p, out=sums)


def _blocks(m: int, n: int, d: int):
    step = max(1, _BUDGET // max(1, n * d))
    for s in range(0, m, step):
        yield s, min(m, s + step)


def exact_knn(pts, Q, k, exclude_ids=None, p=1.0):
    """Exact toroidal ``L^p`` k-NN of ``Q`` against ``pts``.

    Selection runs on :func:`pairwise_lp`'s un-rooted sums, which are
    rank-equivalent to the distance, so the root costs one ``pow`` per
    selected row rather than one per pair.

    Args:
        pts: ``(n, d)`` float64 points in ``[0, 1)``.
        Q: ``(m, d)`` float64 queries in ``[0, 1)``.
        k: Neighbours per query.
        exclude_ids: Optional int64 ``(m,)`` — one point id excluded per
            query (the self-join).
        p: Metric exponent, ``> 0``. Defaults to toroidal L1.

    Returns:
        ``(idx, dist)`` of shape ``(m, k)``: int64 ids and float64
        ``L^p`` distances, rows sorted ascending, padded with ``-1`` /
        ``inf`` where fewer than ``k`` points exist.
    """
    m, (n, d) = Q.shape[0], pts.shape
    idx = np.full((m, k), -1, dtype=np.int64)
    dst = np.full((m, k), np.inf)
    kk = min(k, n)
    for s, e in _blocks(m, n, d):
        D = pairwise_lp(Q[s:e], pts, p)
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
        dst[s:e, :kk] = np.where(finite, _root(pd, p), np.inf)
    return idx, dst


def exact_radius(pts, Q, radius, exclude_ids=None, p=1.0):
    """Exact toroidal ``L^p`` range query.

    The threshold is raised to ``p`` once and the comparison runs on
    :func:`pairwise_lp`'s un-rooted sums; only the kept distances are
    rooted.

    Args:
        pts: ``(n, d)`` float64 points in ``[0, 1)``.
        Q: ``(m, d)`` float64 queries in ``[0, 1)``.
        radius: Inclusive distance threshold, in ``L^p``.
        exclude_ids: Optional int64 ``(m,)`` — one point id excluded per
            query.
        p: Metric exponent, ``> 0``. Defaults to toroidal L1.

    Returns:
        CSR triple ``(indptr, ids, dists)``: row ``i`` is
        ``ids[indptr[i]:indptr[i+1]]``, sorted by distance.
    """
    m, (n, d) = Q.shape[0], pts.shape
    # Negative radii admit nothing and would go complex under a fractional
    # power, so they pass through as the sentinel they already are.
    thr = radius if (p == 1.0 or radius < 0.0) else radius ** p
    counts, all_ids, all_dst = [], [], []
    for s, e in _blocks(m, n, d):
        D = pairwise_lp(Q[s:e], pts, p)
        if exclude_ids is not None:
            D[np.arange(e - s), exclude_ids[s:e]] = np.inf
        for row in D:
            ids = np.flatnonzero(row <= thr)
            order = np.argsort(row[ids], kind="stable")
            counts.append(ids.size)
            all_ids.append(ids[order])
            all_dst.append(_root(row[ids[order]], p))
    indptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(np.asarray(counts, dtype=np.int64), out=indptr[1:])
    return (indptr,
            np.concatenate(all_ids) if all_ids else np.empty(0, np.int64),
            np.concatenate(all_dst) if all_dst else np.empty(0))


class BruteIndex(BaseIndex):
    """The exact implementation of the index contract.

    Args:
        p: Metric exponent, ``> 0``. The default is toroidal L1, the metric
            the contract and every LSH implementation are written against;
            other ``p`` are exact here and unavailable anywhere else.
    """

    def __init__(self, p: float = 1.0):
        if not p > 0.0:
            raise ValueError(f"p must be positive, got {p}")
        self.p = float(p)
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
        """Exact k-NN — see :func:`exact_knn`."""
        return exact_knn(self._pts, queries, k, exclude_ids, self.p)

    def query_radius(self, queries, radius, exclude_ids=None):
        """Exact range query — see :func:`exact_radius`."""
        return exact_radius(self._pts, queries, radius, exclude_ids, self.p)
