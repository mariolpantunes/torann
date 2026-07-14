"""Exact toroidal-L1 search — blocked NumPy, no approximation.

Below the wrapper's ``brute_threshold`` this *is* the index (crossover
measurements in ANALYSIS.md: exact search wins outright at small n). It is
also the ground truth the LSH implementations are validated against, and
serves ``query_radius(exact=True)`` at any size via the module functions.
"""

from __future__ import annotations

import numpy as np

from .base import BaseIndex

__all__ = ["BruteIndex", "pairwise_l1", "exact_knn", "exact_radius"]

# Element budget for the temporary (queries x points) distance blocks.
_BUDGET = 1 << 24


def pairwise_l1(Q: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Dense toroidal-L1 distances, (m, n) — call through a block loop."""
    diff = np.abs(Q[:, None, :] - pts[None, :, :])
    np.minimum(diff, 1.0 - diff, out=diff)
    return diff.sum(-1)


def _blocks(m: int, n: int, d: int):
    step = max(1, _BUDGET // max(1, n * d))
    for s in range(0, m, step):
        yield s, min(m, s + step)


def exact_knn(pts, Q, k, exclude_ids=None):
    """Exact k-NN of Q against pts → (idx, dist), padded past n."""
    m, (n, d) = Q.shape[0], pts.shape
    idx = np.full((m, k), -1, dtype=np.int64)
    dst = np.full((m, k), np.inf)
    kk = min(k, n)
    for s, e in _blocks(m, n, d):
        D = pairwise_l1(Q[s:e], pts)
        if exclude_ids is not None:
            D[np.arange(e - s), exclude_ids[s:e]] = np.inf
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
    """Exact range query → CSR (indptr, ids, dists), rows distance-sorted."""
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


class BruteIndex(BaseIndex):
    """The exact implementation of the index contract."""

    def __init__(self):
        self._pts = np.empty((0, 0))
        self.n_points = 0
        self.n_static = 0

    def build(self, points, n_static):
        self._pts = np.ascontiguousarray(points, dtype=np.float64).copy()
        self.n_points = self._pts.shape[0]
        self.n_static = int(n_static)

    def update(self, coords):
        self._pts[self.n_static:] = coords

    def promote(self, new_candidates):
        self.n_static = self.n_points
        if new_candidates.size:
            self._pts = np.vstack([self._pts, new_candidates])
            self.n_points = self._pts.shape[0]

    def query_knn(self, queries, k, exclude_ids=None):
        return exact_knn(self._pts, queries, k, exclude_ids)

    def query_radius(self, queries, radius, exclude_ids=None):
        return exact_radius(self._pts, queries, radius, exclude_ids)
