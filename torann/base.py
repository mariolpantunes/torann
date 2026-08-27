"""The index contract, as code.

Every implementation — the exact brute force (`brute.py`), the pure-Python
LSH reference (`lsh.py`), the Rust acceleration (`rust.py`) — answers the
same calls, so the public wrapper (`wrapper.py`) can select one at fit time
and the conformance suite can run unchanged against each.

The *normative* behaviour of the L1 LSH implementations (hash formula, tie
rules, byte-identical tables) is defined by ``lsh.py``, its executable
reference. The Rust class is a native ``pyclass`` and is
registered as a virtual subclass rather than inheriting.

Array conventions (validated by the wrapper before any call): points are
C-contiguous float64 ``(n, d)`` in ``[0, 1)``; ids are int64;
``query_radius`` returns a CSR triple ``(indptr, ids, dists)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["BaseIndex", "csr_to_padded"]


def csr_to_padded(
    indptr: np.ndarray, ids: np.ndarray, dists: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """CSR range results → the dense padded form ``query_knn`` returns.

    Rows are padded with ``-1`` / ``inf``, so a radius result has exactly the
    shape and the missing-neighbour convention a k-NN result has and feeds
    the same downstream kernel unchanged.

    Vectorised on purpose. The obvious version slices the CSR into a list of
    ``m`` tuples and then loops again to pad them, and *that* was the whole
    Python cost of a range query: two passes over ``m`` per call, every
    epoch. Building the scatter indices instead is 10-16x faster and
    bit-identical (nothing is arithmetic here — the values are copied). It
    is worth the trade only because the compiled kernel made the loop
    visible: at 0.4 ms of a 21 ms NumPy call it was 2%, at 0.4 ms of a
    0.5 ms compiled call it is a third of the query.

    Args:
        indptr: ``(m + 1,)`` int64 row offsets.
        ids: ``(nnz,)`` int64 neighbour ids, grouped by row.
        dists: ``(nnz,)`` float64 distances, grouped by row.
        m: Number of queries.

    Returns:
        ``(ids, dists)`` of shape ``(m, width)``, where ``width`` is the
        largest neighbourhood found — at least 1, so downstream shapes stay
        valid when every query came back empty.
    """
    counts = np.diff(indptr)
    width = max(1, int(counts.max()) if m else 1)
    # Position within its own row, for every entry: a global ramp minus each
    # entry's row start.
    col = np.arange(ids.size, dtype=np.int64) - np.repeat(indptr[:-1], counts)
    row = np.repeat(np.arange(m, dtype=np.int64), counts)
    out_ids = np.full((m, width), -1, dtype=np.int64)
    out_dst = np.full((m, width), np.inf)
    out_ids[row, col] = ids
    out_dst[row, col] = dists
    return out_ids, out_dst


class BaseIndex(ABC):
    """Two-tier toroidal-L1 index: static anchors + a moving candidate tier.

    Implementations expose ``n_points`` and ``n_static`` (attribute or
    property); ``n_candidates`` derives from them.
    """

    n_points: int
    n_static: int

    @abstractmethod
    def build(self, points: np.ndarray, n_static: int) -> None:
        """Build both tiers from zero: rows ``[0, n_static)`` are the static
        tier, the rest the candidate tier."""

    @abstractmethod
    def update(self, coords: np.ndarray) -> None:
        """Replace the candidate coordinates (same row order). LSH
        implementations re-place only entries whose key changed."""

    @abstractmethod
    def promote(self, new_candidates: np.ndarray) -> None:
        """Freeze the candidate tier into the static tier, then install
        ``new_candidates`` (possibly ``(0, d)``) as the new candidate tier."""

    @abstractmethod
    def query_knn(self, queries: np.ndarray, k: int,
                  exclude_ids: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Batch k-NN → ``(idx, dist)`` of shape ``(m, k)``, rows sorted by
        distance, padded with ``-1`` / ``inf`` when fewer than k reachable
        points exist. ``exclude_ids`` is one point id per query or None."""

    @abstractmethod
    def query_radius(self, queries: np.ndarray, radius: float,
                     exclude_ids: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch range query → CSR ``(indptr, ids, dists)``; row ``i`` is
        ``ids[indptr[i]:indptr[i+1]]``, sorted by distance."""

    @property
    def n_candidates(self) -> int:
        """Points in the candidate tier."""
        return self.n_points - self.n_static
