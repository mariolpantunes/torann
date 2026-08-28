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

Implementations return bare tuples, because two of the three are compiled
and a native ``pyclass`` cannot cheaply build a Python type per call. The
public wrapper names them (`Neighbours`) once, at the boundary a user
sees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["BaseIndex", "DistArray", "IdArray", "Neighbours", "csr_to_padded"]

type IdArray = NDArray[np.int64]
"""Neighbour ids. ``-1`` marks padding, never a point."""

type DistArray = NDArray[np.float64]
"""Toroidal-L1 distances. ``inf`` pairs with an id of ``-1``."""


class Neighbours(NamedTuple):
    """The answer to a query: who, and how far.

    A named tuple rather than a bare pair because the two arrays are always
    read together and nothing but position told them apart -- ``res[0]``
    says nothing, ``res.ids`` does. It *is* a tuple, so unpacking, indexing
    and equality are unchanged and no caller has to be updated.

    Both modes return this shape. ``query`` fills ``(m, k)``;
    ``query_radius(pad=True)`` fills ``(m, width)`` with the same ``-1`` /
    ``inf`` convention, which is what lets a caller switch between them
    without reshaping. ``query_radius(pad=False)`` returns one of these per
    query, ragged and unpadded.
    """

    ids: IdArray
    distances: DistArray


def csr_to_padded(
    indptr: IdArray, ids: IdArray, dists: DistArray, m: int
) -> Neighbours:
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
        `Neighbours` of shape ``(m, width)``, where ``width`` is the
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
    return Neighbours(out_ids, out_dst)


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
