"""The index contract, as code.

Every implementation — the exact brute force (`brute.py`), the pure-Python
LSH reference (`lsh.py`), the Rust acceleration (`rust.py`) — answers the
same calls, so the public wrapper (`wrapper.py`) can select one at fit time
and the conformance suite can run unchanged against each.

The *normative* behaviour of the LSH implementations (hash formula, tie
rules, byte-identical tables) is specified in ``CONTRACT.md``; ``lsh.py``
is its executable reference. The Rust class is a native ``pyclass`` and is
registered as a virtual subclass rather than inheriting.

Array conventions (validated by the wrapper before any call): points are
C-contiguous float64 ``(n, d)`` in ``[0, 1)``; ids are int64;
``query_radius`` returns a CSR triple ``(indptr, ids, dists)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


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
        return self.n_points - self.n_static
