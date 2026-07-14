"""Toroidal L1 nearest-neighbour search for ESS-style epoch workloads.

Contract (PLAN.md, gate outcomes 2026-07-13):

* **Metric**: toroidal L1 on the unit torus [0,1)^d, and nothing else. The
  hash family's guarantee is an L1 guarantee, so L1 is the public contract.
* **Hash** (validated in exploration/): per sampled dimension, an integer
  resolution ``B`` and offset ``u ~ U[0,1)`` give the cell
  ``c(x) = floor(B((x+u) mod 1))`` with collision probability **exactly**
  ``max(0, 1 - B*delta)`` — seamless on the torus. A table concatenates K
  sampled dimensions: collisions decay as ``prod(1-B*delta_i) ~ exp(-B*L1)``,
  a uniformly random pair collides per dimension with probability exactly
  ``1/B`` (bucket load ``n/B**K``), and a point that moves by ``s`` changes a
  cell with probability ``B*s`` (churn is proportional to movement).
* **Two tiers**: a *static tier*, hashed and key-sorted once, that grows only
  by promotion (a linear merge, never a re-sort), and a small *candidate
  tier* refreshed per epoch. Queries default to "each candidate against
  everything" — the ESS inner loop.
* **Selective updates**: ``update`` re-places only the points that moved far
  enough to change a cell (P[change] = B*step per dimension, experiment D),
  by delete + merge — never a full re-sort. The index is always exact after
  an update: fast *and* accurate.
* **No brute-force fallback**: an under-filled query widens its buckets by
  *prefix relaxation* — keys are digit-concatenations, so dropping low-order
  digits turns a bucket into a contiguous range of the sorted key array
  (two searchsorted calls per table per level). Level K spans the whole
  table, so k results are structurally guaranteed without a distance scan.
* **Tuning**: ``fit(..., k=...)`` or ``radius=...`` feeds ESS's own query
  heuristic into the index — B from the neighbour scale, K from the
  closed-form bucket load, L from a recall target. Explicit constructor
  arguments always win over tuning.

This class is a *facade*: it owns validation, tuning, hash-parameter drawing
and the exact brute-force path (used below ``brute_threshold`` total
points). The LSH index itself lives behind the backend contract in
``src/backends/CONTRACT.md`` — pure NumPy (``python``, the reference), C
(``c``) or Rust (``rust``); ``backend="auto"`` picks the fastest installed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .backends import resolve_backend

logger = logging.getLogger(__name__)

__all__ = ["ToroidalNN"]

# Element budget for temporary matrices in blocked exact search.
_BRUTE_BUDGET = 1 << 24
# Sampling caps for the tune() estimate of the neighbour scale.
_TUNE_QUERIES = 256
_TUNE_REFERENCE = 8192


class ToroidalNN:
    """Exact + approximate toroidal L1 k-NN with an ESS-shaped lifecycle.

    Lifecycle::

        nn = ToroidalNN(seed=0)
        nn.fit(static_pts, candidate_pts, k=2*d)  # build + tune from zero
        for epoch in range(E):
            idx, dist = nn.query()                # candidates vs everything
            new = move(nn.candidates, idx, dist)  # ESS force step
            nn.update(new)                        # selective refresh
        nn.promote(next_batch)                    # candidates -> static

    Args:
        seed: Seed for hash functions and tuning samples.
        brute_threshold: Exact search while total points <= this. ``None``
            (default) picks the measured brute/LSH crossover of the backend:
            4096 for python, 512 for the native backends (see ANALYSIS.md,
            crossover section).
        num_tables: Tables L. ``None`` = tuned (default 16 without hints).
        resolution: Cells per dimension B >= 2. ``None`` = tuned (default 3).
        dims_per_table: Sampled dimensions K. ``None`` = closed-form bucket
            load ``K = round(log_B(n / target_bucket_size))``.
        target_bucket_size: Bucket-load target for the K rule. ``None`` =
            ``max(32, k_hint)``.
        probes: Neighbour-cell probes per table per query.
        query_block_size: Queries per vectorised block.
        backend: LSH backend — ``"auto"`` (first installed of c, rust,
            python), or an explicit name.
    """

    # Measured brute/LSH crossover per backend (ANALYSIS.md): the NumPy
    # brute path stays competitive to ~4-8k points against the python
    # backend, while the native backends win from the smallest sizes tested.
    _BRUTE_DEFAULTS = {"python": 4096, "rust": 512}

    def __init__(
        self,
        seed: int | None = None,
        brute_threshold: int | None = None,
        num_tables: int | None = None,
        resolution: int | None = None,
        dims_per_table: int | None = None,
        target_bucket_size: int | None = None,
        probes: int = 4,
        query_block_size: int = 1024,
        backend: str = "auto",
    ):
        if resolution is not None and resolution < 2:
            raise ValueError("resolution must be >= 2")
        if num_tables is not None and num_tables < 1:
            raise ValueError("num_tables must be >= 1")
        if dims_per_table is not None and dims_per_table < 1:
            raise ValueError("dims_per_table must be >= 1")
        if probes < 0:
            raise ValueError("probes must be >= 0")

        self.brute_threshold = None if brute_threshold is None \
            else int(brute_threshold)
        self.num_tables = num_tables
        self.resolution = resolution
        self.dims_per_table = dims_per_table
        self.target_bucket_size = target_bucket_size
        self.probes = int(probes)
        self.query_block_size = int(query_block_size)
        self.backend = backend

        self._rng = np.random.default_rng(seed)
        self._arena = np.empty((0, 0))
        self._n_static = 0
        self._d = 0
        self._use_lsh = False
        self._k_hint: int | None = None
        self._lsh: Any = None  # backend instance; set when LSH mode engages

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def fit(
        self,
        static_points: np.ndarray,
        candidate_points: np.ndarray | None = None,
        k: int | None = None,
        radius: float | None = None,
    ) -> "ToroidalNN":
        """Build the index from zero. Hash functions are drawn (and tuned)
        here and kept stable for the whole run.

        Args:
            static_points: (n, d) anchors; never move.
            candidate_points: (c, d) points that move each epoch. Optional.
            k: The k the workload will query (ESS: 2*d). Drives tuning.
            radius: Alternative tuning hint: the range-query radius.
        """
        S = np.atleast_2d(np.asarray(static_points, dtype=np.float64))
        if S.ndim != 2 or S.shape[0] == 0:
            raise ValueError("static_points must be a non-empty (n, d) array")
        self._d = S.shape[1]
        C = self._as_batch(candidate_points)
        self._arena = np.mod(np.vstack([S, C]), 1.0)
        self._n_static = S.shape[0]
        self._k_hint = int(k) if k is not None else 2 * self._d

        self._use_lsh = self.n_points > self._threshold()
        if self._use_lsh:
            self._tune(radius)
            self._make_lsh()
        return self

    def update(self, coordinates: np.ndarray) -> None:
        """Move the candidate tier (one epoch step).

        The refresh is *selective*: only points whose key actually changed —
        i.e. that moved far enough to leave a cell — are re-placed, by
        delete + merge. The index is exact after every update.

        Args:
            coordinates: (n_candidates, d) new positions, same order as the
                current candidate tier.
        """
        self._check_fitted()
        new = np.mod(np.atleast_2d(np.asarray(coordinates, dtype=np.float64)), 1.0)
        if new.shape != (self.n_candidates, self._d):
            raise ValueError(
                f"expected coordinates of shape {(self.n_candidates, self._d)}")
        self._arena[self._n_static:] = new
        if self._use_lsh:
            self._lsh.update(np.ascontiguousarray(new))

    def promote(self, new_candidates: np.ndarray | None = None) -> None:
        """Freeze the candidate tier into the static tier and install a new
        batch. The merge is linear (searchsorted + insert), never a re-sort.
        """
        self._check_fitted()
        C = self._as_batch(new_candidates)
        if self._use_lsh:
            self._lsh.promote(C)
        self._n_static = self.n_points  # old candidates are static now
        if C.size:
            self._arena = np.vstack([self._arena, C])
        if not self._use_lsh and self.n_points > self._threshold():
            # Crossed the threshold: first (and only) full LSH build.
            self._use_lsh = True
            self._tune(None)
            self._make_lsh()

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def query(
        self,
        k: int | None = None,
        queries: np.ndarray | None = None,
        exclude_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch toroidal-L1 k-NN.

        With no arguments this is the ESS inner loop: every candidate point
        queries the whole index (both tiers), excluding itself.

        Args:
            k: Neighbours per query; defaults to the fit-time hint (2*d).
            queries: Optional (m, d) explicit queries; default is the
                candidate tier.
            exclude_ids: Optional per-query point id to exclude (set
                automatically for the default self-join).

        Returns:
            (indices, distances) of shape (m, k), sorted by distance; rows
            are padded with -1 / inf only when fewer than k points exist.
            Under-filled queries are completed by prefix relaxation, so k
            results are guaranteed without any brute-force scan.
        """
        self._check_fitted()
        kq = int(k) if k is not None else (self._k_hint or 2 * self._d)
        if kq < 1:
            raise ValueError("k must be >= 1")
        Q, ex = self._resolve_queries(queries, exclude_ids)
        if not self._use_lsh:
            return self._brute_query(Q, kq, ex)
        return self._lsh.query_knn(np.ascontiguousarray(Q), kq, ex)

    def query_radius(
        self,
        radius: float,
        queries: np.ndarray | None = None,
        exact: bool = False,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batch range query: indexed points within toroidal L1 ``radius``.

        In LSH mode this is a post-filter on the hash candidate set (recall
        falls off for radii beyond the probe reach ~2/B per dimension);
        ``exact=True`` forces the exact path.

        Returns:
            One ``(ids, distances)`` pair per query, sorted by distance. The
            querying candidate itself is excluded for the default self-join.
        """
        self._check_fitted()
        Q, ex = self._resolve_queries(queries, None)
        m = Q.shape[0]

        if not self._use_lsh or exact:
            out: list[tuple[np.ndarray, np.ndarray]] = []
            block = max(1, _BRUTE_BUDGET // max(1, self.n_points * self._d))
            for s in range(0, m, block):
                e = min(m, s + block)
                D = self._pairwise(Q[s:e])
                if ex is not None:
                    D[np.arange(e - s), ex[s:e]] = np.inf
                for row in D:
                    ids = np.flatnonzero(row <= radius)
                    order = np.argsort(row[ids], kind="stable")
                    out.append((ids[order], row[ids[order]]))
            return out

        indptr, ids, dst = self._lsh.query_radius(
            np.ascontiguousarray(Q), float(radius), ex)
        return [(ids[indptr[i]:indptr[i + 1]], dst[indptr[i]:indptr[i + 1]])
                for i in range(m)]

    # ------------------------------------------------------------------ #
    # Tuning and backend construction
    # ------------------------------------------------------------------ #

    def _tune(self, radius: float | None) -> None:
        """Set (B, K, L) from the workload hints; explicit args always win."""
        n, d, k = self.n_points, self._d, self._k_hint or 2 * self._d

        if radius is not None:
            r_hat = float(radius)
        else:
            # Neighbour scale: k-th NN distance of a sample against a sample.
            q = self._arena[self._rng.choice(n, min(_TUNE_QUERIES, n), replace=False)]
            ref = self._arena[self._rng.choice(
                n, min(_TUNE_REFERENCE, n), replace=False)]
            diff = np.abs(q[:, None, :] - ref[None, :, :])
            D = np.minimum(diff, 1.0 - diff).sum(-1)
            kk = min(k, ref.shape[0] - 1)
            r_hat = float(np.median(np.partition(D, kk, axis=1)[:, kk]))
        delta = max(1e-9, r_hat / d)  # mean per-dimension neighbour distance

        # B: want per-dim collision 1-B*delta comfortably positive; high-d
        # concentration (large delta) pushes B to its coarsest value 2.
        B = self.resolution if self.resolution is not None else \
            int(np.clip(round(0.3 / delta), 2, 8))

        target = self.target_bucket_size if self.target_bucket_size is not None \
            else max(32, k)
        if self.dims_per_table is not None:
            K = self.dims_per_table
        else:
            K = int(round(np.log(max(2.0, n / target)) / np.log(B)))
        K = max(1, min(K, int(62 // np.log2(B))))  # key fits in int64

        if self.num_tables is not None:
            L = self.num_tables
        else:
            p1 = (1.0 - min(0.95, B * delta)) ** K  # per-table collision of a k-NN
            L = int(np.clip(np.ceil(np.log(0.10) / np.log(1 - min(0.99, p1 + 1e-9))),
                            4, 24)) if p1 < 1.0 else 4
        self._B, self._K, self._L = B, K, L

        self._S = self._rng.integers(0, d, size=(L, K))
        self._U = self._rng.random((L, K))
        self._pw = B ** np.arange(K, dtype=np.int64)
        logger.info("tuned: n=%d d=%d r_hat=%.3f -> B=%d K=%d L=%d",
                    n, d, r_hat, B, K, L)

    def _threshold(self) -> int:
        """Explicit brute_threshold, or the backend's measured crossover."""
        if self.brute_threshold is not None:
            return self.brute_threshold
        name, _ = resolve_backend(self.backend)
        return self._BRUTE_DEFAULTS.get(name, 4096)

    def _make_lsh(self) -> None:
        """Instantiate the backend and build both tiers from the arena."""
        name, cls = resolve_backend(self.backend)
        self.backend_name = name
        self._lsh = cls(self._B, self._K, self._L, self.probes,
                        self._S, self._U, self.query_block_size)
        self._lsh.build(np.ascontiguousarray(self._arena), self._n_static)
        logger.info("LSH backend: %s", name)

    # ------------------------------------------------------------------ #
    # Exact path
    # ------------------------------------------------------------------ #

    def _pairwise(self, Qb: np.ndarray) -> np.ndarray:
        diff = np.abs(Qb[:, None, :] - self._arena[None, :, :])
        np.minimum(diff, 1.0 - diff, out=diff)
        return diff.sum(-1)

    def _brute_query(self, Q, k, exclude_ids) -> tuple[np.ndarray, np.ndarray]:
        m, n = Q.shape[0], self.n_points
        idx = np.full((m, k), -1, dtype=np.int64)
        dst = np.full((m, k), np.inf)
        kk = min(k, n)
        block = max(1, _BRUTE_BUDGET // max(1, n * self._d))
        for s in range(0, m, block):
            e = min(m, s + block)
            D = self._pairwise(Q[s:e])
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

    # ------------------------------------------------------------------ #
    # Helpers & introspection
    # ------------------------------------------------------------------ #

    def _resolve_queries(self, queries, exclude_ids):
        if queries is None:
            if self.n_candidates == 0:
                raise ValueError("no candidate tier; pass explicit queries")
            Q = self._arena[self._n_static:]
            ex = np.arange(self._n_static, self.n_points)
        else:
            Q = np.mod(np.atleast_2d(np.asarray(queries, dtype=np.float64)), 1.0)
            if Q.shape[1] != self._d:
                raise ValueError(f"queries must have {self._d} dimensions")
            ex = None
        if exclude_ids is not None:
            ex = np.asarray(exclude_ids, dtype=np.int64)
            if ex.shape != (Q.shape[0],):
                raise ValueError("exclude_ids must have one id per query")
        return Q, ex

    def _as_batch(self, points) -> np.ndarray:
        if points is None:
            return np.empty((0, self._d))
        P = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if P.shape[1] != self._d:
            raise ValueError(f"points must have {self._d} dimensions")
        return np.mod(P, 1.0)

    def _check_fitted(self) -> None:
        if self._arena.size == 0:
            raise RuntimeError("index is empty; call fit() first")

    @property
    def n_points(self) -> int:
        return self._arena.shape[0]

    @property
    def n_static(self) -> int:
        return self._n_static

    @property
    def n_candidates(self) -> int:
        return self._arena.shape[0] - self._n_static

    @property
    def candidates(self) -> np.ndarray:
        """Current candidate-tier coordinates (view)."""
        return self._arena[self._n_static:]

    @property
    def dimensions(self) -> int:
        return self._d

    @property
    def is_approximate(self) -> bool:
        return self._use_lsh

    @property
    def n_tables(self) -> int:
        return self._L if self._use_lsh else 0
