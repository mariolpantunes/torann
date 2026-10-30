r"""torann — TORoidal Approximate Nearest Neighbours (public wrapper).

Contract:

* **Metric**: toroidal L1 on the unit torus $[0,1)^d$, and nothing else.
  The hash family's guarantee is an L1 guarantee, so L1 is the public
  contract.
* **Hash** (validated in exploration/): per sampled dimension, an integer
  resolution $B$ and offset $u \sim U[0,1)$ give the cell

  $$c(x) = \lfloor B\,((x+u) \bmod 1) \rfloor$$

  with collision probability **exactly**
  $P[c(x){=}c(y)] = \max(0,\, 1 - B\delta)$ — seamless on the torus. A
  table concatenates $K$ sampled dimensions, each with its own offset:
  a pair collides with probability $\prod_j \max(0, 1-B\delta_j) \le
  e^{-B \sum_j \delta_j}$, the L1 over the table's own coordinates.
  Averaged over the dimension draw, a pair at toroidal L1 distance $r$
  collides with probability $(1 - Br/d)^K \le e^{-BKr/d}$ while every
  $\delta_i \le 1/B$ (the `p1` that tunes $L$). A uniformly
  random pair collides per dimension with probability exactly $1/B$
  (bucket load $n/B^K$), and a point that moves by $s$ changes a cell
  with probability $B s$ (churn is proportional to movement).
* **Two tiers**: a *static tier*, hashed and key-sorted once, that grows only
  by promotion (a linear merge, never a re-sort), and a small *candidate
  tier* refreshed per epoch. Queries default to "each candidate against
  everything" — the ESS inner loop.
* **Selective updates**: ``update`` re-places only the points that moved far
  enough to change a cell, by delete + merge — never a full re-sort. The
  index is always exact after an update: fast *and* accurate.
* **No brute-force fallback in LSH mode**: an under-filled query widens its
  buckets by *prefix relaxation* (keys are digit-concatenations; dropping
  low-order digits turns a bucket into a contiguous sorted-key range), so k
  results are structurally guaranteed without a distance scan.
* **Tuning**: ``fit(..., k=...)`` or ``radius=...`` feeds ESS's own query
  heuristic into the index — $B$ from the neighbour scale, $K$ from the
  closed-form bucket load $K = \operatorname{round}(\log_B(n/\text{target}))$,
  $L$ from a recall target and the per-dimension gaps of sampled neighbours,
  which predict each table's hit rate on the real hash and probe rule for
  any $p$. Explicit constructor arguments always win over tuning.

This class owns validation, tuning, hash-parameter drawing and *selection*;
the search itself lives in interchangeable implementations of
``base.BaseIndex``: exact brute force (``brute.py``) below the measured
crossover threshold, and above it an LSH implementation — Rust
(``torann._native``) when the compiled module is present, the pure-Python
reference (``lsh.py``) otherwise — ``lsh.py`` is the specification the
native core is validated against, byte-identical tables included.

**Input contract** (checked here, before any state changes, so a rejected
call leaves the index as it was): coordinates may be any finite real
numbers and are reduced mod 1, with a result of exactly ``1.0`` stored as
``0.0`` (the same torus point); NaN or inf raises ``ValueError``. Every
result row is sorted by distance, equal distances by ascending id, on
every backend and path -- callers may rely on that order.

Note the compiled core is gated on AVX2 (see ``torann.rust``): published
x86-64 wheels are built with that floor, and a CPU without it falls back
to ``lsh.py`` rather than crashing.
"""

from __future__ import annotations

import logging
from math import comb
from time import perf_counter
from typing import Any, ClassVar, Literal, overload

import numpy as np

from . import rust
from .base import IdArray, Neighbours, csr_to_padded
from .brute import BruteIndex, knn_scan, radius_scan
from .lsh import PythonLshIndex


def _to_torus(points: Any) -> np.ndarray:
    """Coordinates as float64 on the unit torus, ``[0, 1)``.

    Every input reaches the index through here. Non-finite values are
    refused before anything is stored. ``np.mod`` returns exactly ``1.0``
    for tiny negatives (``-1e-20``), which is the same torus point as
    ``0.0``, so it is stored as ``0.0``: the arena never holds a ``1.0``.
    """
    X = np.asarray(points, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("coordinates must be finite (no NaN or inf)")
    X = np.mod(X, 1.0)
    X[X == 1.0] = 0.0
    return X


logger = logging.getLogger(__name__)

__all__ = ["ToroidalNN", "available_backends", "tune_thresholds"]

# Sampling caps for the tune() estimate of the neighbour scale.
_TUNE_QUERIES = 256
_TUNE_REFERENCE = 8192
# recall="auto" targets for the k-th neighbour, calibrated so the defaults keep
# the recall they had under the older, pessimistic model, which over-built p < 1.
_AUTO_RECALL = 0.94
_AUTO_RECALL_LP = 0.98
_MAX_TABLES = 24
_WALL_GRID = 64

_AUTO_ORDER = ("rust", "python")


def _lsh_class(name: str):
    """The LSH implementation class for ``name``; ImportError if absent."""
    if name == "python":
        return PythonLshIndex
    if name == "rust":
        # Test the class itself, not the parallel `AVAILABLE` flag: they can
        # drift, and the class is what the caller is about to instantiate.
        if rust.RustLshIndex is None:
            raise ImportError(
                "torann._native is not available (not built, or this CPU is "
                "below the AVX2 floor the published wheels are built with)"
            )
        return rust.RustLshIndex
    raise ValueError(f"unknown backend {name!r}; expected 'auto', {', '.join(map(repr, _AUTO_ORDER))}")


def _torus_gaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-dimension toroidal distance of two point arrays."""
    diff = np.abs(a - b)
    return np.minimum(diff, 1.0 - diff)


def _ball_gaps(q: np.ndarray, ref: np.ndarray, radius: float, p: float) -> np.ndarray:
    """Gaps to the farthest other point within ``radius`` of each query: its hardest neighbour."""
    indptr, ids, dst = radius_scan(ref, q, radius, None, p=p)
    far = np.empty(q.shape[0], dtype=np.int64)
    found = np.zeros(q.shape[0], dtype=bool)
    for i in range(q.shape[0]):
        row = indptr[i + 1] - 1
        if row >= indptr[i] and dst[row] > 0.0:
            far[i] = ids[row]
            found[i] = True
    if not found.any():
        return np.zeros((1, q.shape[1]))
    return _torus_gaps(q[found], ref[far[found]])


def _wall_hit_rate(gaps: np.ndarray, B: int, K: int, m: int) -> np.ndarray:
    r"""Per-pair probability that one table finds the pair, for ``gaps`` (pairs, d).

    A table draws its K dimensions i.i.d. from the pair's d gaps, so with
    $u = B\,\delta$ and $G(t) = P[u > t]$ over those d values, a query at
    distance $w \sim U[0, 1/2]$ from its nearest wall keeps the neighbour in
    its cell with probability $s(w) = 1 - (G(w) + G(1-w))/2$, and has it one
    step across that wall with probability $a(w) = (G(w) - G(1+w))/2$; at
    $B = 2$ both walls lead to the same cell, so $a(w) = 1 - s(w)$.

    A probe flips one of the m closest walls, so the pair is found in the base
    bucket, or when exactly one dimension differs, across its nearest wall,
    and at most m-1 of the other dimensions have a closer wall:

    $$P = \bar s^K + K \int_0^{1/2} 2\,a(w) \sum_{i<m} \binom{K-1}{i}
          A(w)^i D(w)^{K-1-i}\, dw$$

    with $A(w) = \int_0^w 2 s$, $D(w) = \int_w^{1/2} 2 s$ and $\bar s = A(1/2)$.
    See exploration/exp_tune_model.py for the check against the real hash.
    """
    u = B * gaps
    w = (np.arange(_WALL_GRID) + 0.5) / (2 * _WALL_GRID)
    dw = 0.5 / _WALL_GRID

    def tail(t: np.ndarray) -> np.ndarray:
        return (u[:, :, None] > t[None, None, :]).mean(axis=1)

    near, far = tail(w), tail(1.0 - w)
    same = 1.0 - 0.5 * (near + far)
    adjacent = 1.0 - same if B == 2 else 0.5 * (near - tail(1.0 + w))
    mass = 2.0 * same * dw
    closer = np.cumsum(mass, axis=1) - 0.5 * mass
    total = mass.sum(axis=1)
    if m == 0:
        return total**K
    farther = total[:, None] - closer
    ways = sum(comb(K - 1, i) * closer**i * farther ** (K - 1 - i) for i in range(m))
    return total**K + K * (2.0 * adjacent * ways * dw).sum(axis=1)


def _resolve(name: str = "auto"):
    """``(name, class)``, resolving 'auto' to the fastest installed."""
    if name != "auto":
        return name, _lsh_class(name)
    for cand in _AUTO_ORDER:
        try:
            return cand, _lsh_class(cand)
        except ImportError:
            continue
    raise ImportError("no LSH implementation available (not even 'python')")


def available_backends() -> list[str]:
    """Names of the LSH implementations importable in this environment."""
    out = []
    for name in _AUTO_ORDER:
        try:
            _lsh_class(name)
        except ImportError:
            continue
        out.append(name)
    return out


def _validate_threshold(val: int | None, name: str) -> int | None:
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
        raise TypeError(f"{name} must be an integer >= 0 or None, got {type(val).__name__}")
    out = int(val)
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {out}")
    return out


class ToroidalNN:
    r"""Exact + approximate toroidal L1 k-NN with an ESS-shaped lifecycle.

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
            4096 for python, 512 for rust.
        num_tables: Tables $L$. ``None`` = tuned (default 16 without hints).
        resolution: Cells per dimension $B \ge 2$. ``None`` = tuned
            (default 3).
        dims_per_table: Sampled dimensions $K$. ``None`` = closed-form
            bucket load $K = \operatorname{round}(\log_B(n/\text{target}))$.
        target_bucket_size: Bucket-load target for the $K$ rule. ``None`` =
            ``max(32, k_hint)``.
        probes: Neighbour-cell probes per table per query.
        query_block_size: Queries per block of the batched join. ``None``
            (default) is one block per worker, which is what bucket reuse
            wants: a bucket is loaded once per block and scored against
            every query in it. Two attempts to raise that reuse further —
            partitioning the work by bucket, and reordering queries by
            hash key — were built and both lost to their own overhead, so
            this sizing is deliberate rather than a placeholder. ``1``
            selects the per-query path (one deduplicated candidate list
            per query), which is what small batches want, and is also the
            independent reference the batched join is validated against.
            The join serves `query` from 64 queries per call and
            `query_radius` from 64 queries *per block* at ``d <= 32``; above
            that width the radius scan stays faster (the join scores a point
            once per table it shares with the query), and it returns
            bit-identical results either way.
        backend: LSH implementation — ``"auto"`` (first installed of rust,
            python), or an explicit name.
        recall: Target chance in ``(0, 1)`` of finding a query's k-th nearest
            neighbour, the hardest of its k, used to size the number of tables
            $L$. ``"auto"`` (default) is 0.94 for L1 and 0.98 for $p < 1$.
            `expected_recall` reports what the tuned index predicts; it falls
            short of the target where $L$ reaches its cap of 24.
        p: Toroidal $L^p$ exponent in ``(0, 1]`` (default ``1.0``, toroidal L1).
        brute_threshold_knn: Mode-specific override for :meth:`query` exact/LSH
            crossover (takes precedence over ``brute_threshold``).
        brute_threshold_radius: Mode-specific override for :meth:`query_radius`
            exact/LSH crossover (takes precedence over ``brute_threshold``).
    """

    # Measured brute/LSH crossover per backend and query mode:
    # - For k-NN (p = 1.0, k = 2*d), the Rust heap brute scan stays competitive
    #   at n <= 3072 on moderate/high dimensions, while LSH wins from n = 4096
    #   across all dimensions d in [2, 200] (1.1x-5.0x).
    # - For range search (`query_radius`), the batched LSH radius join + exact
    #   u8 SIMD prefilter wins across all dimensions from n = 2048 (1.2x-3.0x
    #   at n = 2048, 1.8x-4.2x at n = 3072), while brute ties/wins at n <= 1024.
    # - For fractional toroidal L^p (p < 1.0), exact brute pays fractional powers
    #   or sqrt on every coordinate pair and LSH wins across all dimensions from
    #   n > 1024 for both k-NN and radius queries.
    _BRUTE_DEFAULTS_KNN: ClassVar[dict[str, int]] = {"python": 12288, "rust": 3072}
    _BRUTE_DEFAULTS_RADIUS: ClassVar[dict[str, int]] = {"python": 12288, "rust": 1024}
    _BRUTE_DEFAULTS_LP: ClassVar[dict[str, int]] = {"python": 12288, "rust": 1024}
    _BRUTE_DEFAULTS: ClassVar[dict[str, int]] = _BRUTE_DEFAULTS_KNN

    def __init__(
        self,
        seed: int | None = None,
        brute_threshold: int | None = None,
        num_tables: int | None = None,
        resolution: int | None = None,
        dims_per_table: int | None = None,
        target_bucket_size: int | None = None,
        probes: int = 4,
        query_block_size: int | None = None,
        backend: str = "auto",
        recall: float | str = "auto",
        p: float = 1.0,
        brute_threshold_knn: int | None = None,
        brute_threshold_radius: int | None = None,
    ):
        if resolution is not None and resolution < 2:
            raise ValueError("resolution must be >= 2")
        if query_block_size is not None and query_block_size < 1:
            raise ValueError("query_block_size must be >= 1")
        if num_tables is not None and num_tables < 1:
            raise ValueError("num_tables must be >= 1")
        if dims_per_table is not None and dims_per_table < 1:
            raise ValueError("dims_per_table must be >= 1")
        if probes < 0:
            raise ValueError("probes must be >= 0")
        if recall == "auto":
            self.recall: float | str = "auto"
        elif (
            not isinstance(recall, bool)
            and isinstance(recall, (int, float))
            and np.isfinite(recall)
            and 0.0 < float(recall) < 1.0
        ):
            self.recall = float(recall)
        else:
            raise ValueError(f"recall must be 'auto' or a float in (0, 1), got {recall!r}")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p) or not (0.0 < float(p) <= 1.0):
            raise ValueError(f"p must be a float in (0, 1], got {p!r}")

        self.p = float(p)
        self.brute_threshold = _validate_threshold(brute_threshold, "brute_threshold")
        self.brute_threshold_knn = _validate_threshold(brute_threshold_knn, "brute_threshold_knn")
        self.brute_threshold_radius = _validate_threshold(brute_threshold_radius, "brute_threshold_radius")
        self.num_tables = num_tables
        self.resolution = resolution
        self.dims_per_table = dims_per_table
        self.target_bucket_size = target_bucket_size
        self.probes = int(probes)
        self.query_block_size = None if query_block_size is None else int(query_block_size)
        self.backend = backend
        self.backend_name: str | None = None

        self._rng = np.random.default_rng(seed)
        self._arena = np.empty((0, 0))
        self._expected_recall: float | None = None
        self._n_static = 0
        self._d = 0
        self._use_lsh = False
        self._k_hint: int | None = None
        self._radius_hint: float | None = None
        self._impl: Any = None  # BaseIndex instance; set at fit()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def fit(
        self,
        static_points: np.ndarray,
        candidate_points: np.ndarray | None = None,
        k: int | None = None,
        radius: float | None = None,
    ) -> ToroidalNN:
        """Build the index from zero. Hash functions are drawn (and tuned)
        here and kept stable for the whole run.

        Args:
            static_points: (n, d) anchors; never move. May be empty
                (shape ``(0, d)``) when ``candidate_points`` is given —
                the from-scratch case, where the first batch has nothing
                to anchor against but itself.
            candidate_points: (c, d) points that move each epoch. Optional.
            k: The k the workload will query (ESS: 2*d). Drives tuning.
            radius: Alternative tuning hint: the range-query radius.
        """
        S = np.atleast_2d(_to_torus(static_points))
        if S.ndim != 2:
            raise ValueError("static_points must be an (n, d) array")
        C = np.atleast_2d(_to_torus(candidate_points)) if candidate_points is not None else None
        if S.shape[0] == 0:
            if C is None or C.shape[0] == 0:
                raise ValueError("fit needs at least one point (static or candidate)")
            S = np.empty((0, C.shape[1]))
        if C is None:
            C = np.empty((0, S.shape[1]))
        if C.ndim != 2 or C.shape[1] != S.shape[1]:
            raise ValueError(f"points must have {S.shape[1]} dimensions")
        self._d = S.shape[1]
        self._arena = np.vstack([S, C])
        self._n_static = S.shape[0]
        self._k_hint = int(k) if k is not None else 2 * self._d
        self._radius_hint = float(radius) if radius is not None else None

        fit_thr = self._threshold("radius") if self._radius_hint is not None else self._threshold("knn")
        if self.n_points > fit_thr:
            self._build_lsh(self._radius_hint)
        else:
            self._use_lsh = False
            self._impl = BruteIndex(p=self.p)
            self._impl.build(self._arena, self._n_static)
        return self

    def update(self, coordinates: np.ndarray) -> None:
        """Move the candidate tier (one epoch step).

        In LSH mode the refresh is *selective*: only points whose key
        actually changed — i.e. that moved far enough to leave a cell — are
        re-placed, by delete + merge. The index is exact after every update.

        Args:
            coordinates: (n_candidates, d) new positions, same order as the
                current candidate tier.
        """
        self._check_fitted()
        new = np.atleast_2d(_to_torus(coordinates))
        if new.shape != (self.n_candidates, self._d):
            raise ValueError(f"expected coordinates of shape {(self.n_candidates, self._d)}")
        self._arena[self._n_static :] = new
        self._impl.update(np.ascontiguousarray(new))

    def promote(self, new_candidates: np.ndarray | None = None) -> None:
        """Freeze the candidate tier into the static tier and install a new
        batch. The merge is linear (searchsorted + insert), never a re-sort.
        """
        self._check_fitted()
        C = self._as_batch(new_candidates)
        self._impl.promote(np.ascontiguousarray(C))
        self._n_static = self.n_points  # old candidates are static now
        if C.size:
            self._arena = np.vstack([self._arena, C])
        fit_thr = self._threshold("radius") if self._radius_hint is not None else self._threshold("knn")
        if not self._use_lsh and self.n_points > fit_thr:
            # Crossed the threshold: first (and only) full LSH build.
            self._build_lsh(self._radius_hint)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def query(
        self,
        k: int | None = None,
        queries: np.ndarray | None = None,
        exclude_ids: IdArray | None = None,
    ) -> Neighbours:
        """Batch toroidal-L1 k-NN.

        With no arguments this is the ESS inner loop: every candidate point
        queries the whole index (both tiers), excluding itself.

        Args:
            k: Neighbours per query, an integer >= 1 (``int`` or a NumPy
                integer; a float or bool raises ``TypeError``); defaults
                to the fit-time hint (2*d).
            queries: Optional (m, d) explicit queries, or one (d,) query;
                default is the candidate tier.
            exclude_ids: Optional per-query point id to exclude, each in
                ``[-1, n_points)``; ``-1`` excludes nothing (set
                automatically for the default self-join).

        Returns:
            `Neighbours` of shape (m, k), sorted by distance with equal
            distances in ascending id order; rows are padded with
            -1 / inf only when fewer than k points exist.
            In LSH mode under-filled queries are completed by prefix
            relaxation, so k results are guaranteed without a brute scan.
        """
        self._check_fitted()
        if k is not None and (isinstance(k, bool) or not isinstance(k, (int, np.integer))):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        kq = int(k) if k is not None else (self._k_hint or 2 * self._d)
        if kq < 1:
            raise ValueError("k must be >= 1")
        Q, ex = self._resolve_queries(queries, exclude_ids)
        if self.is_approximate_knn:
            if not self._use_lsh:
                self._build_lsh(self._radius_hint)
            return Neighbours(*self._impl.query_knn(np.ascontiguousarray(Q), kq, ex))
        if self._use_lsh:
            ids, dst = knn_scan(self._arena, Q, kq, ex, p=self.p)
            return Neighbours(ids, dst)
        return Neighbours(*self._impl.query_knn(np.ascontiguousarray(Q), kq, ex))

    # `pad` chooses between two unrelated shapes, so it is overloaded rather
    # than left as a union: without this every caller of the padded form has
    # to prove to a type checker that it did not get the list, which is a
    # cast at each site to restate what the literal already says.
    @overload
    def query_radius(
        self,
        radius: float,
        queries: np.ndarray | None = ...,
        exact: bool = ...,
        *,
        pad: Literal[True],
    ) -> Neighbours: ...

    @overload
    def query_radius(
        self,
        radius: float,
        queries: np.ndarray | None = ...,
        exact: bool = ...,
        pad: Literal[False] = ...,
    ) -> list[Neighbours]: ...

    def query_radius(
        self,
        radius: float,
        queries: np.ndarray | None = None,
        exact: bool = False,
        pad: bool = False,
    ) -> list[Neighbours] | Neighbours:
        """Batch range query: indexed points within toroidal L1 ``radius``.

        In LSH mode this is a post-filter on the hash candidate set (recall
        falls off for radii beyond the probe reach ~2/B per dimension);
        ``exact=True`` forces the exact path.

        Args:
            radius: Inclusive toroidal-L1 cutoff (closed ball), ``>= 0``;
                ``inf`` returns every point, NaN or a negative value
                raises ``ValueError``.
            queries: Optional (m, d) explicit queries; default is the
                candidate tier.
            exact: Force the exact scan even in LSH mode.
            pad: Return the dense ``(m, width)`` `Neighbours` instead of a
                list — ``-1`` / ``inf`` padded, exactly what :meth:`query`
                returns,
                so a caller can switch between k-NN and radius without
                reshaping anything. Prefer it: the list costs a Python pass
                over ``m`` to build and almost always a second one to
                consume, and against the compiled kernel those two passes are
                a third of the call.

        Returns:
            With ``pad``: one `Neighbours` of shape ``(m, width)``.
            Otherwise one `Neighbours` per query, ragged and sorted by
            distance, equal distances by ascending id. The querying
            candidate itself is excluded for the default self-join.
        """
        self._check_fitted()
        radius = float(radius)
        if not radius >= 0.0:  # also catches NaN
            raise ValueError(f"radius must be >= 0 (inf allowed), got {radius}")
        Q, ex = self._resolve_queries(queries, None)
        if not exact and self.is_approximate_radius:
            if not self._use_lsh:
                self._build_lsh(radius)
            indptr, ids, dst = self._impl.query_radius(np.ascontiguousarray(Q), radius, ex)
        elif self._use_lsh:
            indptr, ids, dst = radius_scan(self._arena, Q, radius, ex, p=self.p)
        else:
            indptr, ids, dst = self._impl.query_radius(np.ascontiguousarray(Q), radius, ex)
        m = Q.shape[0]
        if pad:
            return csr_to_padded(indptr, ids, dst, m)
        return [Neighbours(ids[indptr[i] : indptr[i + 1]], dst[indptr[i] : indptr[i + 1]]) for i in range(m)]

    # ------------------------------------------------------------------ #
    # Tuning and implementation selection
    # ------------------------------------------------------------------ #

    def _tune(self, radius: float | None) -> None:
        """Set (B, K, L) from the workload hints; explicit args always win."""
        n, d, k = self.n_points, self._d, self._k_hint or 2 * self._d

        q = self._arena[self._rng.choice(n, min(_TUNE_QUERIES, n), replace=False)]
        ref = self._arena[self._rng.choice(n, min(_TUNE_REFERENCE, n), replace=False)]
        if radius is not None:
            r_hat = float(radius)
            delta = (
                max(1e-9, r_hat / d) if self.p == 1.0 else max(1e-9, (r_hat / (d ** (1.0 / self.p))) * (self.p**-0.32))
            )
            gaps = None if self.num_tables is not None else _ball_gaps(q, ref, r_hat, self.p)
        else:
            # Neighbour scale: k-th NN distance of a sample against a sample.
            # This is a k-NN query, so it is served by the k-NN kernel.
            #
            # It used to compute the full (256 x 8192) distance block in
            # NumPy and partition it, and that was **95% of an LSH fit** --
            # 1.975 s of 2.078 s at n=16384, d=100, against 31 ms for the
            # Rust index build it exists to configure. The LSH machinery was
            # never the slow part; choosing its radius was.
            #
            # The per-dimension gaps come from the neighbours' ids, not from
            # a distance: a hash collision depends on the 1-D gaps whatever
            # the metric that ranked the neighbour.
            #
            # `r_hat` is quantised because the two paths must agree exactly.
            # `brute_knn` reassociates (the algebraic float methods) where
            # `exact_knn` sums in NumPy's order, so they differ in the last
            # bits -- and `r_hat` feeds `round(0.3 / delta)`, where a boundary
            # case would pick a different `B` and build entirely different
            # tables. A wheel with the compiled kernel would then disagree
            # with a pure-Python install, which is the one contract this file
            # cannot break. Twelve digits is far below anything a radius
            # estimated from a random sample can resolve and far above where
            # the two paths part.
            kk = min(k, ref.shape[0] - 1)
            ids = knn_scan(ref, q, kk + 1, p=self.p)[0]
            r_hat = float(f"{float(np.median(_torus_gaps(q, ref[ids[:, kk]]).sum(axis=1))):.12e}")
            delta = max(1e-9, r_hat / d)  # mean per-dimension neighbour distance
            # `ref` is a sample of the index, so its k-th neighbour is farther
            # out than the index's: the same rank scaled by the sampling rate.
            rank = int(np.clip(round(k * ref.shape[0] / n), 1, kk))
            gaps = _torus_gaps(q, ref[ids[:, rank]])

        # B: want per-dim collision 1-B*delta comfortably positive; high-d
        # concentration (large delta) pushes B to its coarsest value 2.
        B = self.resolution if self.resolution is not None else int(np.clip(round(0.3 / delta), 2, 8))

        target = self.target_bucket_size if self.target_bucket_size is not None else max(32, k)
        if self.dims_per_table is not None:
            K = self.dims_per_table
        else:
            K = round(np.log(max(2.0, n / target)) / np.log(B))
        K = max(1, min(K, int(62 // np.log2(B))))  # key fits in int64

        if self.num_tables is not None:
            L = self.num_tables
        else:
            auto = _AUTO_RECALL if self.p == 1.0 else _AUTO_RECALL_LP
            goal = auto if self.recall == "auto" else float(self.recall)
            assert gaps is not None  # set on every path that reaches tuning L
            L = self._tables_for(gaps, B, K, goal)
        self._B, self._K, self._L = B, K, L

        self._S = self._rng.integers(0, d, size=(L, K))
        self._U = self._rng.random((L, K))
        self._pw = B ** np.arange(K, dtype=np.int64)
        logger.info("tuned: n=%d d=%d r_hat=%.3f -> B=%d K=%d L=%d", n, d, r_hat, B, K, L)

    def _tables_for(self, gaps: np.ndarray, B: int, K: int, target: float) -> int:
        """Fewest tables whose predicted recall of the sampled pairs reaches ``target``.

        The result is capped at `_MAX_TABLES`: past it the target is out of
        reach, and `expected_recall` reports what the index can give instead.
        """
        l_min = 4 if self.recall == "auto" else 1
        hit = _wall_hit_rate(gaps, B, K, min(self.probes, K))
        tables = np.arange(1, _MAX_TABLES + 1)
        # Rounded so the last bits of a sampled gap cannot move L across backends.
        reach = np.round((1.0 - (1.0 - hit[None, :]) ** tables[:, None]).mean(axis=1), 9)
        L = int(max(l_min, np.searchsorted(reach, target) + 1))
        L = min(L, _MAX_TABLES)
        self._expected_recall = float(reach[L - 1])
        return L

    def _threshold(self, mode: Literal["knn", "radius"] | None = None) -> int:
        """Mode-specific or global brute_threshold, else the backend's crossover."""
        if mode == "knn" and self.brute_threshold_knn is not None:
            return self.brute_threshold_knn
        if mode == "radius" and self.brute_threshold_radius is not None:
            return self.brute_threshold_radius
        if self.brute_threshold is not None:
            return self.brute_threshold
        name, _ = _resolve(self.backend)
        if self.p < 1.0:
            return self._BRUTE_DEFAULTS_LP.get(name, 1024)
        if mode == "radius":
            return self._BRUTE_DEFAULTS_RADIUS.get(name, 1024)
        if mode == "knn":
            return self._BRUTE_DEFAULTS_KNN.get(name, 3072)
        return self._BRUTE_DEFAULTS.get(name, 3072)

    def _build_lsh(self, radius: float | None) -> None:
        """Tune and instantiate the LSH backend from the current arena."""
        self._use_lsh = True
        self._tune(radius)
        self._make_lsh()

    def _make_lsh(self) -> None:
        """Instantiate the LSH implementation and build both tiers."""
        name, cls = _resolve(self.backend)
        self.backend_name = name
        if self.p == 1.0:
            self._impl = cls(self._B, self._K, self._L, self.probes, self._S, self._U, self.query_block_size or 0)
        else:
            self._impl = cls(
                self._B, self._K, self._L, self.probes, self._S, self._U, self.query_block_size or 0, self.p
            )
        self._impl.build(np.ascontiguousarray(self._arena), self._n_static)
        logger.info("LSH implementation: %s", name)

    # ------------------------------------------------------------------ #
    # Helpers & introspection
    # ------------------------------------------------------------------ #

    def _resolve_queries(self, queries, exclude_ids):
        if queries is None:
            if self.n_candidates == 0:
                raise ValueError("no candidate tier; pass explicit queries")
            Q = self._arena[self._n_static :]
            ex = np.arange(self._n_static, self.n_points)
        else:
            Q = np.atleast_2d(_to_torus(queries))
            if Q.ndim != 2:
                raise ValueError(f"queries must be (m, d) or (d,); got shape {Q.shape}")
            if Q.shape[1] != self._d:
                raise ValueError(f"queries must have {self._d} dimensions")
            ex = None
        if exclude_ids is not None:
            ex = np.asarray(exclude_ids, dtype=np.int64)
            if ex.shape != (Q.shape[0],):
                raise ValueError("exclude_ids must have one id per query")
            if ex.size and (ex.min() < -1 or ex.max() >= self.n_points):
                raise ValueError(f"exclude_ids must be in [-1, {self.n_points}); -1 excludes nothing")
        return Q, ex

    def _as_batch(self, points) -> np.ndarray:
        if points is None:
            return np.empty((0, self._d))
        P = np.atleast_2d(_to_torus(points))
        if P.ndim != 2 or P.shape[1] != self._d:
            raise ValueError(f"points must have {self._d} dimensions")
        return P

    def _check_fitted(self) -> None:
        if self._arena.size == 0:
            raise RuntimeError("index is empty; call fit() first")

    @property
    def n_points(self) -> int:
        """Total indexed points (both tiers)."""
        return self._arena.shape[0]

    @property
    def n_static(self) -> int:
        """Points in the static tier (anchors; never move)."""
        return self._n_static

    @property
    def n_candidates(self) -> int:
        """Points in the candidate tier (moved by ``update``)."""
        return self._arena.shape[0] - self._n_static

    @property
    def candidates(self) -> np.ndarray:
        """Current candidate-tier coordinates (view)."""
        return self._arena[self._n_static :]

    @property
    def dimensions(self) -> int:
        """Dimensionality ``d`` of the indexed space."""
        return self._d

    @property
    def is_approximate(self) -> bool:
        """True in LSH mode, False while the exact brute path serves."""
        return self._use_lsh

    @property
    def is_approximate_knn(self) -> bool:
        """True when :meth:`query` routes through the LSH index (``n_points > threshold('knn')``)."""
        return self._arena.size > 0 and self.n_points > self._threshold("knn")

    @property
    def is_approximate_radius(self) -> bool:
        """True when :meth:`query_radius` routes through the LSH index (``n_points > threshold('radius')``)."""
        return self._arena.size > 0 and self.n_points > self._threshold("radius")

    @property
    def expected_recall(self) -> float | None:
        """Predicted chance of finding a query's hardest neighbour, None while exact or with explicit tables."""
        return self._expected_recall

    @property
    def n_tables(self) -> int:
        """Hash tables in use (0 in exact mode)."""
        return self._L if self._use_lsh else 0


def tune_thresholds(
    points: np.ndarray,
    queries: np.ndarray | None = None,
    *,
    k: int | None = None,
    radius: float | None = None,
    p: float = 1.0,
    recall: float | str = "auto",
    backend: str = "auto",
    repeats: int = 3,
    seed: int = 0,
) -> dict[str, int]:
    """Measure brute vs LSH query latency on a workload and return threshold overrides.

    Compares exact brute-force search against the configured LSH backend on
    ``points`` (and ``queries``, or a candidate self-join slice when
    ``queries is None``) for both k-NN and range search, returning a dict
    ``{"brute_threshold_knn": ..., "brute_threshold_radius": ...}`` ready to
    splat into :class:`ToroidalNN`:

    * ``0`` when LSH is faster than exact brute at ``len(points)``, activating
      LSH immediately at this size.
    * ``len(points)`` when exact brute is faster at ``len(points)``, keeping
      exact search active at this size.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or int(repeats) < 1:
        raise ValueError(f"repeats must be an integer >= 1, got {repeats!r}")
    rep = int(repeats)
    P = np.atleast_2d(_to_torus(points))
    if P.ndim != 2 or P.shape[0] < 2:
        raise ValueError("points must be an (n, d) array with n >= 2")
    n, d = P.shape
    kq = int(k) if k is not None else min(2 * d, n - 1)
    if kq < 1:
        raise ValueError("k must be >= 1")
    if queries is not None:
        Q = np.atleast_2d(_to_torus(queries))
        if Q.ndim != 2 or Q.shape[1] != d:
            raise ValueError(f"queries must have {d} dimensions")
        static_pts, cand_pts, q_arg = P, None, Q
    else:
        m = min(256, max(1, n // 4))
        static_pts, cand_pts, q_arg = P[: n - m], P[n - m :], None

    brute_nn = ToroidalNN(seed=seed, brute_threshold=n + 1, backend=backend, recall=recall, p=p).fit(
        static_pts, cand_pts, k=kq
    )
    lsh_nn = ToroidalNN(seed=seed, brute_threshold=0, backend=backend, recall=recall, p=p).fit(
        static_pts, cand_pts, k=kq, radius=radius
    )

    brute_knn = float("inf")
    b_dst = np.empty((0, 0))
    for _ in range(rep):
        t0 = perf_counter()
        _, b_dst = brute_nn.query(k=kq, queries=q_arg)
        brute_knn = min(brute_knn, perf_counter() - t0)

    lsh_knn = float("inf")
    for _ in range(rep):
        t0 = perf_counter()
        lsh_nn.query(k=kq, queries=q_arg)
        lsh_knn = min(lsh_knn, perf_counter() - t0)

    r_q = float(radius) if radius is not None else float(np.median(b_dst[:, min(kq - 1, b_dst.shape[1] - 1)]))
    brute_rad = float("inf")
    for _ in range(rep):
        t0 = perf_counter()
        brute_nn.query_radius(r_q, queries=q_arg, pad=True)
        brute_rad = min(brute_rad, perf_counter() - t0)

    lsh_rad = float("inf")
    for _ in range(rep):
        t0 = perf_counter()
        lsh_nn.query_radius(r_q, queries=q_arg, pad=True)
        lsh_rad = min(lsh_rad, perf_counter() - t0)

    return {
        "brute_threshold_knn": 0 if lsh_knn < brute_knn else n,
        "brute_threshold_radius": 0 if lsh_rad < brute_rad else n,
    }
