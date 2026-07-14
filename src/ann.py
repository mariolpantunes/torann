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

Below ``brute_threshold`` total points everything is answered by the exact
blocked brute-force path.
"""

from __future__ import annotations

import logging

import numpy as np

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
            nn.update(new)                        # drift-budgeted refresh
        nn.promote(next_batch)                    # candidates -> static

    Args:
        seed: Seed for hash functions and tuning samples.
        brute_threshold: Exact search while total points <= this.
        num_tables: Tables L. ``None`` = tuned (default 16 without hints).
        resolution: Cells per dimension B >= 2. ``None`` = tuned (default 3).
        dims_per_table: Sampled dimensions K. ``None`` = closed-form bucket
            load ``K = round(log_B(n / target_bucket_size))``.
        target_bucket_size: Bucket-load target for the K rule. ``None`` =
            ``max(32, k_hint)``.
        probes: Neighbour-cell probes per table per query.
        query_block_size: Queries per vectorised block.
    """

    def __init__(
        self,
        seed: int | None = None,
        brute_threshold: int = 4096,
        num_tables: int | None = None,
        resolution: int | None = None,
        dims_per_table: int | None = None,
        target_bucket_size: int | None = None,
        probes: int = 4,
        query_block_size: int = 1024,
    ):
        if resolution is not None and resolution < 2:
            raise ValueError("resolution must be >= 2")
        if num_tables is not None and num_tables < 1:
            raise ValueError("num_tables must be >= 1")
        if dims_per_table is not None and dims_per_table < 1:
            raise ValueError("dims_per_table must be >= 1")
        if probes < 0:
            raise ValueError("probes must be >= 0")

        self.brute_threshold = int(brute_threshold)
        self.num_tables = num_tables
        self.resolution = resolution
        self.dims_per_table = dims_per_table
        self.target_bucket_size = target_bucket_size
        self.probes = int(probes)
        self.query_block_size = int(query_block_size)

        self._rng = np.random.default_rng(seed)
        self._arena = np.empty((0, 0))
        self._n_static = 0
        self._d = 0
        self._use_lsh = False
        self._k_hint: int | None = None

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

        self._use_lsh = self.n_points > self.brute_threshold
        if self._use_lsh:
            self._pts32 = self._arena.astype(np.float32)
            self._tune(radius)
            self._build_static()
            self._build_candidates()
        return self

    def update(self, coordinates: np.ndarray) -> None:
        """Move the candidate tier (one epoch step).

        Coordinates are stored immediately (refinement always sees current
        positions) and the refresh is *selective*: only points whose key
        actually changed — i.e. that moved far enough to leave a cell — are
        re-placed, by delete + merge. The index is exact after every update.

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
        if not self._use_lsh:
            return
        self._pts32[self._n_static:] = new.astype(np.float32)
        self._refresh_candidates()

    def promote(self, new_candidates: np.ndarray | None = None) -> None:
        """Freeze the candidate tier into the static tier and install a new
        batch. The merge is linear (searchsorted + insert), never a re-sort;
        candidate keys are rehashed exactly first, since they become static
        for the rest of the run.
        """
        self._check_fitted()
        C = self._as_batch(new_candidates)
        if self._use_lsh and self.n_candidates:
            # candidate keys are always fresh (update() is exact)
            skeys = np.empty((self._L, self.n_points), dtype=np.int64)
            sids = np.empty((self._L, self.n_points), dtype=np.int64)
            for t in range(self._L):
                p = np.searchsorted(self._skeys_s[t], self._skeys_c[t])
                skeys[t] = np.insert(self._skeys_s[t], p, self._skeys_c[t])
                sids[t] = np.insert(self._sids_s[t], p, self._sids_c[t])
            self._skeys_s, self._sids_s = skeys, sids

        self._n_static = self.n_points  # old candidates are static now
        if C.size:
            self._arena = np.vstack([self._arena, C])
        if self._use_lsh:
            if C.size:
                self._pts32 = np.vstack([self._pts32, C.astype(np.float32)])
            self._build_candidates()
        elif self.n_points > self.brute_threshold:
            # Crossed the threshold: first (and only) full LSH build.
            self._use_lsh = True
            self._pts32 = self._arena.astype(np.float32)
            self._tune(None)
            self._build_static()
            self._build_candidates()

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
        m = Q.shape[0]

        if not self._use_lsh:
            return self._brute_query(Q, kq, ex)

        idx = np.full((m, kq), -1, dtype=np.int64)
        dst = np.full((m, kq), np.inf)
        bs = self.query_block_size
        for s in range(0, m, bs):
            e = min(m, s + bs)
            self._lsh_block(Q[s:e], kq, None if ex is None else ex[s:e],
                            idx[s:e], dst[s:e])

        reachable = self.n_points - (0 if ex is None else 1)
        kk = min(kq, reachable)
        if kk > 0:
            short = np.flatnonzero(idx[:, kk - 1] == -1)
            if short.size:  # prefix relaxation: k results, no brute force
                bi, bd = self._relaxed_query(
                    Q[short], kq, None if ex is None else ex[short])
                idx[short] = bi
                dst[short] = bd
        return idx, dst

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
        out: list[tuple[np.ndarray, np.ndarray]] = []

        if not self._use_lsh or exact:
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

        bs = self.query_block_size
        for s in range(0, m, bs):
            e = min(m, s + bs)
            Qb = Q[s:e]
            qids, cands = self._gather(Qb)
            if ex is not None and qids.size:
                keep = cands != ex[s:e][qids]
                qids, cands = qids[keep], cands[keep]
            dist = self._refine(Qb, qids, cands)
            keep = dist <= radius
            qids, cands, dist = qids[keep], cands[keep], dist[keep]
            order = np.argsort(qids * np.float64(self._d + 1) + dist)
            qids, cands, dist = qids[order], cands[order], dist[order]
            bounds = np.cumsum(np.bincount(qids, minlength=Qb.shape[0]))[:-1]
            out.extend(zip(np.split(cands, bounds), np.split(dist, bounds)))
        return out

    # ------------------------------------------------------------------ #
    # Tuning and table construction
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

    def _table_codes(self, X: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        frac = np.mod(X[:, self._S[t]] + self._U[t], 1.0) * self._B
        codes = frac.astype(np.int64)
        np.minimum(codes, self._B - 1, out=codes)  # guard float round-up
        return codes, np.clip(frac - codes, 0.0, 1.0)

    def _tier_keys(self, ids: np.ndarray) -> np.ndarray:
        keys = np.empty((self._L, ids.size), dtype=np.int64)
        for t in range(self._L):
            codes, _ = self._table_codes(self._arena[ids], t)
            keys[t] = (codes * self._pw).sum(axis=1)
        return keys

    def _build_static(self) -> None:
        ids = np.arange(self._n_static)
        keys = self._tier_keys(ids)
        order = np.argsort(keys, axis=1, kind="stable")
        self._skeys_s = np.take_along_axis(keys, order, axis=1)
        self._sids_s = order  # static ids start at 0

    def _build_candidates(self) -> None:
        ids = np.arange(self._n_static, self.n_points)
        self._cand_sids = ids
        self._keys_c = self._tier_keys(ids)  # unsorted, aligned to ids
        order = np.argsort(self._keys_c, axis=1, kind="stable")
        self._skeys_c = np.take_along_axis(self._keys_c, order, axis=1)
        self._sids_c = ids[order]

    def _refresh_candidates(self) -> None:
        """Selective candidate refresh: re-place only the points whose key
        actually changed — i.e. that moved far enough to leave a cell — via
        delete + merge, never a full re-sort."""
        ids = self._cand_sids
        new_keys = self._tier_keys(ids)
        off = self._n_static
        for t in range(self._L):
            moved = new_keys[t] != self._keys_c[t]
            if not moved.any():
                continue
            keep = ~moved[self._sids_c[t] - off]
            sk, si = self._skeys_c[t][keep], self._sids_c[t][keep]
            order = np.argsort(new_keys[t][moved], kind="stable")
            nk = new_keys[t][moved][order]
            ni = ids[moved][order]
            p = np.searchsorted(sk, nk)
            self._skeys_c[t] = np.insert(sk, p, nk)
            self._sids_c[t] = np.insert(si, p, ni)
        self._keys_c = new_keys

    # ------------------------------------------------------------------ #
    # Query internals
    # ------------------------------------------------------------------ #

    def _lsh_block(self, Qb, k, exb, idx_view, dst_view) -> None:
        qids, cands = self._gather(Qb)
        self._finish_topk(Qb, k, exb, qids, cands, idx_view, dst_view)

    def _finish_topk(self, Qb, k, exb, qids, cands, idx_view, dst_view) -> None:
        if exb is not None and qids.size:
            keep = cands != exb[qids]
            qids, cands = qids[keep], cands[keep]
        if qids.size == 0:
            return
        dist = self._refine(Qb, qids, cands)
        # Top-k per query: qids is ascending, so one composite float key
        # replaces a two-pass lexsort (distances < d keep segments disjoint).
        order = np.argsort(qids * np.float64(self._d + 1) + dist)
        sq, sc, sd = qids[order], cands[order], dist[order]
        counts = np.bincount(sq, minlength=Qb.shape[0])
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        rank = np.arange(sq.size) - starts[sq]
        keep = rank < k
        idx_view[sq[keep], rank[keep]] = sc[keep]
        dst_view[sq[keep], rank[keep]] = sd[keep]

    def _relaxed_query(self, Q, k, ex) -> tuple[np.ndarray, np.ndarray]:
        """k-NN for queries whose probed buckets under-filled, without brute
        force: keys are digit-concatenations ``sum(code_j * B**j)``, so the
        sorted key arrays are in lexicographic digit order and dropping the
        low ``lev`` digits turns a bucket into a contiguous key range
        ``[key//B**lev * B**lev, +B**lev)``. Each query widens level by level
        until enough raw candidates exist; level K spans the whole table, so
        k results are structurally guaranteed — two searchsorted calls per
        table per level, never a distance scan of the dataset.
        """
        m, n = Q.shape[0], self.n_points
        tiers = [(self._skeys_s, self._sids_s)]
        if self.n_candidates:
            tiers.append((self._skeys_c, self._sids_c))
        base = np.empty((self._L, m), dtype=np.int64)
        for t in range(self._L):
            codes, _ = self._table_codes(Q, t)
            base[t] = (codes * self._pw).sum(axis=1)

        need = 4 * k
        level = np.full(m, self._K, dtype=np.int64)
        active = np.ones(m, dtype=bool)
        for lev in range(1, self._K):
            if not active.any():
                break
            width = self._B ** lev
            aidx = np.flatnonzero(active)
            counts = np.zeros(aidx.size, dtype=np.int64)
            for t in range(self._L):
                lo_key = base[t, aidx] // width * width
                for skeys, _ in tiers:
                    lo = np.searchsorted(skeys[t], lo_key, side="left")
                    hi = np.searchsorted(skeys[t], lo_key + width, side="left")
                    counts += hi - lo
            done = counts >= need
            level[aidx[done]] = lev
            active[aidx[done]] = False

        chunks = []
        for lev in np.unique(level):
            gidx = np.flatnonzero(level == lev)
            width = self._B ** lev
            # Level K covers every point: one pass per tier is exhaustive.
            for t in range(self._L) if lev < self._K else (0,):
                lo_key = base[t, gidx] // width * width
                for skeys, sids in tiers:
                    lo = np.searchsorted(skeys[t], lo_key, side="left")
                    hi = np.searchsorted(skeys[t], lo_key + width, side="left")
                    cnt = hi - lo
                    total = int(cnt.sum())
                    if total == 0:
                        continue
                    seg = np.cumsum(cnt) - cnt
                    pos = np.arange(total) + np.repeat(lo - seg, cnt)
                    chunks.append(np.repeat(gidx, cnt) * n + sids[t, pos])

        idx = np.full((m, k), -1, dtype=np.int64)
        dst = np.full((m, k), np.inf)
        if chunks:
            pairs = np.concatenate(chunks)
            pairs.sort()
            keep = np.empty(pairs.size, dtype=bool)
            keep[0] = True
            np.not_equal(pairs[1:], pairs[:-1], out=keep[1:])
            pairs = pairs[keep]
            self._finish_topk(Q, k, ex, pairs // n, pairs % n, idx, dst)
        return idx, dst

    def _gather(self, Qb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Deduplicated (query_row, point_id) pairs across tables and tiers."""
        n = self.n_points
        tiers = [(self._skeys_s, self._sids_s)]
        if self.n_candidates:
            tiers.append((self._skeys_c, self._sids_c))
        chunks = []
        for t in range(self._L):
            codes, f = self._table_codes(Qb, t)
            keys = (codes * self._pw).sum(axis=1)
            nprobe = min(self.probes, self._K)
            if nprobe > 0:
                dirs = np.where(f >= 0.5, 1, -1)
                alt = (codes + dirs) % self._B
                key_delta = (alt - codes) * self._pw
                closeness = np.minimum(f, 1.0 - f)
                top = np.argpartition(closeness, nprobe - 1, axis=1)[:, :nprobe]
                probe_keys = keys[:, None] + np.take_along_axis(key_delta, top, axis=1)
                all_keys = np.concatenate([keys[:, None], probe_keys], axis=1)
            else:
                all_keys = keys[:, None]
            flat = all_keys.ravel()
            key_qrow = np.arange(flat.size) // all_keys.shape[1]
            for skeys, sids in tiers:
                lo = np.searchsorted(skeys[t], flat, side="left")
                hi = np.searchsorted(skeys[t], flat, side="right")
                counts = hi - lo
                total = int(counts.sum())
                if total == 0:
                    continue
                seg = np.cumsum(counts) - counts
                pos = np.arange(total) + np.repeat(lo - seg, counts)
                chunks.append(np.repeat(key_qrow, counts) * n + sids[t, pos])
        if not chunks:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        pairs = np.concatenate(chunks)
        pairs.sort()  # sort + neighbour-diff dedupe, grouped by query row
        keep = np.empty(pairs.size, dtype=bool)
        keep[0] = True
        np.not_equal(pairs[1:], pairs[:-1], out=keep[1:])
        pairs = pairs[keep]
        return pairs // n, pairs % n

    def _refine(self, Qb, qids, cands) -> np.ndarray:
        q32 = Qb.astype(np.float32)
        delta = np.abs(q32[qids] - self._pts32[cands])
        np.minimum(delta, np.float32(1.0) - delta, out=delta)
        return delta.sum(axis=1)

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
