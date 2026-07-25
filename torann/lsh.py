"""Pure NumPy reference implementation of the LSH index contract.

This implementation defines the normative behaviour (see CONTRACT.md): the
native implementation must produce byte-identical hash tables and
equivalent query results for the same parameters. Keep it readable — it is
the specification the Rust port is validated against.
"""

from __future__ import annotations

import numpy as np

from .base import BaseIndex

__all__ = ["PythonLshIndex"]


class PythonLshIndex(BaseIndex):
    """Two-tier toroidal-L1 LSH index (pure NumPy).

    Args:
        B: Cells per dimension (integer resolution >= 2).
        K: Sampled dimensions per table; ``B**K`` must fit in int64.
        L: Number of tables.
        probes: Neighbour-cell probes per table per query.
        S: int64 ``(L, K)`` sampled dimension indices (drawn by the facade).
        U: float64 ``(L, K)`` per-dimension offsets in ``[0, 1)``.
        block: Advisory query batch size for the vectorised blocks.
    """

    def __init__(self, B, K, L, probes, S, U, block=1024):
        self.B = int(B)
        self.K = int(K)
        self.L = int(L)
        self.probes = int(probes)
        self.S = np.ascontiguousarray(S, dtype=np.int64)
        self.U = np.ascontiguousarray(U, dtype=np.float64)
        self.block = int(block)
        self._pw = self.B ** np.arange(self.K, dtype=np.int64)
        self.n_points = 0
        self.n_static = 0
        self._pts = np.empty((0, 0))

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def build(self, points: np.ndarray, n_static: int) -> None:
        """Hash and key-sort both tiers from zero (stable sort: ties keep
        ascending id order — the normative table layout)."""
        self._pts = np.ascontiguousarray(points, dtype=np.float64).copy()
        self._pts32 = self._pts.astype(np.float32)
        self.n_points = self._pts.shape[0]
        self.n_static = int(n_static)
        self._build_static()
        self._build_candidates()

    def update(self, coords: np.ndarray) -> None:
        """Move the candidate tier. Selective: only entries whose key
        changed — i.e. that moved far enough to leave a cell — are
        re-placed, by delete + merge. Exact after every call."""
        self._pts[self.n_static:] = coords
        self._pts32[self.n_static:] = coords.astype(np.float32)
        ids = self._cand_sids
        new_keys = self._tier_keys(ids)
        off = self.n_static
        for t in range(self.L):
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

    def promote(self, new_candidates: np.ndarray) -> None:
        """Merge the candidate tier into the static tier (linear merge,
        never a re-sort) and install the new batch."""
        if self.n_candidates:
            skeys = np.empty((self.L, self.n_points), dtype=np.int64)
            sids = np.empty((self.L, self.n_points), dtype=np.int64)
            for t in range(self.L):
                p = np.searchsorted(self._skeys_s[t], self._skeys_c[t])
                skeys[t] = np.insert(self._skeys_s[t], p, self._skeys_c[t])
                sids[t] = np.insert(self._sids_s[t], p, self._sids_c[t])
            self._skeys_s, self._sids_s = skeys, sids
        self.n_static = self.n_points
        if new_candidates.size:
            C = np.ascontiguousarray(new_candidates, dtype=np.float64)
            self._pts = np.vstack([self._pts, C])
            self._pts32 = np.vstack([self._pts32, C.astype(np.float32)])
            self.n_points = self._pts.shape[0]
        self._build_candidates()

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def query_knn(self, queries, k, exclude_ids=None):
        """Batch k-NN: multi-probe gather → dedupe → float32 refine →
        per-query top-k; under-filled rows are completed by prefix
        relaxation (see :meth:`_relaxed_query`), never by a brute scan.

        Returns:
            ``(idx, dist)`` of shape ``(m, k)``, rows sorted by distance,
            padded with ``-1`` / ``inf`` only when fewer than ``k``
            reachable points exist.
        """
        m = queries.shape[0]
        idx = np.full((m, k), -1, dtype=np.int64)
        dst = np.full((m, k), np.inf)
        ex = exclude_ids
        for s in range(0, m, self.block):
            e = min(m, s + self.block)
            self._lsh_block(queries[s:e], k, None if ex is None else ex[s:e],
                            idx[s:e], dst[s:e])

        reachable = self.n_points - (0 if ex is None else 1)
        kk = min(k, reachable)
        if kk > 0:
            short = np.flatnonzero(idx[:, kk - 1] == -1)
            if short.size:  # prefix relaxation: k results, no brute force
                bi, bd = self._relaxed_query(
                    queries[short], k, None if ex is None else ex[short])
                idx[short] = bi
                dst[short] = bd
        return idx, dst

    def query_radius(self, queries, radius, exclude_ids=None):
        """Batch range query as a post-filter on the gathered candidate
        set (no relaxation — recall falls off beyond the probe reach).

        Returns:
            CSR triple ``(indptr, ids, dists)``; row ``i`` is
            ``ids[indptr[i]:indptr[i+1]]``, sorted by distance.
        """
        m = queries.shape[0]
        counts = []
        all_ids = []
        all_dst = []
        for s in range(0, m, self.block):
            e = min(m, s + self.block)
            Qb = queries[s:e]
            qids, cands = self._gather(Qb)
            if exclude_ids is not None and qids.size:
                keep = cands != exclude_ids[s:e][qids]
                qids, cands = qids[keep], cands[keep]
            dist = self._refine(Qb, qids, cands)
            keep = dist <= radius
            qids, cands, dist = qids[keep], cands[keep], dist[keep]
            d = self._pts.shape[1]
            order = np.argsort(qids * np.float64(d + 1) + dist)
            counts.append(np.bincount(qids, minlength=Qb.shape[0]))
            all_ids.append(cands[order])
            all_dst.append(dist[order].astype(np.float64))
        indptr = np.zeros(m + 1, dtype=np.int64)
        np.cumsum(np.concatenate(counts), out=indptr[1:])
        return indptr, np.concatenate(all_ids), np.concatenate(all_dst)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def tables(self) -> dict:
        """The hash tables, for conformance tests and benchmarks.

        Returns:
            Dict of copies: ``static_keys`` / ``static_ids`` (int64
            ``(L, n_static)``, key-sorted), ``cand_keys`` (unsorted,
            aligned to candidate row order) and ``cand_sorted_keys`` /
            ``cand_sorted_ids``.
        """
        return {
            "static_keys": self._skeys_s.copy(),
            "static_ids": self._sids_s.copy(),
            "cand_keys": self._keys_c.copy(),
            "cand_sorted_keys": self._skeys_c.copy(),
            "cand_sorted_ids": self._sids_c.copy(),
        }

    # ------------------------------------------------------------------ #
    # Table construction
    # ------------------------------------------------------------------ #

    def _table_codes(self, X, t):
        """The normative hash: per sampled dimension of table ``t``,
        ``code = min(int64(((x+u) mod 1) * B), B-1)``; also returns the
        in-cell fractional part (drives multi-probe direction)."""
        frac = np.mod(X[:, self.S[t]] + self.U[t], 1.0) * self.B
        codes = frac.astype(np.int64)
        np.minimum(codes, self.B - 1, out=codes)  # guard float round-up
        return codes, np.clip(frac - codes, 0.0, 1.0)

    def _tier_keys(self, ids):
        keys = np.empty((self.L, ids.size), dtype=np.int64)
        for t in range(self.L):
            codes, _ = self._table_codes(self._pts[ids], t)
            keys[t] = (codes * self._pw).sum(axis=1)
        return keys

    def _build_static(self) -> None:
        keys = self._tier_keys(np.arange(self.n_static))
        order = np.argsort(keys, axis=1, kind="stable")
        self._skeys_s = np.take_along_axis(keys, order, axis=1)
        self._sids_s = order  # static ids start at 0

    def _build_candidates(self) -> None:
        ids = np.arange(self.n_static, self.n_points)
        self._cand_sids = ids
        self._keys_c = self._tier_keys(ids)  # unsorted, aligned to ids
        order = np.argsort(self._keys_c, axis=1, kind="stable")
        self._skeys_c = np.take_along_axis(self._keys_c, order, axis=1)
        self._sids_c = ids[order]

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
        d = self._pts.shape[1]
        order = np.argsort(qids * np.float64(d + 1) + dist)
        sq, sc, sd = qids[order], cands[order], dist[order]
        counts = np.bincount(sq, minlength=Qb.shape[0])
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        rank = np.arange(sq.size) - starts[sq]
        keep = rank < k
        idx_view[sq[keep], rank[keep]] = sc[keep]
        dst_view[sq[keep], rank[keep]] = sd[keep]

    def _relaxed_query(self, Q, k, ex):
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
        base = np.empty((self.L, m), dtype=np.int64)
        for t in range(self.L):
            codes, _ = self._table_codes(Q, t)
            base[t] = (codes * self._pw).sum(axis=1)

        need = 4 * k
        level = np.full(m, self.K, dtype=np.int64)
        active = np.ones(m, dtype=bool)
        for lev in range(1, self.K):
            if not active.any():
                break
            width = self.B ** lev
            aidx = np.flatnonzero(active)
            counts = np.zeros(aidx.size, dtype=np.int64)
            for t in range(self.L):
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
            width = self.B ** lev
            # Level K covers every point: one pass per tier is exhaustive.
            for t in range(self.L) if lev < self.K else (0,):
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

    def _gather(self, Qb):
        """Deduplicated (query_row, point_id) pairs across tables and tiers."""
        n = self.n_points
        tiers = [(self._skeys_s, self._sids_s)]
        if self.n_candidates:
            tiers.append((self._skeys_c, self._sids_c))
        chunks = []
        for t in range(self.L):
            codes, f = self._table_codes(Qb, t)
            keys = (codes * self._pw).sum(axis=1)
            nprobe = min(self.probes, self.K)
            if nprobe > 0:
                dirs = np.where(f >= 0.5, 1, -1)
                alt = (codes + dirs) % self.B
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

    def _refine(self, Qb, qids, cands):
        """Toroidal L1 in float32 for every gathered (query, candidate)
        pair — the distance the contract mandates for refinement."""
        q32 = Qb.astype(np.float32)
        delta = np.abs(q32[qids] - self._pts32[cands])
        np.minimum(delta, np.float32(1.0) - delta, out=delta)
        return delta.sum(axis=1)
