//! torann's native core: the toroidal L1 LSH index in Rust.
//!
//! Implements `torann/CONTRACT.md` exactly: given the same parameters
//! the tables are byte-identical to the pure-NumPy reference. The tie rules
//! that guarantee it:
//!   - tier builds sort by (key, id)            [stable argsort of asc. ids]
//!   - selective update merges new-before-kept  [np.searchsorted side=left]
//!   - promote merges candidate-before-static   [np.searchsorted side=left]
//! and keys are computed in f64 exactly as the reference:
//!   code = min((frac(x + u) * B) as i64, B - 1),
//! with frac evaluated as `s - s.floor()` — bit-identical to np.mod for the
//! non-negative inputs the facade guarantees (see `wrap_unit`).
//!
//! Internal acceleration (non-normative — tables and results are unchanged):
//! a per-table direct-address offset array over the key space `[0, B^K)`
//! answers every bucket lookup and relaxation range in O(1) (`offs_s`,
//! falling back to a batched branchless binary search when the key space
//! outgrows `OFFS_BUDGET`), key computation streams through per-chunk
//! buffers with no per-point allocation, and refinement runs on eight
//! parallel f32 accumulators (one AVX lane set). Rejected-by-measurement
//! variants are recorded in ANALYSIS.md so they are not re-tried.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

/// Lower bound: first index whose value is >= v.
#[inline]
fn lower_bound(a: &[i64], v: i64) -> usize {
    a.partition_point(|&x| x < v)
}

/// `s mod 1` for non-negative s: bit-identical to fmod/np.mod on the whole
/// domain (fmod(s, 1) = s - floor(s) as reals, and the subtraction is
/// exact — floor(s) cancels exactly the leading significand bits, so the
/// fractional part is always representable). Unlike fmod this is two
/// branchless instructions (floor lowers to vroundsd), inlineable and
/// vectorizable. Negativity is the one precondition, and the facade's
/// mod-1 reduction (CONTRACT.md) guarantees it with room to spare.
#[inline(always)]
fn wrap_unit(s: f64) -> f64 {
    debug_assert!(s >= 0.0, "point + offset must be non-negative");
    s - s.floor()
}

/// Toroidal L1 in f32. Eight parallel accumulators break the serial
/// float-add dependency so LLVM vectorizes the loop (one AVX lane set);
/// f32 addition is not associative, so the plain `s +=` form cannot be
/// auto-vectorized and runs scalar. Summation order differs from NumPy's
/// pairwise sum either way — the contract allows 1e-5.
#[inline]
fn dist_l1_32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for j in 0..8 {
            let t = (ca[j] - cb[j]).abs();
            let w = 1.0f32 - t;
            acc[j] += if t < w { t } else { w };
        }
    }
    let mut s = 0.0f32;
    for (x, y) in a.chunks_exact(8).remainder().iter()
        .zip(b.chunks_exact(8).remainder()) {
        let t = (x - y).abs();
        let w = 1.0f32 - t;
        s += if t < w { t } else { w };
    }
    s + acc.iter().sum::<f32>()
}

/// Hint the cache that an address is about to be read (gather and refine
/// walk effectively random locations, so the hardware prefetcher cannot
/// help). No-op off x86_64. Safety: prefetch never faults, and every call
/// site passes an in-bounds reference anyway.
#[inline(always)]
fn prefetch<T>(r: &T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(
            r as *const T as *const i8,
            core::arch::x86_64::_MM_HINT_T0,
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = r;
    }
}

/// Bounded max-heap of (distance bits, id); f32 bit patterns of
/// non-negative floats order like the floats themselves.
struct TopK {
    heap: Vec<(u32, i64)>, // binary max-heap by bits
    k: usize,
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK { heap: Vec::with_capacity(k), k }
    }

    fn clear(&mut self) {
        self.heap.clear();
    }

    fn push(&mut self, dist: f32, id: i64) {
        let bits = dist.to_bits();
        if self.heap.len() < self.k {
            self.heap.push((bits, id));
            let mut i = self.heap.len() - 1;
            while i > 0 {
                let up = (i - 1) >> 1;
                if self.heap[up].0 >= self.heap[i].0 {
                    break;
                }
                self.heap.swap(up, i);
                i = up;
            }
        } else if bits < self.heap[0].0 {
            self.heap[0] = (bits, id);
            let mut i = 0;
            loop {
                let (l, r) = (2 * i + 1, 2 * i + 2);
                let mut big = i;
                if l < self.k && self.heap[l].0 > self.heap[big].0 {
                    big = l;
                }
                if r < self.k && self.heap[r].0 > self.heap[big].0 {
                    big = r;
                }
                if big == i {
                    break;
                }
                self.heap.swap(i, big);
                i = big;
            }
        }
    }

    /// Ascending (dist, id); pads the row to k with (-1, inf).
    fn emit(&mut self, out_idx: &mut [i64], out_dst: &mut [f64]) {
        self.heap.sort_unstable();
        for (i, &(bits, id)) in self.heap.iter().enumerate() {
            out_idx[i] = id;
            out_dst[i] = f32::from_bits(bits) as f64;
        }
        for i in self.heap.len()..self.k {
            out_idx[i] = -1;
            out_dst[i] = f64::INFINITY;
        }
    }
}

/// Per-worker query workspace (rayon `map_init`).
struct Ws {
    stamp: Vec<u32>,
    stamp_now: u32,
    cand: Vec<i64>,
    q32: Vec<f32>,
    codes: Vec<i64>,
    frac: Vec<f64>,
    clos: Vec<f64>,
    kbuf: Vec<(u32, i64)>, // (table, key) of every probe of one query
    base: Vec<usize>,      // batched lower_bound state, aligned to kbuf
}

impl Ws {
    fn new(n: usize, d: usize, k_dims: usize) -> Self {
        Ws {
            stamp: vec![0; n],
            stamp_now: 0,
            cand: Vec::with_capacity(4096),
            q32: vec![0.0; d],
            codes: vec![0; k_dims],
            frac: vec![0.0; k_dims],
            clos: vec![0.0; k_dims],
            kbuf: Vec::new(),
            base: Vec::new(),
        }
    }
}

#[pyclass]
struct RustLshIndex {
    b: i64,
    kdim: usize,
    l: usize,
    probes: usize,
    s: Vec<i64>,
    u: Vec<f64>,
    pw: Vec<i64>,
    d: usize,
    n_points: usize,
    n_static: usize,
    pts: Vec<f64>,
    pts32: Vec<f32>,
    skeys_s: Vec<i64>,
    sids_s: Vec<i64>,
    /// Direct-address bucket bounds for the static tier: per table,
    /// `B^K + 1` cumulative counts with `offs[t][v] = lower_bound(skeys_s[t], v)`
    /// — keys live in `[0, B^K)`, so one L2-resident load replaces every
    /// binary search. Empty when the key space is too large (see
    /// `OFFS_BUDGET`); all lookups then fall back to the batched search.
    offs_s: Vec<u32>,
    keys_c: Vec<i64>,
    skeys_c: Vec<i64>,
    sids_c: Vec<i64>,
    stats: QueryStats,
}

/// Cumulative per-phase query timing (CPU-ns summed across rayon workers)
/// — the built-in profiler behind `stats()` / `reset_stats()`.
#[derive(Default)]
struct QueryStats {
    gather_ns: std::sync::atomic::AtomicU64,
    refine_ns: std::sync::atomic::AtomicU64,
    relax_ns: std::sync::atomic::AtomicU64,
    queries: std::sync::atomic::AtomicU64,
    cands: std::sync::atomic::AtomicU64,
    relaxed: std::sync::atomic::AtomicU64,
}

/// Memory budget for the direct-address offset tables, in bytes. The tuner
/// keeps `B^K ~ n/64`, so real indexes use a few MB; the budget only trips
/// on hand-forced parameters, where gather falls back to binary search.
const OFFS_BUDGET: usize = 64 << 20;

impl RustLshIndex {
    #[inline]
    fn n_cand(&self) -> usize {
        self.n_points - self.n_static
    }

    /// Key-space size `B^K` when direct addressing is enabled, else 0.
    #[inline]
    fn key_space(&self) -> usize {
        let bk = self.pw[self.kdim - 1].saturating_mul(self.b);
        let entries = (bk as usize).saturating_add(1);
        if entries.saturating_mul(self.l * 4) <= OFFS_BUDGET {
            bk as usize
        } else {
            0
        }
    }

    /// (Re)build `offs_s` from the sorted static keys — O(n + B^K) per
    /// table. Called at build() and promote(); update() never moves the
    /// static tier.
    fn build_static_offsets(&mut self) {
        let bk = self.key_space();
        if bk == 0 {
            self.offs_s = Vec::new();
            return;
        }
        let ns = self.n_static;
        let mut offs = vec![0u32; self.l * (bk + 1)];
        offs.par_chunks_mut(bk + 1).enumerate().for_each(|(t, ot)| {
            let sk = &self.skeys_s[t * ns..(t + 1) * ns];
            let mut i = 0usize;
            for v in 0..=bk {
                while i < ns && sk[i] < v as i64 {
                    i += 1;
                }
                ot[v] = i as u32;
            }
        });
        self.offs_s = offs;
    }

    /// The normative hash of one point for one table.
    #[inline]
    fn point_key(&self, x: &[f64], t: usize) -> i64 {
        let st = &self.s[t * self.kdim..(t + 1) * self.kdim];
        let ut = &self.u[t * self.kdim..(t + 1) * self.kdim];
        let mut key = 0i64;
        for j in 0..self.kdim {
            // wrap_unit matches np.mod on [0, 2); `as` truncates.
            let frac = wrap_unit(x[st[j] as usize] + ut[j]) * self.b as f64;
            let mut code = frac as i64;
            if code > self.b - 1 {
                code = self.b - 1;
            }
            key += code * self.pw[j];
        }
        key
    }

    /// keys\[t * n + i\] for point rows [row0, row0 + n).
    fn compute_keys(&self, row0: usize, n: usize) -> Vec<i64> {
        let mut keys = vec![0i64; self.l * n];
        // Parallel over point chunks; each chunk fills a small t-major
        // buffer (point row stays in L1 across all tables), then lands in
        // the shared array with one contiguous copy per table. Chunk
        // writes are disjoint by construction. Chunk size shrinks with n
        // so small tiers (the per-epoch update) still fan out over all
        // workers.
        let chunk = (n.div_ceil(4 * rayon::current_num_threads()))
            .clamp(16, 1024);
        struct SendPtr(*mut i64);
        unsafe impl Sync for SendPtr {}
        let out = SendPtr(keys.as_mut_ptr());
        (0..n.div_ceil(chunk)).into_par_iter().for_each(|c| {
            let base = c * chunk;
            let len = chunk.min(n - base);
            let mut buf = vec![0i64; self.l * len];
            for i in 0..len {
                let row = row0 + base + i;
                let x = &self.pts[row * self.d..(row + 1) * self.d];
                for t in 0..self.l {
                    buf[t * len + i] = self.point_key(x, t);
                }
            }
            let p = &out;
            for t in 0..self.l {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf.as_ptr().add(t * len),
                        p.0.add(t * n + base),
                        len,
                    );
                }
            }
        });
        keys
    }

    /// Sort one tier per table by (key, id), ids starting at id0.
    fn sort_tier(&self, keys: &[i64], n: usize, id0: usize) -> (Vec<i64>, Vec<i64>) {
        let mut skeys = vec![0i64; self.l * n];
        let mut sids = vec![0i64; self.l * n];
        skeys
            .par_chunks_mut(n.max(1))
            .zip(sids.par_chunks_mut(n.max(1)))
            .enumerate()
            .for_each(|(t, (sk, si))| {
                if n == 0 {
                    return;
                }
                let mut tmp: Vec<(i64, i64)> = (0..n)
                    .map(|i| (keys[t * n + i], (id0 + i) as i64))
                    .collect();
                tmp.sort_unstable();
                for i in 0..n {
                    sk[i] = tmp[i].0;
                    si[i] = tmp[i].1;
                }
            });
        (skeys, sids)
    }

    fn build_candidate_tier(&mut self) {
        let nc = self.n_cand();
        self.keys_c = self.compute_keys(self.n_static, nc);
        let (sk, si) = self.sort_tier(&self.keys_c, nc, self.n_static);
        self.skeys_c = sk;
        self.sids_c = si;
    }

    /// Static keys/ids of table t as slices.
    #[inline]
    fn stat(&self, t: usize) -> (&[i64], &[i64]) {
        let ns = self.n_static;
        (
            &self.skeys_s[t * ns..(t + 1) * ns],
            &self.sids_s[t * ns..(t + 1) * ns],
        )
    }

    #[inline]
    fn cand_tier(&self, t: usize) -> (&[i64], &[i64]) {
        let nc = self.n_cand();
        (
            &self.skeys_c[t * nc..(t + 1) * nc],
            &self.sids_c[t * nc..(t + 1) * nc],
        )
    }

    /// Codes + in-cell fractions of one query for one table.
    fn query_codes(&self, x: &[f64], t: usize, ws: &mut Ws) -> i64 {
        let st = &self.s[t * self.kdim..(t + 1) * self.kdim];
        let ut = &self.u[t * self.kdim..(t + 1) * self.kdim];
        let mut key = 0i64;
        for j in 0..self.kdim {
            let f = wrap_unit(x[st[j] as usize] + ut[j]) * self.b as f64;
            let mut code = f as i64;
            if code > self.b - 1 {
                code = self.b - 1;
            }
            ws.codes[j] = code;
            ws.frac[j] = (f - code as f64).clamp(0.0, 1.0);
            key += code * self.pw[j];
        }
        key
    }

    /// Multi-probe gather across all tables into ws.cand (deduplicated).
    ///
    /// The L(1+probes) bucket lookups of one query are independent chains
    /// of random cache misses, so the static-tier binary searches run
    /// *batched*: one level-synchronous branchless lower_bound over all
    /// keys at once, keeping every miss chain in flight simultaneously
    /// (memory-level parallelism — the FAISS lesson applied to lookups).
    fn gather_query(&self, ws: &mut Ws, x: &[f64]) {
        ws.stamp_now += 1;
        ws.cand.clear();
        ws.kbuf.clear();
        let nprobe = self.probes.min(self.kdim);
        for t in 0..self.l {
            let key = self.query_codes(x, t, ws);
            ws.kbuf.push((t as u32, key));
            for j in 0..self.kdim {
                let f = ws.frac[j];
                ws.clos[j] = f.min(1.0 - f);
            }
            for _ in 0..nprobe {
                let mut best = 0usize;
                let mut best_c = 2.0f64;
                for j in 0..self.kdim {
                    if ws.clos[j] < best_c {
                        best_c = ws.clos[j];
                        best = j;
                    }
                }
                ws.clos[best] = 2.0; // consume
                let f = ws.frac[best];
                let dir: i64 = if f >= 0.5 { 1 } else { -1 };
                let alt = (ws.codes[best] + dir).rem_euclid(self.b);
                let pkey = key + (alt - ws.codes[best]) * self.pw[best];
                ws.kbuf.push((t as u32, pkey));
            }
        }

        // Static tier. With direct addressing, each probe is one pair of
        // L2-resident offset loads instead of a log(n) miss chain.
        let m = ws.kbuf.len();
        let ns = self.n_static;
        if ns > 0 && !self.offs_s.is_empty() {
            let row = self.offs_s.len() / self.l;
            for i in 0..m {
                let (t, key) = ws.kbuf[i];
                let ot = &self.offs_s[t as usize * row..(t as usize + 1) * row];
                let lo = ot[key as usize] as usize;
                let hi = ot[key as usize + 1] as usize;
                let si = &self.sids_s[t as usize * ns..(t as usize + 1) * ns];
                for &id in &si[lo..hi] {
                    if ws.stamp[id as usize] != ws.stamp_now {
                        ws.stamp[id as usize] = ws.stamp_now;
                        ws.cand.push(id);
                    }
                }
            }
        } else if ns > 0 {
            // Fallback (key space over budget): batched branchless
            // lower_bound — one level-synchronous pass keeps every miss
            // chain in flight. No prefetch here: 120 independent chains
            // already saturate the load buffers; hints measured slower.
            ws.base.clear();
            ws.base.resize(m, 0);
            let mut len = ns;
            while len > 1 {
                let half = len / 2;
                for i in 0..m {
                    let (t, key) = ws.kbuf[i];
                    let sk = &self.skeys_s[t as usize * ns..(t as usize + 1) * ns];
                    let b = ws.base[i];
                    ws.base[i] = b + usize::from(sk[b + half - 1] < key) * half;
                }
                len -= half;
            }
            for i in 0..m {
                let (t, key) = ws.kbuf[i];
                let sk = &self.skeys_s[t as usize * ns..(t as usize + 1) * ns];
                let si = &self.sids_s[t as usize * ns..(t as usize + 1) * ns];
                let mut lo = ws.base[i] + usize::from(sk[ws.base[i]] < key);
                while lo < ns && sk[lo] == key {
                    let id = si[lo] as usize;
                    if ws.stamp[id] != ws.stamp_now {
                        ws.stamp[id] = ws.stamp_now;
                        ws.cand.push(si[lo]);
                    }
                    lo += 1;
                }
            }
        }
        // Candidate tier: small (one batch), plain per-key search.
        if self.n_cand() > 0 {
            for i in 0..m {
                let (t, key) = ws.kbuf[i];
                let (sk, si) = self.cand_tier(t as usize);
                let mut lo = lower_bound(sk, key);
                while lo < sk.len() && sk[lo] == key {
                    let id = si[lo] as usize;
                    if ws.stamp[id] != ws.stamp_now {
                        ws.stamp[id] = ws.stamp_now;
                        ws.cand.push(si[lo]);
                    }
                    lo += 1;
                }
            }
        }
    }

    fn refine_topk(&self, ws: &Ws, top: &mut TopK, ex: i64) {
        const PF: usize = 8; // prefetch distance, rows ahead
        for (i, &id) in ws.cand.iter().enumerate() {
            if let Some(&nid) = ws.cand.get(i + PF) {
                prefetch(&self.pts32[nid as usize * self.d]);
            }
            if id == ex {
                continue;
            }
            let p = &self.pts32[id as usize * self.d..(id as usize + 1) * self.d];
            top.push(dist_l1_32(&ws.q32, p), id);
        }
    }

    /// [lo, hi) of `[lo_key, lo_key + width)` in the sorted static keys;
    /// `lo_key + width <= B^K`, so the offset table answers it directly.
    #[inline]
    fn static_range(&self, t: usize, lo_key: i64, width: i64) -> (usize, usize) {
        if !self.offs_s.is_empty() {
            let row = self.offs_s.len() / self.l;
            let ot = &self.offs_s[t * row..(t + 1) * row];
            (ot[lo_key as usize] as usize, ot[(lo_key + width) as usize] as usize)
        } else {
            let (sk, _) = self.stat(t);
            (lower_bound(sk, lo_key), lower_bound(sk, lo_key + width))
        }
    }

    fn range_count(&self, t: usize, lo_key: i64, width: i64) -> usize {
        let (lo, hi) = self.static_range(t, lo_key, width);
        let mut cnt = hi - lo;
        if self.n_cand() > 0 {
            let (sk, _) = self.cand_tier(t);
            cnt += lower_bound(sk, lo_key + width) - lower_bound(sk, lo_key);
        }
        cnt
    }

    fn gather_range(&self, ws: &mut Ws, t: usize, lo_key: i64, width: i64) {
        let (_, si) = self.stat(t);
        let (lo, hi) = self.static_range(t, lo_key, width);
        for &id in &si[lo..hi] {
            if ws.stamp[id as usize] != ws.stamp_now {
                ws.stamp[id as usize] = ws.stamp_now;
                ws.cand.push(id);
            }
        }
        if self.n_cand() > 0 {
            let (sk, si) = self.cand_tier(t);
            let (lo, hi) = (lower_bound(sk, lo_key), lower_bound(sk, lo_key + width));
            for &id in &si[lo..hi] {
                if ws.stamp[id as usize] != ws.stamp_now {
                    ws.stamp[id as usize] = ws.stamp_now;
                    ws.cand.push(id);
                }
            }
        }
    }

    /// Prefix relaxation for one under-filled query (see CONTRACT.md).
    fn relax_query(&self, ws: &mut Ws, top: &mut TopK, x: &[f64], k: usize, ex: i64) {
        let mut base = vec![0i64; self.l];
        for t in 0..self.l {
            base[t] = self.query_codes(x, t, ws);
        }
        let need = 4 * k;
        let mut level = self.kdim;
        for lev in 1..self.kdim {
            let width = self.pw[lev];
            let mut cnt = 0usize;
            for t in 0..self.l {
                cnt += self.range_count(t, base[t] / width * width, width);
            }
            if cnt >= need {
                level = lev;
                break;
            }
        }
        ws.stamp_now += 1;
        ws.cand.clear();
        if level < self.kdim {
            let width = self.pw[level];
            for t in 0..self.l {
                self.gather_range(ws, t, base[t] / width * width, width);
            }
        } else {
            // level K spans every point: one exhaustive pass over table 0
            let width = self.pw[self.kdim - 1] * self.b;
            self.gather_range(ws, 0, base[0] / width * width, width);
        }
        top.clear();
        self.refine_topk(ws, top, ex);
    }
}

#[pymethods]
impl RustLshIndex {
    #[new]
    #[pyo3(signature = (b, k, l, probes, s, u, block=1024))]
    fn new(
        b: i64,
        k: usize,
        l: usize,
        probes: usize,
        s: PyReadonlyArray2<i64>,
        u: PyReadonlyArray2<f64>,
        block: usize,
    ) -> PyResult<Self> {
        let _ = block; // advisory; rayon schedules per query
        if s.shape() != [l, k] || u.shape() != [l, k] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "S and U must have shape (L, K)",
            ));
        }
        let mut pw = vec![1i64; k];
        for j in 1..k {
            pw[j] = pw[j - 1] * b;
        }
        Ok(RustLshIndex {
            b,
            kdim: k,
            l,
            probes,
            s: s.as_slice()?.to_vec(),
            u: u.as_slice()?.to_vec(),
            pw,
            d: 0,
            n_points: 0,
            n_static: 0,
            pts: Vec::new(),
            pts32: Vec::new(),
            skeys_s: Vec::new(),
            sids_s: Vec::new(),
            offs_s: Vec::new(),
            keys_c: Vec::new(),
            skeys_c: Vec::new(),
            sids_c: Vec::new(),
            stats: QueryStats::default(),
        })
    }

    fn build(&mut self, py: Python<'_>, points: PyReadonlyArray2<f64>, n_static: usize) -> PyResult<()> {
        let shape = points.shape();
        let (n, d) = (shape[0], shape[1]);
        let pts = points.as_slice()?.to_vec();
        py.detach(|| {
            self.d = d;
            self.n_points = n;
            self.n_static = n_static;
            self.pts32 = pts.iter().map(|&v| v as f32).collect();
            self.pts = pts;
            let keys = self.compute_keys(0, n_static);
            let (sk, si) = self.sort_tier(&keys, n_static, 0);
            self.skeys_s = sk;
            self.sids_s = si;
            self.build_static_offsets();
            self.build_candidate_tier();
        });
        Ok(())
    }

    /// Selective refresh: re-place only entries whose key changed, by
    /// delete + merge (new-before-kept on equal keys).
    fn update(&mut self, py: Python<'_>, coords: PyReadonlyArray2<f64>) -> PyResult<()> {
        let c = coords.as_slice()?.to_vec();
        py.detach(|| {
            let (nc, d, off) = (self.n_cand(), self.d, self.n_static);
            self.pts[off * d..].copy_from_slice(&c);
            for (dst, &v) in self.pts32[off * d..].iter_mut().zip(c.iter()) {
                *dst = v as f32;
            }
            let new_keys = self.compute_keys(off, nc);

            let old_keys = std::mem::take(&mut self.keys_c);
            let mut skeys_c = std::mem::take(&mut self.skeys_c);
            let mut sids_c = std::mem::take(&mut self.sids_c);
            skeys_c
                .par_chunks_mut(nc.max(1))
                .zip(sids_c.par_chunks_mut(nc.max(1)))
                .enumerate()
                .for_each(|(t, (sk, si))| {
                    let nk = &new_keys[t * nc..(t + 1) * nc];
                    let ok = &old_keys[t * nc..(t + 1) * nc];
                    let mut moved: Vec<(i64, i64)> = (0..nc)
                        .filter(|&i| nk[i] != ok[i])
                        .map(|i| (nk[i], (off + i) as i64))
                        .collect();
                    if moved.is_empty() {
                        return;
                    }
                    moved.sort_unstable();
                    let kept: Vec<(i64, i64)> = (0..nc)
                        .filter(|&i| {
                            let row = (si[i] as usize) - off;
                            nk[row] == ok[row]
                        })
                        .map(|i| (sk[i], si[i]))
                        .collect();
                    let (mut a, mut b) = (0usize, 0usize);
                    for i in 0..nc {
                        if a < moved.len() && (b >= kept.len() || moved[a].0 <= kept[b].0) {
                            sk[i] = moved[a].0;
                            si[i] = moved[a].1;
                            a += 1;
                        } else {
                            sk[i] = kept[b].0;
                            si[i] = kept[b].1;
                            b += 1;
                        }
                    }
                });
            self.skeys_c = skeys_c;
            self.sids_c = sids_c;
            self.keys_c = new_keys;
        });
        Ok(())
    }

    /// Merge candidates into the static tier (candidate-before-static on
    /// equal keys), then install the new batch.
    fn promote(&mut self, py: Python<'_>, new_candidates: PyReadonlyArray2<f64>) -> PyResult<()> {
        let c = new_candidates.as_slice()?.to_vec();
        let m = new_candidates.shape()[0];
        py.detach(|| {
            let nc = self.n_cand();
            if nc > 0 {
                let n = self.n_points;
                let ns = self.n_static;
                let mut skeys = vec![0i64; self.l * n];
                let mut sids = vec![0i64; self.l * n];
                skeys
                    .par_chunks_mut(n)
                    .zip(sids.par_chunks_mut(n))
                    .enumerate()
                    .for_each(|(t, (dk, di))| {
                        let sk = &self.skeys_s[t * ns..(t + 1) * ns];
                        let si = &self.sids_s[t * ns..(t + 1) * ns];
                        let ck = &self.skeys_c[t * nc..(t + 1) * nc];
                        let ci = &self.sids_c[t * nc..(t + 1) * nc];
                        let (mut a, mut b) = (0usize, 0usize);
                        for i in 0..n {
                            if a < nc && (b >= ns || ck[a] <= sk[b]) {
                                dk[i] = ck[a];
                                di[i] = ci[a];
                                a += 1;
                            } else {
                                dk[i] = sk[b];
                                di[i] = si[b];
                                b += 1;
                            }
                        }
                    });
                self.skeys_s = skeys;
                self.sids_s = sids;
            }
            self.n_static = self.n_points;
            self.build_static_offsets();
            if m > 0 {
                self.pts.extend_from_slice(&c);
                self.pts32.extend(c.iter().map(|&v| v as f32));
                self.n_points += m;
            }
            self.build_candidate_tier();
        });
        Ok(())
    }

    #[pyo3(signature = (queries, k, exclude_ids=None))]
    fn query_knn<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f64>,
        k: usize,
        exclude_ids: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        let shape = queries.shape();
        let (m, d) = (shape[0], shape[1]);
        let q = queries.as_slice()?.to_vec();
        let ex: Option<Vec<i64>> = match &exclude_ids {
            Some(e) => Some(e.as_slice()?.to_vec()),
            None => None,
        };
        let mut out_idx = vec![-1i64; m * k];
        let mut out_dst = vec![f64::INFINITY; m * k];

        py.detach(|| {
            let reachable = self.n_points - usize::from(ex.is_some());
            let kk = k.min(reachable);
            out_idx
                .par_chunks_mut(k)
                .zip(out_dst.par_chunks_mut(k))
                .enumerate()
                .for_each_init(
                    || (Ws::new(self.n_points, d, self.kdim), TopK::new(k)),
                    |(ws, top), (qi, (row_idx, row_dst))| {
                        use std::sync::atomic::Ordering::Relaxed;
                        let x = &q[qi * d..(qi + 1) * d];
                        for j in 0..d {
                            ws.q32[j] = x[j] as f32;
                        }
                        let exq = ex.as_ref().map_or(-1, |e| e[qi]);
                        let t0 = std::time::Instant::now();
                        self.gather_query(ws, x);
                        let t1 = std::time::Instant::now();
                        top.clear();
                        self.refine_topk(ws, top, exq);
                        let t2 = std::time::Instant::now();
                        let st = &self.stats;
                        st.cands.fetch_add(ws.cand.len() as u64, Relaxed);
                        if top.heap.len() < kk {
                            self.relax_query(ws, top, x, k, exq);
                            st.relax_ns.fetch_add(
                                t2.elapsed().as_nanos() as u64, Relaxed);
                            st.relaxed.fetch_add(1, Relaxed);
                        }
                        top.emit(row_idx, row_dst);
                        st.gather_ns
                            .fetch_add((t1 - t0).as_nanos() as u64, Relaxed);
                        st.refine_ns
                            .fetch_add((t2 - t1).as_nanos() as u64, Relaxed);
                        st.queries.fetch_add(1, Relaxed);
                    },
                );
        });
        Ok((
            PyArray1::from_vec(py, out_idx).reshape([m, k])?,
            PyArray1::from_vec(py, out_dst).reshape([m, k])?,
        ))
    }

    #[pyo3(signature = (queries, radius, exclude_ids=None))]
    fn query_radius<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f64>,
        radius: f64,
        exclude_ids: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let shape = queries.shape();
        let (m, d) = (shape[0], shape[1]);
        let q = queries.as_slice()?.to_vec();
        let ex: Option<Vec<i64>> = match &exclude_ids {
            Some(e) => Some(e.as_slice()?.to_vec()),
            None => None,
        };
        let r32 = radius as f32;

        let rows: Vec<(Vec<i64>, Vec<f64>)> = py.detach(|| {
            (0..m)
                .into_par_iter()
                .map_init(
                    || Ws::new(self.n_points, d, self.kdim),
                    |ws, qi| {
                        let x = &q[qi * d..(qi + 1) * d];
                        for j in 0..d {
                            ws.q32[j] = x[j] as f32;
                        }
                        let exq = ex.as_ref().map_or(-1, |e| e[qi]);
                        self.gather_query(ws, x);
                        let mut hits: Vec<(u32, i64)> = Vec::new();
                        for &id in &ws.cand {
                            if id == exq {
                                continue;
                            }
                            let p = &self.pts32
                                [id as usize * self.d..(id as usize + 1) * self.d];
                            let dd = dist_l1_32(&ws.q32, p);
                            if dd <= r32 {
                                hits.push((dd.to_bits(), id));
                            }
                        }
                        hits.sort_unstable();
                        let ids: Vec<i64> = hits.iter().map(|h| h.1).collect();
                        let ds: Vec<f64> =
                            hits.iter().map(|h| f32::from_bits(h.0) as f64).collect();
                        (ids, ds)
                    },
                )
                .collect()
        });

        let mut indptr = vec![0i64; m + 1];
        for i in 0..m {
            indptr[i + 1] = indptr[i] + rows[i].0.len() as i64;
        }
        let total = indptr[m] as usize;
        let mut ids = Vec::with_capacity(total);
        let mut ds = Vec::with_capacity(total);
        for (ri, rd) in rows {
            ids.extend(ri);
            ds.extend(rd);
        }
        Ok((
            PyArray1::from_vec(py, indptr),
            PyArray1::from_vec(py, ids),
            PyArray1::from_vec(py, ds),
        ))
    }

    fn tables<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let ns = self.n_static;
        let nc = self.n_cand();
        let d = PyDict::new(py);
        d.set_item(
            "static_keys",
            self.skeys_s.clone().into_pyarray(py).reshape([self.l, ns])?,
        )?;
        d.set_item(
            "static_ids",
            self.sids_s.clone().into_pyarray(py).reshape([self.l, ns])?,
        )?;
        d.set_item(
            "cand_keys",
            self.keys_c.clone().into_pyarray(py).reshape([self.l, nc])?,
        )?;
        d.set_item(
            "cand_sorted_keys",
            self.skeys_c.clone().into_pyarray(py).reshape([self.l, nc])?,
        )?;
        d.set_item(
            "cand_sorted_ids",
            self.sids_c.clone().into_pyarray(py).reshape([self.l, nc])?,
        )?;
        Ok(d)
    }

    /// Cumulative query-phase profile: CPU-ns summed across workers, plus
    /// gathered-candidate and relaxation counters. Reset with reset_stats().
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        use std::sync::atomic::Ordering::Relaxed;
        let d = PyDict::new(py);
        d.set_item("gather_ns", self.stats.gather_ns.load(Relaxed))?;
        d.set_item("refine_ns", self.stats.refine_ns.load(Relaxed))?;
        d.set_item("relax_ns", self.stats.relax_ns.load(Relaxed))?;
        d.set_item("queries", self.stats.queries.load(Relaxed))?;
        d.set_item("cands", self.stats.cands.load(Relaxed))?;
        d.set_item("relaxed", self.stats.relaxed.load(Relaxed))?;
        Ok(d)
    }

    fn reset_stats(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.stats.gather_ns.store(0, Relaxed);
        self.stats.refine_ns.store(0, Relaxed);
        self.stats.relax_ns.store(0, Relaxed);
        self.stats.queries.store(0, Relaxed);
        self.stats.cands.store(0, Relaxed);
        self.stats.relaxed.store(0, Relaxed);
    }

    #[getter]
    fn n_points(&self) -> usize {
        self.n_points
    }

    #[getter]
    fn n_static(&self) -> usize {
        self.n_static
    }

    #[getter]
    fn n_candidates(&self) -> usize {
        self.n_cand()
    }
}

#[pymodule]
#[pyo3(name = "_native")]
fn torann_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustLshIndex>()?;
    Ok(())
}
