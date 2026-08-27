//! torann's native core: the toroidal L1 LSH index in Rust.
//!
//! Implements `torann/lsh.py` exactly: given the same parameters the
//! tables are byte-identical to that pure-NumPy reference, which is the
//! specification. The tie rules that guarantee it:
//!
//! - tier builds sort by (key, id)            [stable argsort of asc. ids]
//! - selective update merges new-before-kept  [np.searchsorted side=left]
//! - promote merges candidate-before-static   [np.searchsorted side=left]
//!
//! Keys are computed in f64 exactly as the reference:
//! `code = min((frac(x + u) * B) as i64, B - 1)`, with frac evaluated as
//! `s - s.floor()` — bit-identical to np.mod for the non-negative inputs
//! the facade guarantees (see `wrap_unit`).
//!
//! Internal acceleration (non-normative — tables and results are unchanged):
//! per-table direct-address offset arrays over the key space `[0, B^K)`
//! answer every bucket lookup and relaxation range in O(1) for both tiers
//! (`offs_s`, `offs_c`, falling back to a batched branchless binary search
//! when the key space outgrows `OFFS_BUDGET`), key computation streams
//! through per-chunk buffers with no per-point allocation, and a batch of
//! queries is answered by a **bucket-centric join** (`query_block`) that
//! loads each bucket once per block instead of once per query. Refinement
//! scores four tile rows at a time (`ROWS`) through a plain `s +=` loop
//! that LLVM vectorizes, given the reassociation freedom of the Rust 1.98
//! algebraic float methods (see [`l1`]).
//!
//! Three things about this file are load-bearing and easy to undo by
//! accident. The kernel is bound by **dependency-chain latency, not
//! instruction count** — reducing instructions made it measurably worse,
//! and the fix was ILP (`ROWS = 4`), so a "simplification" that shortens
//! the code is likely to cost time. The index is **execution-bound, not
//! memory-bound**, so optimisations justified by fewer memory loads do not
//! convert: two attempts to raise bucket reuse (bucket-major work
//! partitioning, and reordering queries by hash key) each reduced traffic
//! as designed and moved the clock by nothing or less. A proposal here
//! needs a cycle argument, not a traffic argument. And the distance kernel
//! is **written scalar on purpose** — `algebraic_add` is what makes that
//! fast, and hand-vectorizing it back (as this file did with
//! `wide::f32x8` until 1.98) costs 8-50%, most of it at the dimensions
//! where a hand-picked eight lanes are the wrong shape.

// The kernels index with `for i in 0..n` on purpose: most loops write to
// several arrays at once and were profile-tuned in this exact shape. The
// introspection API also returns a few unavoidably wide tuple types.
#![allow(clippy::needless_range_loop, clippy::type_complexity)]
// Enforced, not asserted. The crate had exactly one `unsafe` block -- an
// `_mm_prefetch` hint, which cannot fault, alias or race and was `unsafe`
// only because `core::arch` intrinsics are gated on target features. It was
// still the one thing keeping "this code is safe" a claim a reader had to
// audit rather than one the compiler checks, so it was measured and dropped:
// ~10% on the LSH query path, and nothing at all on the brute kernels, which
// is where a population-sized caller spends every cycle. See `refine_topk`.
#![forbid(unsafe_code)]

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
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
/// mod-1 reduction guarantees it with room to spare.
#[inline(always)]
fn wrap_unit(s: f64) -> f64 {
    debug_assert!(s >= 0.0, "point + offset must be non-negative");
    s - s.floor()
}

/// Toroidal L1 in f32 of one pair, `D` dimensions (`D = 0`: use `d`).
///
/// `algebraic_add`/`algebraic_sub` (Rust 1.98) hand LLVM the reassociation
/// freedom that plain f32 `+` withholds, and that freedom is what lets this
/// be a plain `s +=` chain at all: the auto-vectorizer splits it into as
/// many independent accumulators as the target has registers, at the
/// target's *native* width, and folds them with a tree at the end. The
/// `wide::f32x8` kernel this replaced had to spell all of that out by hand
/// and was pinned to eight lanes on every target — which cost it the
/// dimensions below one full vector (`d < 8` never vectorized at all) and
/// the vector width above it.
///
/// `D` is the dimensionality when the caller knows it at compile time.
/// It is not a micro-optimisation: with a runtime trip count LLVM must
/// guard the vector loop and emit a scalar epilogue, and at one, two or
/// three vectors' worth of dimensions that scaffolding costs more than
/// the arithmetic it guards. Const `D` deletes both. `score_tile`'s
/// dispatch says which values earn it and why not more.
///
/// **What this gives up.** Reassociation means the summation order is the
/// compiler's to choose, so it is no longer part of the contract the way
/// the old kernel's eight fixed lanes were. These are compile-time
/// freedoms rather than runtime ones — a given machine instruction always
/// does the same thing — but the standard library is explicit that the
/// same source may be optimized differently at different call sites, so
/// bit-identity is not promised even within one binary, let alone across
/// builds or LLVM versions. Here that means the `D`-specialized
/// instantiations and the runtime-`d` one may fold the same `d` terms in
/// different orders.
///
/// The error that admits is a few ulp on a sum of `d` non-negative terms
/// each in `[0, 0.5]` — orders of magnitude inside the `atol=1e-5` the
/// cross-backend conformance test holds every backend to, and far too
/// small to move a distance a caller can observe. What it can do, rarely,
/// is reorder two candidates already within an ulp of each other, which is
/// why that test allows ids to differ on ties (it requires 99.5%
/// agreement).
///
/// That was measured rather than assumed, on fixed indexes so that no
/// converging loop could confound it: against the pre-1.98 kernel, recall
/// is *identical to five decimals* and the returned neighbour ids are
/// identical at d = 2, 8, 16, 24, 32 (k=5 and k=64) and 64, with distances
/// agreeing to ~1e-7 relative. The per-query and batched paths — which now
/// reach different instantiations of this function — still agree exactly
/// on all six shapes of the ESS suite.
///
/// The hash keys are untouched by any of this. They are computed in f64
/// with plain operators (see `point_key`), and *they* are the byte-identical
/// contract against `lsh.py` — reassociating them was measured too, and
/// bought nothing (OPTIMIZE.md), so it was not done.
#[inline(always)]
fn l1<const D: usize>(a: &[f32], b: &[f32], d: usize) -> f32 {
    let n = if D == 0 { d } else { D };
    let mut s = 0.0f32;
    for j in 0..n {
        let t = a[j].algebraic_sub(b[j]).abs();
        let w = 1.0f32.algebraic_sub(t);
        s = s.algebraic_add(if t < w { t } else { w });
    }
    s
}

/// Toroidal L1 in f32, dimensionality known only at run time.
#[inline]
fn dist_l1_32(a: &[f32], b: &[f32]) -> f32 {
    l1::<0>(a, b, a.len().min(b.len()))
}

/// Tile rows scored per call of [`dist_l1_rows`].
///
/// The kernel is bound by dependency-chain latency, not instruction count,
/// so what matters is having several independent accumulator chains in
/// flight; `ROWS` rows give `ROWS` of them, and the query chunk is loaded
/// once for all of them. Four was profile-tuned and eight measured worse.
const ROWS: usize = 4;

/// Toroidal L1 of one query against `ROWS` consecutive tile rows.
#[inline(always)]
fn dist_l1_rows<const D: usize>(q: &[f32], tile: &[f32], d: usize) -> [f32; ROWS] {
    let n = if D == 0 { d } else { D };
    let mut out = [0.0f32; ROWS];
    for r in 0..ROWS {
        out[r] = l1::<D>(q, &tile[r * n..(r + 1) * n], n);
    }
    out
}

/// The canonical k-NN order: distance first, id as the tie-break. f32 bit
/// patterns of non-negative floats order like the floats themselves, so the
/// pair compares as the tuple it is. Ranking by (dist, id) rather than by
/// arrival makes the k-th slot independent of the order candidates are
/// scored in — which is what lets the batched join (which visits a query's
/// candidates bucket-major rather than table-major) return the same rows.
#[inline(always)]
fn hless(a: (u32, i64), b: (u32, i64)) -> bool {
    a.0 < b.0 || (a.0 == b.0 && a.1 < b.1)
}

/// Sentinel worst-so-far for an under-full heap: everything beats it.
const WORST: (u32, i64) = (u32::MAX, i64::MAX);

/// Offer (bits, id) to the bounded max-heap in `h[..*len]` (capacity
/// `h.len()`), keeping the k smallest by [`hless`].
///
/// Ids already present are dropped: the batched join sees a point once per
/// table it shares with the query, and a heap must hold distinct ids. The
/// scan is O(k) but only runs on the few offers that actually beat the
/// k-th best, so it costs nothing per candidate.
#[inline]
fn heap_offer(h: &mut [(u32, i64)], len: &mut usize, bits: u32, id: i64) {
    let k = h.len();
    if k == 0 {
        return;
    }
    let e = (bits, id);
    if *len < k {
        if h[..*len].iter().any(|&(_, i)| i == id) {
            return;
        }
        h[*len] = e;
        let mut i = *len;
        *len += 1;
        while i > 0 {
            let up = (i - 1) >> 1;
            if !hless(h[up], h[i]) {
                break;
            }
            h.swap(up, i);
            i = up;
        }
    } else if hless(e, h[0]) {
        if h.iter().any(|&(_, i)| i == id) {
            return;
        }
        h[0] = e;
        let mut i = 0;
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut big = i;
            if l < k && hless(h[big], h[l]) {
                big = l;
            }
            if r < k && hless(h[big], h[r]) {
                big = r;
            }
            if big == i {
                break;
            }
            h.swap(i, big);
            i = big;
        }
    }
}

/// Ascending (dist, id) into one output row; pads to k with (-1, inf).
fn heap_emit(h: &mut [(u32, i64)], len: usize, out_idx: &mut [i64], out_dst: &mut [f64]) {
    h[..len].sort_unstable();
    for (i, &(bits, id)) in h[..len].iter().enumerate() {
        out_idx[i] = id;
        out_dst[i] = f32::from_bits(bits) as f64;
    }
    for i in len..h.len() {
        out_idx[i] = -1;
        out_dst[i] = f64::INFINITY;
    }
}

/// One query's bounded top-k (the per-query path and relaxation; the
/// batched join keeps its heaps in a block-wide slab instead).
struct TopK {
    buf: Vec<(u32, i64)>,
    len: usize,
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK {
            buf: vec![(0, 0); k],
            len: 0,
        }
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    fn push(&mut self, dist: f32, id: i64) {
        heap_offer(&mut self.buf, &mut self.len, dist.to_bits(), id);
    }

    fn emit(&mut self, out_idx: &mut [i64], out_dst: &mut [f64]) {
        heap_emit(&mut self.buf, self.len, out_idx, out_dst);
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

/// Per-worker workspace of the batched join: one query block's f32 tile,
/// top-k slab and grouping buffers. Grows to the block size on first use
/// and is then reused for every block that worker takes.
#[derive(Default)]
struct BWs {
    q32: Vec<f32>,          // nq * d, f32 copy of the block's queries
    exq: Vec<i64>,          // nq, excluded id per query (-1 = none)
    heaps: Vec<(u32, i64)>, // nq * k bounded max-heaps
    hlen: Vec<u32>,         // nq heap lengths
    codes: Vec<i64>,
    frac: Vec<f64>,
    clos: Vec<f64>,
    probe: Vec<(i64, u32)>, // (bucket key, query slot) of one table, key-sorted
    gids: Vec<i64>,         // ids of the bucket being scored
    tile: Vec<f32>,         // TILE * d gathered point coordinates
    tids: Vec<i64>,         // TILE ids, aligned to tile rows
}

impl BWs {
    /// Size the buffers for a block of `nq` queries in `d` dimensions with
    /// `k` neighbours, and reset the heaps.
    fn reset(&mut self, nq: usize, d: usize, k: usize, kdim: usize) {
        self.q32.resize(nq * d, 0.0);
        self.exq.resize(nq, -1);
        self.heaps.resize(nq * k, (0, 0));
        self.hlen.clear();
        self.hlen.resize(nq, 0);
        self.codes.resize(kdim, 0);
        self.frac.resize(kdim, 0.0);
        self.clos.resize(kdim, 0.0);
        self.tile.resize(TILE * d, 0.0);
        self.tids.resize(TILE, 0);
    }
}

/// Points per distance tile: one slice of a bucket, small enough that it
/// stays in L1 while every query of the group scans it (128 * 32 f32 = 16 KiB).
const TILE: usize = 128;

/// Fewer queries than this and the batched join has nothing to amortise
/// (its bucket loads are shared across the block), so the per-query path —
/// which pays for a dedup stamp but scores each candidate once — wins.
const BATCH_MIN: usize = 64;

/// Toroidal-L1 LSH index, compiled.
///
/// Behaviourally identical to `torann.lsh.PythonLshIndex` — given the same
/// parameters the hash tables are byte-identical — at 25-75x the speed. It is
/// selected automatically by `torann.ToroidalNN`; construct that instead of
/// this, which takes pre-drawn hash parameters and validates nothing.
///
/// Two tiers: a *static* tier hashed and key-sorted once, growing only by
/// `promote`, and a *candidate* tier refreshed per epoch by `update`. The
/// index is exact after every operation — the hash decides which points are
/// compared, never how far apart they are.
#[pyclass]
struct RustLshIndex {
    b: i64,
    kdim: usize,
    l: usize,
    probes: usize,
    /// Queries per batched block; 0 = auto (see `block_size`).
    block: usize,
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
    /// The candidate tier's direct-address offsets (see `build_offsets`).
    offs_c: Vec<u32>,
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
    pairs: std::sync::atomic::AtomicU64,
    relaxed: std::sync::atomic::AtomicU64,
}

/// Memory budget for the direct-address offset tables, in bytes, counted
/// over **both** tiers. The tuner keeps `B^K ~ n/64`, so real indexes use a
/// few MB; the budget only trips on hand-forced parameters, where every
/// lookup falls back to binary search.
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
        // Two tiers now carry offsets, so each may claim half the budget.
        if entries.saturating_mul(self.l * 4 * 2) <= OFFS_BUDGET {
            bk as usize
        } else {
            0
        }
    }

    /// Direct-address offsets over `[0, B^K]` for one tier's sorted keys —
    /// `offs[t][v] = lower_bound(keys[t], v)`, so any bucket or prefix range
    /// is two loads instead of a binary search. O(n + B^K) per table; empty
    /// when the key space is over `OFFS_BUDGET`, which puts every lookup
    /// back on the search path.
    fn build_offsets(&self, keys: &[i64], n: usize) -> Vec<u32> {
        let bk = self.key_space();
        if bk == 0 || n == 0 {
            return Vec::new();
        }
        let mut offs = vec![0u32; self.l * (bk + 1)];
        offs.par_chunks_mut(bk + 1).enumerate().for_each(|(t, ot)| {
            let sk = &keys[t * n..(t + 1) * n];
            let mut i = 0usize;
            for v in 0..=bk {
                while i < n && sk[i] < v as i64 {
                    i += 1;
                }
                ot[v] = i as u32;
            }
        });
        offs
    }

    /// Called at build() and promote(); update() never moves the static tier.
    fn build_static_offsets(&mut self) {
        self.offs_s = self.build_offsets(&self.skeys_s, self.n_static);
    }

    /// Called wherever the candidate tier is rebuilt or re-placed — build(),
    /// update(), promote(). The candidate tier is the half of the index the
    /// ESS self-join hits every epoch, so it earns direct addressing as much
    /// as the static tier: without it every bucket group in the batched join
    /// paid two binary searches over `n_candidates` sorted keys.
    fn build_cand_offsets(&mut self) {
        self.offs_c = self.build_offsets(&self.skeys_c, self.n_cand());
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
        use std::sync::atomic::{AtomicI64, Ordering::Relaxed};
        let mut keys = vec![0i64; self.l * n];
        // Parallel over point chunks; each chunk fills a small t-major
        // buffer (point row stays in L1 across all tables), then lands in
        // the shared array one contiguous run per table. Chunk writes are
        // disjoint by construction — but *by construction* is an argument,
        // not a proof the compiler can check, and the shape (each worker
        // scattering into `l` separate runs of one shared buffer) is not
        // one `par_chunks_mut` can express. It used to be spelled with a
        // raw pointer behind a hand-written `unsafe impl Sync`, which
        // silences the checker for the whole buffer, not just for the
        // disjointness being claimed.
        //
        // `AtomicI64::from_mut_slice` (Rust 1.98) buys the same sharing
        // with none of that: it reborrows the `&mut [i64]` we already own
        // exclusively as `&[AtomicI64]`, which is `Sync` on its own terms,
        // so rayon takes it with no `unsafe` and no unchecked claim. The
        // stores are `Relaxed` and disjoint, and the borrow ends before
        // `keys` is returned, so there is no ordering to reason about and
        // no atomic left in the result. On x86-64 a relaxed 64-bit store
        // is a plain `mov` — see OPTIMIZE.md for what it costs (nothing
        // measurable) against the memcpy it replaced.
        //
        // Chunk size shrinks with n so small tiers (the per-epoch update)
        // still fan out over all workers.
        let chunk = (n.div_ceil(4 * rayon::current_num_threads())).clamp(16, 1024);
        let out: &[AtomicI64] = AtomicI64::from_mut_slice(&mut keys);
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
            for t in 0..self.l {
                let dst = &out[t * n + base..t * n + base + len];
                for (slot, &v) in dst.iter().zip(&buf[t * len..(t + 1) * len]) {
                    slot.store(v, Relaxed);
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
        self.build_cand_offsets();
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
    fn query_codes(&self, x: &[f64], t: usize, codes: &mut [i64], frac: &mut [f64]) -> i64 {
        let st = &self.s[t * self.kdim..(t + 1) * self.kdim];
        let ut = &self.u[t * self.kdim..(t + 1) * self.kdim];
        let mut key = 0i64;
        for j in 0..self.kdim {
            let f = wrap_unit(x[st[j] as usize] + ut[j]) * self.b as f64;
            let mut code = f as i64;
            if code > self.b - 1 {
                code = self.b - 1;
            }
            codes[j] = code;
            frac[j] = (f - code as f64).clamp(0.0, 1.0);
            key += code * self.pw[j];
        }
        key
    }

    /// The normative probe set of one query in one table: the query's own
    /// bucket key, then `min(probes, K)` neighbour-cell keys, each obtained
    /// by stepping the digit whose in-cell fraction sits closest to a cell
    /// wall (nearest wall first). Emitted through a callback so both the
    /// per-query gather and the batched join build their own layout.
    #[inline]
    fn probe_keys<F: FnMut(i64)>(
        &self,
        x: &[f64],
        t: usize,
        codes: &mut [i64],
        frac: &mut [f64],
        clos: &mut [f64],
        mut emit: F,
    ) {
        let key = self.query_codes(x, t, codes, frac);
        emit(key);
        let nprobe = self.probes.min(self.kdim);
        if nprobe == 0 {
            return;
        }
        for j in 0..self.kdim {
            let f = frac[j];
            clos[j] = f.min(1.0 - f);
        }
        for _ in 0..nprobe {
            let mut best = 0usize;
            let mut best_c = 2.0f64;
            for j in 0..self.kdim {
                if clos[j] < best_c {
                    best_c = clos[j];
                    best = j;
                }
            }
            clos[best] = 2.0; // consume
            let dir: i64 = if frac[best] >= 0.5 { 1 } else { -1 };
            let alt = (codes[best] + dir).rem_euclid(self.b);
            emit(key + (alt - codes[best]) * self.pw[best]);
        }
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
        {
            let Ws {
                codes,
                frac,
                clos,
                kbuf,
                ..
            } = &mut *ws;
            for t in 0..self.l {
                self.probe_keys(x, t, codes, frac, clos, |key| kbuf.push((t as u32, key)));
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
        // Candidate tier, direct-addressed the same way.
        if self.n_cand() > 0 {
            for i in 0..m {
                let (t, key) = ws.kbuf[i];
                let (_, si) = self.cand_tier(t as usize);
                let (lo, hi) = self.cand_range(t as usize, key, 1);
                for &id in &si[lo..hi] {
                    if ws.stamp[id as usize] != ws.stamp_now {
                        ws.stamp[id as usize] = ws.stamp_now;
                        ws.cand.push(id);
                    }
                }
            }
        }
    }

    // Candidate ids are effectively random, so the hardware prefetcher
    // cannot help and a software hint used to run eight rows ahead here. It
    // was the crate's only `unsafe` -- `_mm_prefetch` is an intrinsic, so it
    // is `unsafe` on target-feature grounds even though the hint itself
    // cannot fault, alias or race. Re-measured, it was worth 3-5% at d=64
    // and at n=32768/d=100, nothing at d=8, 32 and 100, and slightly
    // *negative* at d=2. That is not enough to keep the one construct
    // standing between this crate and `#![forbid(unsafe_code)]`, which is a
    // guarantee a compiler checks rather than one a comment asserts.
    //
    // `#[target_feature(enable = "sse")]` is not a way to keep it: it does
    // not remove the `unsafe` at the call site, and it cannot be combined
    // with `#[inline(always)]`, so the hint would cost a real function call.
    fn refine_topk(&self, ws: &Ws, top: &mut TopK, ex: i64) {
        for &id in ws.cand.iter() {
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
            (
                ot[lo_key as usize] as usize,
                ot[(lo_key + width) as usize] as usize,
            )
        } else {
            let (sk, _) = self.stat(t);
            (lower_bound(sk, lo_key), lower_bound(sk, lo_key + width))
        }
    }

    /// [lo, hi) of `[lo_key, lo_key + width)` in the sorted candidate keys.
    #[inline]
    fn cand_range(&self, t: usize, lo_key: i64, width: i64) -> (usize, usize) {
        if !self.offs_c.is_empty() {
            let row = self.offs_c.len() / self.l;
            let ot = &self.offs_c[t * row..(t + 1) * row];
            (
                ot[lo_key as usize] as usize,
                ot[(lo_key + width) as usize] as usize,
            )
        } else {
            let (sk, _) = self.cand_tier(t);
            (lower_bound(sk, lo_key), lower_bound(sk, lo_key + width))
        }
    }

    fn range_count(&self, t: usize, lo_key: i64, width: i64) -> usize {
        let (lo, hi) = self.static_range(t, lo_key, width);
        let mut cnt = hi - lo;
        if self.n_cand() > 0 {
            let (clo, chi) = self.cand_range(t, lo_key, width);
            cnt += chi - clo;
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
            let (_, si) = self.cand_tier(t);
            let (lo, hi) = self.cand_range(t, lo_key, width);
            for &id in &si[lo..hi] {
                if ws.stamp[id as usize] != ws.stamp_now {
                    ws.stamp[id as usize] = ws.stamp_now;
                    ws.cand.push(id);
                }
            }
        }
    }

    /// Queries per block of the batched join: one block per worker.
    ///
    /// Reuse is everything here — a bucket is loaded once per block, so the
    /// cost of finding and gathering it is divided by the number of queries
    /// in the block that probe it. Measured on the ESS case (50k+50k, d=32,
    /// 16 workers): 782 queries/block 531 ms, 1563 445 ms, 3125 (= m/workers)
    /// **404 ms**. Keeping the block's query tile L2-resident loses to
    /// reuse, so blocks are as large as the worker count allows and nothing
    /// smaller. Explicit `query_block_size` still caps it, for A/B work.
    fn block_size(&self, m: usize) -> usize {
        let workers = rayon::current_num_threads().max(1);
        let qb = m.div_ceil(workers).max(1);
        if self.block > 0 {
            qb.min(self.block)
        } else {
            qb
        }
    }

    /// Score one tile of points against every query of a bucket group.
    ///
    /// The inner loop is the whole point of the batched shape: both
    /// operands are contiguous and cache-resident, the id is only touched
    /// when a candidate beats the query's k-th best, and the heap top is
    /// held in a register across the tile.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn score_tile<const D: usize>(
        &self,
        group: &[(i64, u32)],
        tile: &[f32],
        tids: &[i64],
        q32: &[f32],
        exq: &[i64],
        heaps: &mut [(u32, i64)],
        hlen: &mut [u32],
        k: usize,
    ) {
        let d = self.d;
        let nt = tids.len();
        for &(_, slot) in group {
            let si = slot as usize;
            let qrow = &q32[si * d..(si + 1) * d];
            let h = &mut heaps[si * k..(si + 1) * k];
            let mut len = hlen[si] as usize;
            let ex = exq[si];
            let mut worst = if len == k { h[0] } else { WORST };
            let mut i = 0;
            while i + ROWS <= nt {
                let ds = dist_l1_rows::<D>(qrow, &tile[i * d..(i + ROWS) * d], d);
                for r in 0..ROWS {
                    let (bits, id) = (ds[r].to_bits(), tids[i + r]);
                    if hless((bits, id), worst) && id != ex {
                        heap_offer(h, &mut len, bits, id);
                        worst = if len == k { h[0] } else { WORST };
                    }
                }
                i += ROWS;
            }
            while i < nt {
                let bits = l1::<D>(qrow, &tile[i * d..(i + 1) * d], d).to_bits();
                let id = tids[i];
                if hless((bits, id), worst) && id != ex {
                    heap_offer(h, &mut len, bits, id);
                    worst = if len == k { h[0] } else { WORST };
                }
                i += 1;
            }
            hlen[si] = len as u32;
        }
    }

    /// One block of the batched, bucket-centric join.
    ///
    /// Table by table, the block's probe keys are grouped by bucket (a sort
    /// on `(key, slot)`), and each bucket the block touches is loaded
    /// **once** and scored against every query that probes it — instead of
    /// once per query, as a per-query gather does. That turns the query
    /// into a sequence of dense `queries x points x d` tiles and removes
    /// the random-access dedup stamp entirely: a point shared with the
    /// query by several tables is simply offered to the heap several times,
    /// and `heap_offer` keeps ids distinct.
    ///
    /// Heaps are left in `ws` for the caller to relax and emit.
    fn query_block(&self, ws: &mut BWs, q: &[f64], nq: usize, k: usize, ex: Option<&[i64]>) -> u64 {
        let (d, ns, nc) = (self.d, self.n_static, self.n_cand());
        ws.reset(nq, d, k, self.kdim);
        for i in 0..nq * d {
            ws.q32[i] = q[i] as f32;
        }
        for i in 0..nq {
            ws.exq[i] = ex.map_or(-1, |e| e[i]);
        }

        let mut pairs = 0u64;
        let BWs {
            q32,
            exq,
            heaps,
            hlen,
            codes,
            frac,
            clos,
            probe,
            gids,
            tile,
            tids,
        } = ws;

        for t in 0..self.l {
            probe.clear();
            for i in 0..nq {
                let x = &q[i * d..(i + 1) * d];
                self.probe_keys(x, t, codes, frac, clos, |key| probe.push((key, i as u32)));
            }
            // Group by bucket. The keys of one table live in [0, B^K), so
            // this is a bucket-major reordering of the block's probes.
            probe.sort_unstable();

            let (sk_s, sid_s) = (
                &self.skeys_s[t * ns..(t + 1) * ns],
                &self.sids_s[t * ns..(t + 1) * ns],
            );
            let (sk_c, sid_c) = if nc > 0 {
                self.cand_tier(t)
            } else {
                (&[][..], &[][..])
            };

            let mut gs = 0usize;
            while gs < probe.len() {
                let key = probe[gs].0;
                let mut ge = gs + 1;
                while ge < probe.len() && probe[ge].0 == key {
                    ge += 1;
                }
                let group = &probe[gs..ge];
                gs = ge;
                gids.clear();
                if ns > 0 {
                    let (lo, hi) = if !self.offs_s.is_empty() {
                        let row = self.offs_s.len() / self.l;
                        let ot = &self.offs_s[t * row..(t + 1) * row];
                        (ot[key as usize] as usize, ot[key as usize + 1] as usize)
                    } else {
                        (lower_bound(sk_s, key), lower_bound(sk_s, key + 1))
                    };
                    gids.extend_from_slice(&sid_s[lo..hi]);
                }
                if nc > 0 {
                    let (lo, hi) = if !self.offs_c.is_empty() {
                        let row = self.offs_c.len() / self.l;
                        let ot = &self.offs_c[t * row..(t + 1) * row];
                        (ot[key as usize] as usize, ot[key as usize + 1] as usize)
                    } else {
                        (lower_bound(sk_c, key), lower_bound(sk_c, key + 1))
                    };
                    gids.extend_from_slice(&sid_c[lo..hi]);
                }
                if gids.is_empty() {
                    continue;
                }
                pairs += (gids.len() * group.len()) as u64;

                for chunk in gids.chunks(TILE) {
                    // Gather the scattered rows into one contiguous tile.
                    // This is where the block pays for random access, and it
                    // pays once per block rather than once per query, which
                    // is the whole reason the batched path tiles at all. A
                    // software prefetch used to run ahead of this copy; it
                    // was the crate's only `unsafe` and did not survive
                    // re-measurement (see `refine_topk`).
                    for (j, &id) in chunk.iter().enumerate() {
                        let base = id as usize * d;
                        tile[j * d..(j + 1) * d].copy_from_slice(&self.pts32[base..base + d]);
                        tids[j] = id;
                    }
                    let n = chunk.len();
                    let (tl, ti) = (&tile[..n * d], &tids[..n]);
                    // Dispatch the dimensionality into the kernel's type,
                    // once per tile (a few hundred pairs), so the match is
                    // free. **Only 8, 16 and 24 are specialized, and
                    // widening that list makes it slower.** Those three are
                    // the one-, two- and three-vector cases, where the
                    // vector loop's guard and scalar epilogue cost more
                    // than the arithmetic they guard: const `D` deletes
                    // both and is worth 20-25%. From `d = 32` up the loop amortises
                    // the guards on its own and a constant trip count
                    // instead invites LLVM to *fully unroll*, which is
                    // 41% (d=32) to 73% (d=64) slower than the vectorized
                    // loop the runtime-`d` instantiation gets. Below 8
                    // there is no vector loop to guard and const `D` costs
                    // 20% as well. Measured, all of it — see OPTIMIZE.md.
                    match d {
                        8 => self.score_tile::<8>(group, tl, ti, q32, exq, heaps, hlen, k),
                        16 => self.score_tile::<16>(group, tl, ti, q32, exq, heaps, hlen, k),
                        24 => self.score_tile::<24>(group, tl, ti, q32, exq, heaps, hlen, k),
                        _ => self.score_tile::<0>(group, tl, ti, q32, exq, heaps, hlen, k),
                    }
                }
            }
        }
        pairs
    }

    /// Prefix relaxation for one under-filled query (see `lsh.py`).
    fn relax_query(&self, ws: &mut Ws, top: &mut TopK, x: &[f64], k: usize, ex: i64) {
        let mut base = vec![0i64; self.l];
        for t in 0..self.l {
            base[t] = self.query_codes(x, t, &mut ws.codes, &mut ws.frac);
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

    /// k-NN over blocks of queries: the batched, bucket-centric join.
    ///
    /// Parallel over blocks, so each worker owns its block's heaps
    /// exclusively — no contention and no per-worker `n`-sized state unless
    /// a query actually needs relaxation.
    #[allow(clippy::too_many_arguments)]
    fn knn_batched(
        &self,
        q: &[f64],
        d: usize,
        k: usize,
        kk: usize,
        ex: Option<&[i64]>,
        out_idx: &mut [i64],
        out_dst: &mut [f64],
    ) {
        let m = out_idx.len() / k;
        let qb = self.block_size(m);
        out_idx
            .par_chunks_mut(qb * k)
            .zip(out_dst.par_chunks_mut(qb * k))
            .enumerate()
            .for_each_init(
                || (BWs::default(), None::<Ws>, TopK::new(k)),
                |(bws, ws, top), (bi, (oi, od))| {
                    use std::sync::atomic::Ordering::Relaxed;
                    let q0 = bi * qb;
                    let nq = oi.len() / k;
                    let t0 = std::time::Instant::now();
                    let pairs = self.query_block(
                        bws,
                        &q[q0 * d..(q0 + nq) * d],
                        nq,
                        k,
                        ex.map(|e| &e[q0..q0 + nq]),
                    );
                    let t1 = std::time::Instant::now();
                    let mut relaxed = 0u64;
                    for i in 0..nq {
                        let len = bws.hlen[i] as usize;
                        let (row_idx, row_dst) =
                            (&mut oi[i * k..(i + 1) * k], &mut od[i * k..(i + 1) * k]);
                        if len < kk {
                            let x = &q[(q0 + i) * d..(q0 + i + 1) * d];
                            let w = ws.get_or_insert_with(|| Ws::new(self.n_points, d, self.kdim));
                            for j in 0..d {
                                w.q32[j] = x[j] as f32;
                            }
                            self.relax_query(w, top, x, k, bws.exq[i]);
                            top.emit(row_idx, row_dst);
                            relaxed += 1;
                        } else {
                            heap_emit(&mut bws.heaps[i * k..(i + 1) * k], len, row_idx, row_dst);
                        }
                    }
                    let st = &self.stats;
                    st.refine_ns.fetch_add((t1 - t0).as_nanos() as u64, Relaxed);
                    st.relax_ns
                        .fetch_add(t1.elapsed().as_nanos() as u64, Relaxed);
                    st.pairs.fetch_add(pairs, Relaxed);
                    st.queries.fetch_add(nq as u64, Relaxed);
                    st.relaxed.fetch_add(relaxed, Relaxed);
                },
            );
    }

    /// k-NN one query at a time: gather a deduplicated candidate list, then
    /// refine it. Serves batches too small for the join to amortise its
    /// bucket loads over, where scoring each candidate exactly once is
    /// worth the random-access dedup stamp.
    #[allow(clippy::too_many_arguments)]
    fn knn_per_query(
        &self,
        q: &[f64],
        d: usize,
        k: usize,
        kk: usize,
        ex: Option<&[i64]>,
        out_idx: &mut [i64],
        out_dst: &mut [f64],
    ) {
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
                    let exq = ex.map_or(-1, |e| e[qi]);
                    let t0 = std::time::Instant::now();
                    self.gather_query(ws, x);
                    let t1 = std::time::Instant::now();
                    top.clear();
                    self.refine_topk(ws, top, exq);
                    let t2 = std::time::Instant::now();
                    let st = &self.stats;
                    st.cands.fetch_add(ws.cand.len() as u64, Relaxed);
                    st.pairs.fetch_add(ws.cand.len() as u64, Relaxed);
                    if top.len < kk {
                        self.relax_query(ws, top, x, k, exq);
                        st.relax_ns
                            .fetch_add(t2.elapsed().as_nanos() as u64, Relaxed);
                        st.relaxed.fetch_add(1, Relaxed);
                    }
                    top.emit(row_idx, row_dst);
                    st.gather_ns.fetch_add((t1 - t0).as_nanos() as u64, Relaxed);
                    st.refine_ns.fetch_add((t2 - t1).as_nanos() as u64, Relaxed);
                    st.queries.fetch_add(1, Relaxed);
                },
            );
    }
}

#[pymethods]
impl RustLshIndex {
    #[new]
    /// Build an empty index from drawn hash parameters.
    ///
    /// Args:
    ///     b: Cells per dimension, `B >= 2`.
    ///     k: Sampled dimensions per table, `K`.
    ///     l: Number of tables, `L`.
    ///     probes: Neighbour-cell probes per table per query.
    ///     s: `(L, K)` int64 sampled dimension indices.
    ///     u: `(L, K)` float64 per-dimension offsets in `[0, 1)`.
    ///     block: Queries per block of the batched join; `1` selects the
    ///         per-query path, `0` sizes blocks from the worker count.
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
            block,
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
            offs_c: Vec::new(),
            stats: QueryStats::default(),
        })
    }

    /// Hash and sort both tiers from scratch.
    ///
    /// Args:
    ///     points: `(n, d)` float64 in `[0, 1)`; the first `n_static` rows
    ///         are the static tier, the rest the candidate tier.
    ///     n_static: Size of the static tier.
    fn build(
        &mut self,
        py: Python<'_>,
        points: PyReadonlyArray2<f64>,
        n_static: usize,
    ) -> PyResult<()> {
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
            self.build_cand_offsets();
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

    /// Exact k nearest neighbours of each query under toroidal L1.
    ///
    /// The hash selects candidates; every returned distance is exact. Rows
    /// are ascending by `(distance, id)` and padded with `-1` / `inf` when
    /// fewer than `k` points exist.
    ///
    /// Args:
    ///     queries: `(m, d)` float64 in `[0, 1)`.
    ///     k: Neighbours per query.
    ///     exclude_ids: Optional `(m,)` int64, one id excluded per query —
    ///         the self-join.
    ///
    /// Returns:
    ///     `(idx, dist)`, both `(m, k)`: int64 ids and float64 distances.
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
            if k == 0 || m == 0 {
                return;
            }
            let reachable = self.n_points - usize::from(ex.is_some());
            let kk = k.min(reachable);
            // block == 1 asks for one query per block, i.e. no batching:
            // the per-query path is that, without the wasted machinery.
            if m >= BATCH_MIN && self.block != 1 {
                self.knn_batched(&q, d, k, kk, ex.as_deref(), &mut out_idx, &mut out_dst);
            } else {
                self.knn_per_query(&q, d, k, kk, ex.as_deref(), &mut out_idx, &mut out_dst);
            }
        });
        Ok((
            PyArray1::from_vec(py, out_idx).reshape([m, k])?,
            PyArray1::from_vec(py, out_dst).reshape([m, k])?,
        ))
    }

    /// Every point within `radius` of each query, exactly.
    ///
    /// Args:
    ///     queries: `(m, d)` float64 in `[0, 1)`.
    ///     radius: Toroidal-L1 cutoff.
    ///     exclude_ids: Optional `(m,)` int64, one id excluded per query.
    ///
    /// Returns:
    ///     `(indptr, ids, dists)` in CSR form: query `i` owns
    ///     `ids[indptr[i]:indptr[i+1]]`.
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
                            let p = &self.pts32[id as usize * self.d..(id as usize + 1) * self.d];
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

    /// The raw hash tables, for cross-implementation comparison.
    ///
    /// Returns the sorted keys and ids of both tiers. These are what
    /// "byte-identical to the reference" is asserted on, so the layout is
    /// part of the contract rather than an implementation detail.
    fn tables<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let ns = self.n_static;
        let nc = self.n_cand();
        let d = PyDict::new(py);
        d.set_item(
            "static_keys",
            self.skeys_s
                .clone()
                .into_pyarray(py)
                .reshape([self.l, ns])?,
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
            self.skeys_c
                .clone()
                .into_pyarray(py)
                .reshape([self.l, nc])?,
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
        d.set_item("pairs", self.stats.pairs.load(Relaxed))?;
        d.set_item("relaxed", self.stats.relaxed.load(Relaxed))?;
        d.set_item("block", self.block)?;
        Ok(d)
    }

    /// Zero the counters read by `stats`.
    fn reset_stats(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.stats.gather_ns.store(0, Relaxed);
        self.stats.refine_ns.store(0, Relaxed);
        self.stats.relax_ns.store(0, Relaxed);
        self.stats.queries.store(0, Relaxed);
        self.stats.cands.store(0, Relaxed);
        self.stats.pairs.store(0, Relaxed);
        self.stats.relaxed.store(0, Relaxed);
    }

    /// Points in the index, both tiers.
    #[getter]
    fn n_points(&self) -> usize {
        self.n_points
    }

    /// Points in the static tier.
    #[getter]
    fn n_static(&self) -> usize {
        self.n_static
    }

    /// Points in the candidate tier.
    #[getter]
    fn n_candidates(&self) -> usize {
        self.n_cand()
    }
}

/// Toroidal L1 in f64, for the exact-contract brute path.
///
/// f32 is right for the LSH candidate scan, where a few ulp cannot change
/// which bucket a point sits in and the width buys throughput. It is wrong
/// here: `BruteIndex` is the reference the LSH implementations are validated
/// against and what `metrics.toroidal_separation` reports with, and its tests
/// pin distances to nine decimal places. In f32 the wrap distance 0.02 comes
/// back as 0.019999980926513672, which is a real change of contract rather
/// than a rounding detail.
///
/// The algebraic methods still apply — they permit reassociation and FMA
/// contraction, not a change of width — so this is a few ulp of f64 on a sum
/// of `d` terms in `[0, 0.5]`, about 1e-16 rather than 2e-8.
#[inline(always)]
fn l1_64<const D: usize>(a: &[f64], b: &[f64], d: usize) -> f64 {
    let n = if D == 0 { d } else { D };
    let mut s = 0.0f64;
    for j in 0..n {
        let t = a[j].algebraic_sub(b[j]).abs();
        let w = 1.0f64.algebraic_sub(t);
        s = s.algebraic_add(if t < w { t } else { w });
    }
    s
}

/// Bounded max-heap over f64 distances, keyed on the bit pattern.
///
/// The f32 heap above cannot serve: `f64::to_bits` is a `u64`. Non-negative
/// floats order like their bit patterns in either width, so the comparison
/// is the same one.
#[inline]
fn heap_offer_64(h: &mut [(u64, i64)], len: &mut usize, bits: u64, id: i64) {
    let k = h.len();
    if k == 0 {
        return;
    }
    let e = (bits, id);
    if *len < k {
        h[*len] = e;
        let mut i = *len;
        *len += 1;
        while i > 0 {
            let up = (i - 1) >> 1;
            if h[up] >= h[i] {
                break;
            }
            h.swap(up, i);
            i = up;
        }
    } else if e < h[0] {
        h[0] = e;
        let mut i = 0;
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut big = i;
            if l < k && h[big] < h[l] {
                big = l;
            }
            if r < k && h[big] < h[r] {
                big = r;
            }
            if big == i {
                break;
            }
            h.swap(i, big);
            i = big;
        }
    }
}

/// `l1_64` with the width chosen at run time.
///
/// The const-generic specializations exist so LLVM can fully unroll the
/// widths this workload actually sees; picking between them is the only part
/// that has to know which ones are worth specializing, so both brute kernels
/// share this one copy of the decision rather than each carrying a `match`.
#[inline(always)]
fn l1_64_dyn(a: &[f64], b: &[f64], d: usize) -> f64 {
    match d {
        8 => l1_64::<8>(a, b, d),
        16 => l1_64::<16>(a, b, d),
        24 => l1_64::<24>(a, b, d),
        _ => l1_64::<0>(a, b, d),
    }
}

/// The arrays a brute kernel scans, copied out of NumPy once.
///
/// Both kernels release the GIL for the scan, so they cannot hold a
/// `PyReadonlyArray` across it and the data has to be owned. The copy and
/// the shape checks are identical either way, so they live here once.
struct BruteInputs {
    pts: Vec<f64>,
    qs: Vec<f64>,
    ex: Option<Vec<i64>>,
    n: usize,
    m: usize,
    d: usize,
}

fn brute_inputs(
    points: &PyReadonlyArray2<'_, f64>,
    queries: &PyReadonlyArray2<'_, f64>,
    exclude_ids: &Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<BruteInputs> {
    let (n, d) = (points.shape()[0], points.shape()[1]);
    let m = queries.shape()[0];
    if queries.shape()[1] != d {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "points and queries must have the same number of columns",
        ));
    }
    // Checked rather than left to index into the slice: the scan runs inside
    // rayon with the GIL released, where an out-of-range id would surface as
    // a panic unwinding through the pool instead of as a Python error.
    if let Some(e) = exclude_ids {
        if e.shape()[0] != m {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "exclude_ids must have one id per query",
            ));
        }
    }
    Ok(BruteInputs {
        pts: points.as_slice()?.to_vec(),
        qs: queries.as_slice()?.to_vec(),
        ex: match exclude_ids {
            Some(e) => Some(e.as_slice()?.to_vec()),
            None => None,
        },
        n,
        m,
        d,
    })
}

/// Exact toroidal-L1 k-NN by scanning every point, in parallel.
///
/// **Why this exists at all.** The Python `BruteIndex` is what runs below
/// `ToroidalNN`'s crossover, and until now that was the only path a caller
/// under the threshold could take — so the compiled kernel served
/// exclusively the regime a population-sized caller never enters. ESS and
/// OBLESA sit at 60-400 points and spent 90% of their initialization inside
/// that NumPy scan.
///
/// **Why it is faster than the NumPy it replaces**, which is already a good
/// implementation — per-axis accumulation, eight lanes, in place. Three
/// structural things NumPy cannot do from Python:
///
/// 1. **No `(m, n)` matrix.** NumPy must materialise every distance before
///    it can `argpartition` them, so the array is written once and read
///    again. Here the distance is consumed by the heap the instant it is
///    computed and never reaches memory.
/// 2. **One pass.** The fold is four passes over `(m, n)` in NumPy —
///    subtract, abs, subtract, minimum — against one fused pass per pair,
///    in registers.
/// 3. **Every core.** The NumPy path is single-threaded.
///
/// **What it costs.** f64, not f32 — see `l1_64` for why the width is not
/// negotiable here. Not bit-identical to the NumPy scan even so: the
/// algebraic float methods permit reassociation and FMA contraction, so this
/// is a few ulp on a sum of `d` terms in `[0, 0.5]`, around 1e-16. That can
/// still reorder two candidates already within an ulp of each other, which is
/// why the conformance test allows ids to differ on ties.
///
/// Args:
///     points: `(n, d)` float64 in `[0, 1)`.
///     queries: `(m, d)` float64 in `[0, 1)`.
///     k: Neighbours per query.
///     exclude_ids: Optional `(m,)` int64, one point id excluded per query.
///
/// Returns:
///     `(idx, dist)` of shape `(m, k)`, rows ascending by distance, padded
///     with `-1` / `inf` where fewer than `k` points qualify.
#[pyfunction]
#[pyo3(signature = (points, queries, k, exclude_ids=None))]
fn brute_knn<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    queries: PyReadonlyArray2<'py, f64>,
    k: usize,
    exclude_ids: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
    let BruteInputs {
        pts,
        qs,
        ex,
        n,
        m,
        d,
    } = brute_inputs(&points, &queries, &exclude_ids)?;

    let kk = k.min(n);

    let rows: Vec<(Vec<i64>, Vec<f64>)> = py.detach(|| {
        (0..m)
            .into_par_iter()
            .map_init(
                || vec![(0u64, 0i64); kk],
                |heap, qi| {
                    let mut len = 0usize;
                    let q = &qs[qi * d..(qi + 1) * d];
                    let exq = ex.as_ref().map_or(-1, |e| e[qi]);
                    for id in 0..n {
                        if id as i64 == exq {
                            continue;
                        }
                        let p = &pts[id * d..(id + 1) * d];
                        let dd = l1_64_dyn(q, p, d);
                        heap_offer_64(heap, &mut len, dd.to_bits(), id as i64);
                    }
                    let mut idx = vec![-1i64; k];
                    let mut dst = vec![f64::INFINITY; k];
                    heap[..len].sort_unstable();
                    for (i, &(bits, id)) in heap[..len].iter().enumerate() {
                        idx[i] = id;
                        dst[i] = f64::from_bits(bits);
                    }
                    (idx, dst)
                },
            )
            .collect()
    });

    let mut flat_i = Vec::with_capacity(m * k);
    let mut flat_d = Vec::with_capacity(m * k);
    for (i, dd) in rows {
        flat_i.extend_from_slice(&i);
        flat_d.extend_from_slice(&dd);
    }
    Ok((
        flat_i.into_pyarray(py).reshape([m, k])?,
        flat_d.into_pyarray(py).reshape([m, k])?,
    ))
}

/// Every point within `radius` of each query, exactly, in parallel.
///
/// The range counterpart of `brute_knn`, and it exists for the same reason:
/// `exact_radius` in NumPy has to materialise the `(m, n)` distance matrix
/// before it can threshold it, and that matrix *is* the cost -- 96-100% of
/// the call, measured by timing the block loop alone against the whole
/// function. Here the distance is compared against `radius` in the register
/// that produced it and is written only if it survives, so the matrix never
/// exists. On the shapes ESS and OBLESA give this path that is 50-70x, which
/// is the same factor `brute_knn` found and for the same three reasons: no
/// intermediate matrix, one fused pass instead of four, every core.
///
/// **Why f64, emphatically.** `brute_knn` takes f64 because `BruteIndex` is
/// the reference the LSH implementations are validated against and its tests
/// pin distances to nine decimals. A range query raises the stakes: `<=` is a
/// *threshold*, so an f32 error of ~2e-8 near the boundary does not merely
/// reorder two neighbours, it adds or drops one. The returned set would
/// change, not just its order. The algebraic float methods still apply --
/// they permit reassociation and FMA contraction, not a change of width.
///
/// So this is a few ulp of f64 away from `exact_radius`, about 1e-16 on a sum
/// of `d` terms in `[0, 0.5]`. A point sitting within 1e-16 of the cutoff can
/// still fall on either side of it; the conformance test allows exactly that
/// and nothing wider.
///
/// **Memory.** The output is the only large allocation, and it is inherent to
/// the CSR contract: a radius that matches everything returns `m * n` pairs.
/// That exposure is the caller's to manage and is unchanged from the NumPy
/// path, which additionally held the distance matrix.
///
/// Args:
///     points: `(n, d)` float64 in `[0, 1)`.
///     queries: `(m, d)` float64 in `[0, 1)`.
///     radius: Inclusive toroidal-L1 cutoff.
///     exclude_ids: Optional `(m,)` int64, one point id excluded per query.
///
/// Returns:
///     `(indptr, ids, dists)` in CSR form: query `i` owns
///     `ids[indptr[i]:indptr[i + 1]]`, ascending by distance, ties by id.
#[pyfunction]
#[pyo3(signature = (points, queries, radius, exclude_ids=None))]
fn brute_radius<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    queries: PyReadonlyArray2<'py, f64>,
    radius: f64,
    exclude_ids: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let BruteInputs {
        pts,
        qs,
        ex,
        n,
        m,
        d,
    } = brute_inputs(&points, &queries, &exclude_ids)?;

    // One `Vec` per query rather than a reusable scratch buffer: a hit list
    // is variable-length and has to outlive the closure to be assembled, so
    // a reused buffer would have to be cloned out and would allocate anyway.
    //
    // Keyed on the bit pattern like the k-NN heap, for the same reason --
    // non-negative floats order like their bits -- which also makes the sort
    // a plain tuple sort: ascending distance, then ascending id. That is the
    // order `exact_radius` produces, where a stable argsort over a row breaks
    // ties by original position, which is the id.
    let rows: Vec<Vec<(u64, i64)>> = py.detach(|| {
        (0..m)
            .into_par_iter()
            .map(|qi| {
                let q = &qs[qi * d..(qi + 1) * d];
                let exq = ex.as_ref().map_or(-1, |e| e[qi]);
                let mut hits: Vec<(u64, i64)> = Vec::new();
                for id in 0..n {
                    if id as i64 == exq {
                        continue;
                    }
                    let p = &pts[id * d..(id + 1) * d];
                    let dd = l1_64_dyn(q, p, d);
                    if dd <= radius {
                        hits.push((dd.to_bits(), id as i64));
                    }
                }
                hits.sort_unstable();
                hits
            })
            .collect()
    });

    let mut indptr = Vec::with_capacity(m + 1);
    let mut total = 0i64;
    indptr.push(total);
    for r in &rows {
        total += r.len() as i64;
        indptr.push(total);
    }
    let mut ids = Vec::with_capacity(total as usize);
    let mut dst = Vec::with_capacity(total as usize);
    for r in rows {
        for (bits, id) in r {
            ids.push(id);
            dst.push(f64::from_bits(bits));
        }
    }
    Ok((
        indptr.into_pyarray(py),
        ids.into_pyarray(py),
        dst.into_pyarray(py),
    ))
}

/// Weighted sum of unit toroidal directions, per query.
///
/// $$\mathrm{out}_i = \sum_j w_{ij}\,
///   \frac{\mathrm{sign}(\mathrm{wrap}(q_i - p_{\mathrm{ids}[i,j]}))}
///        {\lVert \mathrm{sign}(\cdot) \rVert_1}$$
///
/// **Why this is torann's and not the caller's.** It is the toroidal
/// displacement and its L1 normalization -- the metric's own definition,
/// applied per axis. What multiplies it is the caller's: `weights` arrives
/// already evaluated on an `(m, k)` array, so a force law stays an ordinary
/// Python callable and never enters this crate. That split is not a
/// convenience, it is what lets ESS keep a pure-Python wheel and a
/// user-supplied `metric` at the same time.
///
/// **Why it exists.** Profiled on ESS's force kernel at `d = 100`, the
/// `(rows, k, D)` tensor was **99.4%** of the call and the force law 0.6%.
/// At `k = 128` that tensor is 20.5 MB and four of them are live at once --
/// `disp`, `grad`, `norms`, `directions` -- so ~80 MB per block, well past
/// any L3, which is why doubling the neighbourhood cost 5x rather than 2x.
/// Here the direction is consumed into the accumulator as it is computed and
/// the tensor never exists: `O(m * D)` of working memory instead of
/// `O(m * k * D)`.
///
/// **Two semantics copied deliberately, not approximated.** NumPy rounds
/// half to *even*, so the wrap uses `round_ties_even` and not `round`, which
/// rounds half away from zero and would disagree on every exact half. And
/// `np.sign(±0.0)` is `0.0`, where Rust's `signum` returns `±1.0` -- so the
/// sign is written out rather than called. Either one would be a silent
/// per-axis error, not a crash.
///
/// **Accuracy.** Each term is `±w / nnz` with `nnz` a small integer count,
/// so the terms themselves are exactly rounded. The sum over `j` is
/// reassociated by the algebraic float methods where NumPy sums pairwise,
/// which is the only source of difference: a few ulp on a sum of `k` terms
/// of similar magnitude, with no cancellation to amplify it.
///
/// It happens to come out bit-identical to the NumPy expression on every
/// shape ESS runs, and that must not be relied on. The algebraic methods are
/// documented as *non-deterministic* -- the same source may fold differently
/// under a different compiler or target -- so the contract here is a few ulp
/// and the agreement is a courtesy, not a guarantee. Anything that needs
/// reproducibility across builds has to pin the wheel, not the source.
///
/// A negative id marks a missing neighbour and contributes nothing, which is
/// what the caller's `-1` padding means. Coincident points are *not* handled
/// here: a zero displacement yields a zero direction, and a caller that
/// wants to perturb those must do it itself, where its own RNG lives.
///
/// Args:
///     points: `(n, d)` float64 in `[0, 1)`.
///     queries: `(m, d)` float64 in `[0, 1)`.
///     ids: `(m, k)` int64 neighbour ids; negative means missing.
///     weights: `(w, m, k)` float64 multipliers -- `w` independent
///         weightings over one shared geometry. Every one of them reuses the
///         same wrap and the same per-axis signs, which is the point: a
///         caller with two force terms over one neighbour list (ESS has
///         exactly two, repulsion and attraction) would otherwise pay for
///         the displacement twice, and the displacement is 99% of the work.
///
/// Returns:
///     `(w, m, d)` float64.
#[pyfunction]
#[pyo3(signature = (points, queries, ids, weights))]
fn weighted_directions<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    queries: PyReadonlyArray2<'py, f64>,
    ids: PyReadonlyArray2<'py, i64>,
    weights: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let (n, d) = (points.shape()[0], points.shape()[1]);
    let (m, k) = (ids.shape()[0], ids.shape()[1]);
    if queries.shape()[1] != d {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "points and queries must have the same number of columns",
        ));
    }
    if queries.shape()[0] != m {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ids must have one row per query",
        ));
    }
    let nw = weights.shape()[0];
    if weights.shape()[1] != m || weights.shape()[2] != k {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights must be (w, m, k) with (m, k) matching ids",
        ));
    }
    let pts = points.as_slice()?.to_vec();
    let qs = queries.as_slice()?.to_vec();
    let idv = ids.as_slice()?.to_vec();
    let wv = weights.as_slice()?.to_vec();

    let out: Vec<f64> = py.detach(|| {
        (0..m)
            .into_par_iter()
            // The per-axis signs of one pair, so the wrap is computed once
            // and not again for the accumulation pass. `d` doubles, not
            // `k * d`.
            .map_init(
                || vec![0.0f64; d],
                |sign: &mut Vec<f64>, i| {
                    // Row `i` of every weighting, laid out `(w, d)`.
                    let mut acc = vec![0.0f64; nw * d];
                    let q = &qs[i * d..(i + 1) * d];
                    for j in 0..k {
                        let id = idv[i * k + j];
                        if id < 0 || (id as usize) >= n {
                            continue;
                        }
                        // The wrap is the expensive part, so it is skipped
                        // only when *every* weighting is zero on this pair.
                        let base = i * k + j;
                        if (0..nw).all(|t| wv[t * m * k + base] == 0.0) {
                            continue;
                        }
                        let p = &pts[id as usize * d..(id as usize + 1) * d];
                        let mut nnz = 0.0f64;
                        for l in 0..d {
                            let t = q[l].algebraic_sub(p[l]);
                            let t = t.algebraic_sub(t.round_ties_even());
                            // `np.sign`, which is 0 at both zeros --
                            // `f64::signum` is +/-1 there and would put a
                            // spurious unit step on every flat axis.
                            sign[l] = if t > 0.0 {
                                1.0
                            } else if t < 0.0 {
                                -1.0
                            } else {
                                0.0
                            };
                            nnz += sign[l].abs();
                        }
                        // `np.maximum(norms, 1e-9)`: an all-zero direction
                        // divides by the floor and stays zero either way.
                        let inv = 1.0f64.algebraic_div(if nnz > 1e-9 { nnz } else { 1e-9 });
                        for t in 0..nw {
                            let coef = wv[t * m * k + base].algebraic_mul(inv);
                            if coef == 0.0 {
                                continue;
                            }
                            let row = &mut acc[t * d..(t + 1) * d];
                            for l in 0..d {
                                row[l] = row[l].algebraic_add(coef.algebraic_mul(sign[l]));
                            }
                        }
                    }
                    acc
                },
            )
            .flatten()
            .collect()
    });

    // `(m, w, d)` came out of the row-parallel loop; the caller asked for
    // `(w, m, d)`, so the two leading axes swap.
    let mut swapped = vec![0.0f64; nw * m * d];
    for i in 0..m {
        for t in 0..nw {
            swapped[(t * m + i) * d..(t * m + i + 1) * d]
                .copy_from_slice(&out[(i * nw + t) * d..(i * nw + t + 1) * d]);
        }
    }
    swapped.into_pyarray(py).reshape([nw, m, d])
}

#[pymodule]
#[pyo3(name = "_native")]
fn torann_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustLshIndex>()?;
    m.add_function(wrap_pyfunction!(brute_knn, m)?)?;
    m.add_function(wrap_pyfunction!(brute_radius, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_directions, m)?)?;
    Ok(())
}
