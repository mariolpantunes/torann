//! torann's native core: the toroidal L1 LSH index in Rust.
//!
//! Implements `torann/CONTRACT.md` exactly: given the same parameters
//! the tables are byte-identical to the pure-NumPy reference. The tie rules
//! that guarantee it:
//!   - tier builds sort by (key, id)            [stable argsort of asc. ids]
//!   - selective update merges new-before-kept  [np.searchsorted side=left]
//!   - promote merges candidate-before-static   [np.searchsorted side=left]
//! and keys are computed in f64 exactly as the reference:
//!   code = min((frac(x + u) * B) as i64, B - 1).

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

/// Upper bound: first index whose value is > v.
#[inline]
fn upper_bound(a: &[i64], v: i64) -> usize {
    a.partition_point(|&x| x <= v)
}

#[inline]
fn dist_l1_32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for j in 0..a.len() {
        let t = (a[j] - b[j]).abs();
        let w = 1.0f32 - t;
        s += if t < w { t } else { w };
    }
    s
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
    stamp: Vec<i64>,
    stamp_now: i64,
    cand: Vec<i64>,
    q32: Vec<f32>,
    codes: Vec<i64>,
    frac: Vec<f64>,
    clos: Vec<f64>,
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
    keys_c: Vec<i64>,
    skeys_c: Vec<i64>,
    sids_c: Vec<i64>,
}

impl RustLshIndex {
    #[inline]
    fn n_cand(&self) -> usize {
        self.n_points - self.n_static
    }

    /// The normative hash of one point for one table.
    #[inline]
    fn point_key(&self, x: &[f64], t: usize) -> i64 {
        let st = &self.s[t * self.kdim..(t + 1) * self.kdim];
        let ut = &self.u[t * self.kdim..(t + 1) * self.kdim];
        let mut key = 0i64;
        for j in 0..self.kdim {
            // x, u >= 0 so `%` (fmod) matches np.mod; `as` truncates.
            let frac = ((x[st[j] as usize] + ut[j]) % 1.0) * self.b as f64;
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
        // parallel over points, transposed writes gathered per table
        let rows: Vec<Vec<i64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let x = &self.pts[(row0 + i) * self.d..(row0 + i + 1) * self.d];
                (0..self.l).map(|t| self.point_key(x, t)).collect()
            })
            .collect();
        for (i, row) in rows.iter().enumerate() {
            for t in 0..self.l {
                keys[t * n + i] = row[t];
            }
        }
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

    fn gather_bucket(&self, ws: &mut Ws, t: usize, key: i64) {
        let (sk, si) = self.stat(t);
        let (lo, hi) = (lower_bound(sk, key), upper_bound(sk, key));
        for &id in &si[lo..hi] {
            if ws.stamp[id as usize] != ws.stamp_now {
                ws.stamp[id as usize] = ws.stamp_now;
                ws.cand.push(id);
            }
        }
        if self.n_cand() > 0 {
            let (sk, si) = self.cand_tier(t);
            let (lo, hi) = (lower_bound(sk, key), upper_bound(sk, key));
            for &id in &si[lo..hi] {
                if ws.stamp[id as usize] != ws.stamp_now {
                    ws.stamp[id as usize] = ws.stamp_now;
                    ws.cand.push(id);
                }
            }
        }
    }

    /// Codes + in-cell fractions of one query for one table.
    fn query_codes(&self, x: &[f64], t: usize, ws: &mut Ws) -> i64 {
        let st = &self.s[t * self.kdim..(t + 1) * self.kdim];
        let ut = &self.u[t * self.kdim..(t + 1) * self.kdim];
        let mut key = 0i64;
        for j in 0..self.kdim {
            let f = ((x[st[j] as usize] + ut[j]) % 1.0) * self.b as f64;
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
    fn gather_query(&self, ws: &mut Ws, x: &[f64]) {
        ws.stamp_now += 1;
        ws.cand.clear();
        let nprobe = self.probes.min(self.kdim);
        for t in 0..self.l {
            let key = self.query_codes(x, t, ws);
            self.gather_bucket(ws, t, key);
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
                self.gather_bucket(ws, t, pkey);
            }
        }
    }

    fn refine_topk(&self, ws: &Ws, top: &mut TopK, ex: i64) {
        for &id in &ws.cand {
            if id == ex {
                continue;
            }
            let p = &self.pts32[id as usize * self.d..(id as usize + 1) * self.d];
            top.push(dist_l1_32(&ws.q32, p), id);
        }
    }

    fn range_count(&self, t: usize, lo_key: i64, width: i64) -> usize {
        let (sk, _) = self.stat(t);
        let mut cnt = lower_bound(sk, lo_key + width) - lower_bound(sk, lo_key);
        if self.n_cand() > 0 {
            let (sk, _) = self.cand_tier(t);
            cnt += lower_bound(sk, lo_key + width) - lower_bound(sk, lo_key);
        }
        cnt
    }

    fn gather_range(&self, ws: &mut Ws, t: usize, lo_key: i64, width: i64) {
        let (sk, si) = self.stat(t);
        let (lo, hi) = (lower_bound(sk, lo_key), lower_bound(sk, lo_key + width));
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
            keys_c: Vec::new(),
            skeys_c: Vec::new(),
            sids_c: Vec::new(),
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
                        let x = &q[qi * d..(qi + 1) * d];
                        for j in 0..d {
                            ws.q32[j] = x[j] as f32;
                        }
                        let exq = ex.as_ref().map_or(-1, |e| e[qi]);
                        self.gather_query(ws, x);
                        top.clear();
                        self.refine_topk(ws, top, exq);
                        if top.heap.len() < kk {
                            self.relax_query(ws, top, x, k, exq);
                        }
                        top.emit(row_idx, row_dst);
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
