# torann LSH implementation contract

Every LSH implementation (pure Python `lsh.py`, Rust `torann._native`)
satisfies the same class contract — `torann/base.py` is the abstract base —
so the wrapper, the conformance tests and the benchmarks are
implementation-agnostic. The pure-Python `PythonLshIndex` is the
**reference implementation**: given the same parameters, a native
implementation must produce **byte-identical hash tables** (keys and id
order) and equivalent query results (identical ids up to float32 distance
ties; distances within `1e-5`).

## Construction

```python
Index(B, K, L, probes, S, U, block)
```

| arg | type | meaning |
|---|---|---|
| `B` | int ≥ 2 | cells per dimension (resolution) |
| `K` | int ≥ 1 | sampled dimensions per table; `B**K` fits in int64 |
| `L` | int ≥ 1 | number of tables |
| `probes` | int ≥ 0 | neighbour-cell probes per table per query |
| `S` | int64 `(L, K)` | sampled dimension indices, drawn by the facade |
| `U` | float64 `(L, K)` | per-dimension offsets in `[0, 1)`, drawn by the facade |
| `block` | int | queries per batched block; `0` = let the backend choose, `1` = one query at a time (see `query_knn`) |

The facade draws `S` and `U` from its own RNG — backends contain **no
randomness**. This is what makes cross-backend byte-equality testable.

## The hash (normative)

For table `t` and point `x` (float64, already reduced mod 1):

```
frac_j  = ((x[S[t,j]] + U[t,j]) mod 1) * B          # float64
code_j  = min(int64(frac_j), B - 1)                  # guard float round-up
key     = sum_j code_j * B**j                        # int64
```

Truncation toward zero, float64 arithmetic throughout — any deviation breaks
byte-equality with the reference.

## Lifecycle methods

All point arrays are C-contiguous float64 `(n, d)` with values in `[0, 1)`;
the facade validates and reduces mod 1 before calling. Backends may rely on
this precondition (the Rust core evaluates the hash's `mod 1` as
`s - floor(s)`, which is bit-identical to `np.mod` for non-negative input;
supporting any other domain is an affine map in the facade, never a backend
concern). The backend keeps its own copy of the points (float64 for keys,
float32 for distance refinement).

- `build(points, n_static)` — build both tiers from zero: rows
  `[0, n_static)` are the static tier, the rest the candidate tier. Per
  tier and table: keys computed as above, then key-sorted with a **stable**
  sort (ties keep ascending id order).
- `update(coords)` — replace the candidate coordinates (`(n_candidates, d)`,
  same row order). *Selective*: re-place only entries whose key changed, by
  delete + stable merge. The tables are exact after every update.
- `promote(new_candidates)` — merge the candidate tier into the static tier
  (linear merge of sorted arrays, never a re-sort), then install
  `new_candidates` (possibly `(0, d)`) as the new candidate tier.

## Query methods

- `query_knn(queries, k, exclude_ids)` → `(idx, dist)`, shapes `(m, k)`,
  int64 / float64. Rows sorted by distance, padded with `-1` / `inf` only
  when fewer than `k` reachable points exist. `exclude_ids` is `None` or an
  int64 `(m,)` of one point id to exclude per query (the self-join).
  Distances are toroidal L1 computed in **float32**, returned as float64.
  Pipeline (normative): multi-probe gather over both tiers → dedupe →
  refine → per-query top-k; under-filled rows are completed by **prefix
  relaxation** (level ℓ widens a bucket to the contiguous sorted-key range
  `[key // B**ℓ * B**ℓ, + B**ℓ)`; level `K` spans the whole table), never by
  a brute-force scan.
  The *candidate set* is normative; the order it is visited in is not. A
  backend may answer a batch bucket-major (loading each bucket once for all
  the queries that probe it) rather than query-major, so the top-k is ranked
  by **`(distance, id)`** — id breaking exact float32 ties — which makes a
  row independent of visiting order, block size and thread count. Ids must
  be distinct within a row: a point reachable through several tables is
  still returned once.
  Multi-probe (normative): per table, probe the `min(probes, K)` digits
  whose fractional part is closest to a cell boundary, stepping each to its
  nearest neighbouring cell (`+1` if `frac ≥ 0.5` else `-1`, wrapping
  mod B).
- `query_radius(queries, radius, exclude_ids)` → CSR triple
  `(indptr, ids, dists)`: int64 `(m+1,)`, int64 `(nnz,)`, float64 `(nnz,)`;
  row `i` is `ids[indptr[i]:indptr[i+1]]`, sorted by distance. Post-filter
  on the gathered candidate set (no relaxation, no exactness guarantee).

## Introspection

- `tables()` → dict of arrays (copies or read-only views), used by the
  conformance tests and benchmarks:
  - `static_keys` int64 `(L, n_static)` — sorted static-tier keys
  - `static_ids` int64 `(L, n_static)` — ids aligned to `static_keys`
  - `cand_keys` int64 `(L, n_candidates)` — *unsorted*, aligned to candidate
    row order (used to count key churn)
  - `cand_sorted_keys`, `cand_sorted_ids` — the sorted candidate tables
- Attributes: `n_points`, `n_static`, `n_candidates`.

## Packaging

One maturin mixed Rust/Python wheel: the compiled core is
`torann._native` (exporting `RustLshIndex`, adapted by `torann/rust.py`);
`torann/wrapper.py` resolves names, and `backend="auto"` picks the first
importable of `rust, python`. (The C contender from the phase-6 bake-off
is preserved at tag `archive/backend-c`; it implements this same
contract.)
