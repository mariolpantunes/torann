# torann query-path optimization — profile, decisions, results

Answers `requirements.md`. Every number here is measured on this machine
(AMD Ryzen AI 7 PRO 350: 8 cores / 16 threads, AVX-512, 16 MiB L3;
Python 3.12, NumPy 2.5, Rust `-C target-cpu=native`, LTO). The workload is
the one ESS actually runs, not an ANN benchmark: **one batch, `k = 5`
(`ess.K_LOCAL`), every candidate queries the whole index each epoch, index
updated in place, repeat to convergence.**

Rejected-by-measurement variants are recorded in §6 so they are not
re-tried — the house rule from ANALYSIS.md.

---

## 1. Where the time actually goes

`requirements.md` §2 proposed the batched join on the grounds that
cross-table duplicates and partial-distance pruning were the prize. Both
were measured and **neither is**:

* **Cross-table duplicates are 6%**, not 3× (`stats()["pairs"]` vs
  `["cands"]`: 4369 raw vs 4118 unique per query at 50k+50k). The dedup
  stamp the old path maintained was almost pure overhead.
* **Pruning cannot fire in this regime.** A probe bucket can only be
  skipped if its lower bound — the query's distance to the cell wall it
  steps over — exceeds the current k-th distance. Measured:

  | d | 5-NN distance | best possible bound (mean over the probed dim) |
  |---|---|---|
  | 32 | 4.94 | 0.25 max, 0.019 mean |
  | 8 | 0.719 | 0.167 max, 0.027 mean |
  | 4 | 0.162 | 0.071 max, 0.023 mean |
  | 2 | 0.0154 | 0.062 max, 0.014 mean |

  At ESS dimensions the bound is 20–200× too small; only at `d = 2` do the
  two become comparable. Concentration of measure kills it, and the same
  argument kills partial-distance early-abandon inside the kernel (at
  `d = 32` the whole distance is ~20 packed instructions, so a checkpoint
  costs more than the dimensions it skips).

The real cost is **locality and dependency latency**. Old per-query path,
CPU-ns per query at 50k+50k: `gather` 224 µs (58%), `refine` 163 µs (42%) —
where gather is 120 bucket lookups plus ~4400 random accesses into an
`n`-sized dedup stamp, and refine chases ~4100 random 128-byte rows.

### Ablation of the batched join (50k+50k, best of 3, 364 ms total)

Components removed one at a time from the finished join:

| component | ms | share |
|---|---|---|
| probe-key hashing (24 tables × 12 dims, f64) | 35 | 11% |
| bucket-key sort + group walk | 26 | 8% |
| bucket lookups + id collection | 6 | 1.5% |
| tile gather (random rows → contiguous tile) | 22 | 6% |
| **distance kernel (loads + arithmetic)** | 118 | 32% |
| **per-pair loop, threshold, heap** | 157 | 43% |

That last row is what drove the kernel work in §3.

### What the machine is bound by

Thread scaling of the query, same work:

| threads | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| ms | 4157 | 1989 | 977 | 533 | 479 | 429 |
| efficiency | 100% | 105% | 106% | 97% | 72% | 61% |

Linear to 8 physical cores, then SMT adds only 24%: **execution-bound, not
memory-bandwidth bound.** Callgrind agrees on the memory side — 1.1 D1
misses per pair, nearly all served by L2/L3, LL misses negligible.

## 2. The batched bucket-centric join

Old shape — one rayon task per query: walk `L × (1 + probes) = 120` buckets,
dedup through an `n`-sized stamp, then score ~4100 scattered rows. A bucket
probed by 50 queries is read 50 times.

New shape — one rayon task per *block* of queries. Per table: compute every
query's probe keys, sort `(key, query)` so the block is bucket-major, then
for each bucket the block touches load its points **once** into a tile and
score them against every query that probes it. Consequences:

1. Bucket lookup and gather are paid once per block, not once per query.
2. The inner loop is a dense `queries × points × d` tile, both operands
   contiguous, the query's k-th-best held in a register across it.
3. **The dedup stamp disappears.** A point reached through several tables is
   offered to the heap several times and `heap_offer` rejects ids it already
   holds — 6% more distance work in exchange for the whole random-access
   dedup array.

One contract consequence: the top-k is ranked by `(distance, id)` instead of
by arrival order, so the k-th slot cannot depend on the order candidates are
scored in. This is observable only when two candidates have *exactly* equal
f32 distance, and it makes results independent of block size and thread
count. On every case measured the outputs are bit-identical to the old path.

### Answers to the three questions raised mid-session

* **"Do we need 120 buckets per query?"** The count is what the recall
  target buys (`requirements.md` §7 settles lowering recall), but after this
  change it is no longer *120 bucket loads per query*: a bucket is loaded
  once per block and shared. What remains per query is 120 group
  memberships — a sort entry and one heap-threshold load each.
* **"Can the hash be vectorized/parallelized?"** It is already parallel
  (rayon over point chunks in `compute_keys`, over blocks at query time) but
  the per-key arithmetic is scalar f64, and it is **11% of query time**
  measured. Vectorizing needs a gather (`x[S[t][j]]` for K sampled dims), so
  the win is bounded; it is the top remaining candidate (§5).
* **"Can we prune other buckets?"** No — see §1. Measured, not argued.

## 3. The distance kernel: what the profiler said vs what was true

Callgrind put ~30% of query instructions in iterator/bounds/pointer
scaffolding, so the first attempt reduced instruction count. It made things
**worse** (450 → 565 ms), and a cheaper threshold test plus an inlined row
copy were both neutral. Instruction counts were the wrong currency.

The disassembly settled it. `dist_l1_32` was compiling to **scalar** code —
28 `vsubss` / 14 `vminss` in the query loop, not one packed op — because
LLVM will not vectorize this reduction on its own, and the doc comment
claiming it did was stale. But making it packed *also* changed nothing,
which pointed at the real limit: for `d = 32` the loop runs one `ymm`
accumulator with a serial `vaddps` chain, then a serial 8-lane horizontal
reduction (a shuffle/`vaddss` ladder) — ~30 cycles of dependent work per
pair with nothing to overlap it. Measured 90 cycles/pair for ~66
instructions.

The fix is **ILP, not fewer instructions**: score `ROWS = 4` tile rows per
call, giving four independent accumulator chains and four independent
reductions that interleave, with the query chunk loaded once for all four.
Row-count sweep (50k, best of 3): 2 → 348 ms, **4 → 335 ms**, 6 → 351 ms,
8 → 360 ms.

Bit-identity is preserved by construction: same eight lanes, same
accumulation order per pair, tail then `acc[0..8]` in order.

### Getting the lanes without `unsafe`

Three ways to express the eight lanes were built and measured (50k case,
best of 3; all three bit-identical):

| kernel | query | `unsafe` | dependencies |
|---|---|---|---|
| plain safe Rust — `&[f32; 8]` chunks, scalar loop | 512 ms | no | none |
| **`wide::f32x8`** — safe SIMD wrapper on stable | **388 ms** | **no** | +3 (`wide`, `safe_arch`, `bytemuck`) |
| `core::arch::x86_64` intrinsics | 320 ms | yes | none |

The plain safe form is the one LLVM refuses to vectorize, and it costs 60%.
`wide` recovers most of the gap with no `unsafe` in this crate and no
nightly (`std::simd` is nightly-only, so it is not an option for a published
wheel); it lowers to AVX2 where available and SSE otherwise, keeping the
lane mapping either way. **The safe `wide` kernel is what is committed** —
the remaining 21% is not worth `unsafe` in a library, and switching back is
a one-function change if it ever is.

## 4. Results

Same build, `query_block_size=1` selecting the old per-query path — a true
A/B. Mean of 6 epochs including `update()`; results verified by SHA-256 of
all `(indices, distances)` arrays.

| case | per-query | batched | speed-up | results |
|---|---|---|---|---|
| d=32, 10k+10k, k=5 | 128.7 ms | **94.2 ms** | 1.37× | bit-identical |
| d=32, 50k+50k, k=5 | 852.5 ms | **411.6 ms** | 2.07× | bit-identical |
| d=8, 5k+5k, k=5 | 27.6 ms | **17.4 ms** | 1.58× | bit-identical |
| d=16, 20k+2k, k=10 | 30.8 ms | **23.1 ms** | 1.34× | bit-identical |

The kernel work also sped up the *old* path, so against the code this
session started from the 50k case is **1218 ms → 412 ms = 2.96×**. (With the
`unsafe` intrinsic kernel of §3 it is 369 ms = 3.3×.)

Block-size sweep (50k, 16 workers) — the knob `requirements.md` §2 found
being accepted and discarded: 782 → 531 ms, 1563 → 445 ms, 3125 (= m/workers)
→ 404 ms. Reuse beats keeping the query tile L2-resident, so `block_size`
now hands each worker exactly one block; `query_block_size` still caps it.

### ESS end to end (`ess.esa`, k_nn mode, one batch, to convergence)

| case | path | total | query | epochs | toroidal Clark-Evans |
|---|---|---|---|---|---|
| d=32, 10k+10k | per-query | 22.2 s | 73.9% (16.4 s) | 82 | 1.1915 |
| d=32, 10k+10k | batched | **13.7 s** | 68.5% (9.4 s) | 82 | 1.1915 |
| d=32, 30k+30k | per-query | 74.1 s | 75.2% (55.7 s) | 77 | 1.1583 |
| d=32, 30k+30k | batched | **42.8 s** | 61.2% (26.2 s) | 77 | 1.1583 |

1.62× / **1.73×** end to end, 1.75× / **2.13×** on query, **identical
output** — same epoch count, same quality metric to four decimals. The force
kernel's share rises only because the total shrank: 8.8 s in both 30k runs,
which is the consistency check that the speed-up is real and confined to the
query.

Query is still the largest single item (61% at 30k), so §5 remains worth
doing.

## 5. What is left, in measured order

1. **Probe-key hashing — 11% of query.** Scalar f64 today. Needs a gather
   per sampled dimension, so expect well under 11%; measure before keeping.
2. **Full bucket reuse — ~21%.** Blocks are query slices, so at 50k each
   bucket still serves only ~3.8 queries' worth of work. Making the work
   item a *(table, key-range)* over all `m` queries, with a per-worker
   `m × k` top-k slab reduced at the end, would raise that to ~61.
   Single-threaded evidence at equal work: 61× reuse 3035 ms, 3.8× 3835 ms,
   1.9× 4557 ms. Costs `threads × m × k × 12` bytes (48 MiB at m=50k, k=5 —
   fine for ESS, needs a budget guard and a fallback for large k).
3. **Bucket-key sort — part of 8%.** `sort_unstable` on `(key, slot)` could
   be a counting sort over `B^K` bins.
4. **`query_radius`** still gathers per query and allocates two `Vec`s per
   query; ESS's `search_mode="radius"` goes through it.
5. **The small-`n` path** (`requirements.md` §5). `ess._smart_init` is
   38–74% of small runs, almost all in `brute.pairwise_l1` materialising an
   `(m, n, d)` f64 block. Note the exactness subtlety: chunking over `m`
   keeps results bit-identical (NumPy's pairwise sum over `d` is untouched),
   whereas accumulating over `d` changes them by ~1e-16. Take the former.
6. **The tuner's own sample.** `_tune` builds a `(256, 8192, d)` difference
   block — ~0.5 GB of traffic at n=50k — inside ESS's `setup` share. One
   chunked loop fixes it.

## 6. Measured and rejected — do not re-try

* **Partial-distance / bucket pruning** — §1, bound is 20–200× too small at
  ESS dimensions.
* **Multi-row kernel with plain slices** (first attempt): 450 → 565 ms. It
  never vectorized; only an explicit SIMD type does.
* **`[f32; 8]` chunks as a vectorization hint**: still scalar codegen, no
  change in time. LLVM needs the SIMD type, not a shape hint.
* **Single-compare threshold fast path** (`bits <= worst.0` before the tuple
  compare): 450 → 461/453 ms, neutral.
* **Inlined fixed-width row copy** instead of `copy_from_slice` → memcpy:
  neutral, despite memcpy being 3.6% of callgrind instructions.
* **Candidate-tier direct addressing (`offs_c`)**: 404 → 402 ms, inside
  noise. Kept anyway — it is 1.5% of the profile, it costs nothing per
  update, and it removes the last binary search from the query path.
* **Keeping the query tile L2-resident** (small blocks): loses to reuse, §4.
* Everything in `requirements.md` §7 (BLAS, L^p hash, cached neighbour
  lists, trading recall away) stays settled.

## 7. Verification protocol (run for every change above)

1. `query_block_size=1` A/B in one build: SHA-256 of all `(idx, dist)`
   arrays must match across paths, 4 shapes × 6 epochs.
2. `python -m pytest test/` — 60 tests, both backends.
3. `cargo fmt --check`, `cargo clippy --release -- -D warnings`.
4. `ess.esa` end to end with the timing sink, before and after.
5. Disassemble the hot closure and check the instruction mix — instruction
   *counts* proved a poor proxy for time here, but codegen shape (packed vs
   scalar, dependency chains) explained every result that mattered.

## 8. Environment trap that invalidated the first measurements

`ess`, and any script run as `python examples/foo.py`, imports the
**installed** `torann` from `~/.local/lib/python3.12/site-packages` — which
was a build from 2026-07-14, not this repository (`sys.path[0]` is the
script's directory, and the repo is only on the path when you run from it).
The ESS decomposition in `requirements.md` §1 was therefore measured against
that older core. Rebuild and reinstall (`maturin build --release` + `pip
install --force-reinstall --no-deps target/wheels/*.whl`) before any
before/after claim about ESS, and assert the import path inside benchmark
scripts.
