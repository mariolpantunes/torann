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

## 4a. The whole gain, against the build this work started from

Everything above is an A/B inside one build. This is the question a user of
the library asks instead: what does ESS get, out of the box, from the
optimized build versus `d8c9904`? Both builds compiled from their own
worktree, driven by the same script (`headline.py`), defaults only, six
representative shapes, one seed:

| shape | base total | new total | | base query | new query | |
|---|---|---|---|---|---|---|
| d=2, 0+256 | 0.428 s | **0.179 s** | 2.40× | 0.396 s | 0.147 s | 2.71× |
| d=2, 256+512 | 0.330 s | **0.241 s** | 1.37× | 0.063 s | 0.065 s | 0.98× |
| d=8, 0+1024 | 0.457 s | **0.412 s** | 1.11× | 0.252 s | 0.189 s | 1.34× |
| d=8, 1024+2048 | 1.164 s | **0.854 s** | 1.36× | 0.855 s | 0.551 s | 1.55× |
| d=32, 0+10k | 20.449 s | **13.818 s** | 1.48× | 15.676 s | 8.256 s | 1.90× |
| d=32, 10k+20k | 96.216 s | **58.730 s** | 1.64× | 81.948 s | 44.902 s | 1.83× |
| **suite** | **119.04 s** | **74.23 s** | **1.60×** | 99.19 s | 54.11 s | **1.83×** |

Query fell from 83% to 73% of ESS wall time. Per-shape geometric mean 1.51×.

**Wall time alone overstates it at d=32.** Epoch count varies ±40% with the
seed (145/97/122 on the baseline over three seeds), so single-seed wall time
measures the trajectory as much as the code. Normalised per epoch, three
seeds, d=32 0+10k: total **162 → 119 ms/epoch (1.36×)**, query **116 → 72
ms/epoch (1.60×)**. Epoch-normalised geometric mean over the six shapes:
1.47× total, 1.59× query. Those are the numbers to quote.

### Why the d=32 trajectories are not reproducible against the old build

The two d=32 shapes end at different point sets. That is not drift — both
builds are deterministic (same build, two runs, bit-identical), and every
direct comparison of the index agrees: `(ids, distances)` are bit-identical
across builds at d ∈ {2,4,8,9,16,32}, on explicit queries and on the ESS
self-join, before and after `update()`.

Tracing every query of the loop found the first divergence: epoch 6 of the
d=32 run, one row of 10 000, **identical distances, different id**. The two
candidates are 5.471940390825072 and 5.47194052418801 away in f64 — the same
number once scored in f32. It is an exact tie for the last of five slots.
The old code returned whichever the heap happened to hold; this build applies
the canonical `(distance bits, id)` rule from `CONTRACT.md`, so it returns
the lower id. From one swapped neighbour the relaxation diverges within a
few epochs.

So the tie-break is a deliberate, documented improvement — results no longer
depend on visiting order, block size or thread count — but it does mean
**ESS trajectories at d ≥ 9 are not comparable against pre-`0bdbe50`
builds.** Quality is unaffected; paired over three seeds at d=32, 0+10k:

| seed | CE base | CE new | ΔCE | separation base | new |
|---|---|---|---|---|---|
| 0 | 1.2349 | 1.2326 | −0.0023 | 4.9046 | 5.0548 |
| 1 | 1.2287 | 1.2290 | +0.0003 | 4.9135 | 5.0351 |
| 2 | 1.2352 | 1.2382 | +0.0030 | 4.9988 | 5.1184 |

Mean ΔCE **+0.0003** (sd 0.0026), against a between-seed sd of 0.0037–0.0046
— unchanged, as intended. Toroidal-L1 separation is +0.13 on all three
seeds; consistent, but a side effect of tie-breaking, not something this work
set out to buy.

## 4b. Is the recall the index delivers good enough for ESS?

`examples/bench_ess_suite.py` reports recall per shape, and at d=32 it is
**0.69** (empty anchor tier) and **0.86** (with anchors) — not the ~1.0 the
smaller shapes get. Recall is a property of the index, not of ESS, so
`examples/bench_ess_quality.py` asks the question in ESS's own metrics
instead: does the result still disperse, and does buying more recall buy
any of it back? (`requirements.md` §7 records that re-exploration quality
tracks recall monotonically, which is what made this worth checking.)

**ESS beats the samplers at every shape** — toroidal Clark-Evans, where
1.0 is Poisson:

| shape | ESS | LHS | random | ESS min-dist | LHS min-dist |
|---|---|---|---|---|---|
| d=2, 0+256 | **2.083** | 1.033 | 1.018 | 0.0503 | 0.0048 |
| d=2, 256+512 | **1.698** | 1.001 | 1.017 | 0.0025 | 0.0009 |
| d=8, 0+1024 | **1.470** | 1.008 | 1.002 | 0.414 | 0.108 |
| d=8, 1024+2048 | **1.332** | 1.000 | 1.004 | 0.098 | 0.098 |
| d=32, 0+10k | **1.233** | 1.042 | 1.043 | 1.198 | 1.028 |
| d=32, 10k+20k | **1.169** | 1.036 | 1.036 | 0.929 | 0.876 |

The margin narrows with dimension (concentration of measure), but it is
never in doubt — and it holds at the shapes where recall is 0.69. Note
that with anchors the minimum pairwise distance saturates: the closest
pair lives in the fixed anchor set, which ESS cannot move, so Clark-Evans
is the metric that discriminates there, not min-dist.

**Perfect recall is worth ~1% of Clark-Evans.** Same shape, same seed, one
index forced to `num_tables=48, probes=8`; three seeds, paired:

| seed | default recall | default CE | high-recall CE | paired Δ |
|---|---|---|---|---|
| 0 | 0.689 | 1.2326 | 1.2484 (recall 1.000) | +0.0158 |
| 1 | 0.745 | 1.2290 | 1.2466 (recall 0.995) | +0.0176 |
| 2 | 0.753 | 1.2382 | 1.2521 (recall 0.995) | +0.0139 |

All three differences positive and tightly clustered — mean **+1.28%** — and
the paired spread (0.0037) is narrower than the between-seed spread of
either arm (0.0092), which is why the comparison has to be paired; an
unpaired two-seed look would have been inconclusive. On the anchored shape
the gain is +0.4% and min-dist does not move at all (anchor-limited).

So the tuner's d=32 choice is **defensible, not broken**: it gives up ~1%
of a ~18% margin over LHS, and full recall would cost `L*(1+probes)` rising
from 120 probes per query to 432 — roughly 3.6x the candidate work (a
structural count, not a timing). Raising recall at d=32 is a quality knob
with a known, small payoff, not a bug to fix.

## 5. What is left — and how well each one is evidenced

Not a uniform list. Every item below states what was actually measured, so
none of them is mistaken for a promise. All ablations come from **one shape**
(50k+50k, d=32, k=5, 16 threads) and run-to-run spread on this machine is
~10% (the same config measured 319/335/364/372 ms across the session), so the
percentages carry a few points of slack.

**Measured gain — the payoff itself was observed**

1. **Full bucket reuse — ~21%.** Blocks are query slices, so at 50k a bucket
   serves only ~3.8 queries' worth of work. Making the work item a
   *(table, key-range)* over all `m` queries, with a per-worker `m × k` top-k
   slab reduced at the end, would raise that to ~61. Measured at equal work,
   single-threaded, using block size as the reuse proxy: 61× reuse 3035 ms,
   3.8× 3835 ms, 1.9× 4557 ms. **Unmeasured:** the scheduler's own cost —
   slab reduction and 3 MiB/worker of L3 pressure — so 21% is an upper
   bound. Costs `threads × m × k × 12` bytes (48 MiB at m=50k, k=5; needs a
   budget guard and a fallback for large k).

**Measured cost, unmeasured payoff**

2. **Probe-key hashing — 11% of query** (ablation: 35.1 ms of 319.5 ms).
   Scalar f64 today. Vectorizing needs a gather per sampled dimension
   (`x[S[t][j]]`), and gathers are mediocre on this core, so the achievable
   share of that 11% is unknown — could be most of it, could be a quarter.
3. **Bucket-key sort — inside an 8% bucket.** The ablation measured *sort +
   group walk together* (25.8 ms); they were never separated, so the sort
   alone is ≤8% by an unknown margin. A counting sort over `B^K` bins is the
   obvious replacement for `sort_unstable` on `(key, slot)`.

**Not profiled — inferred from reading the code**

4. **`query_radius`.** No measurements at all: ESS's
   `search_mode="radius"` was never run. Structurally it still uses the
   per-query gather and allocates two `Vec`s per query, so it received none
   of this work — but that is an inference, not a number.
5. ~~**The small-`n` path**~~ — **done** (`1a41086`, see §4). Profiled on the
   shapes the suite gives `brute`; the cost was `diff.sum(-1)`, not the
   allocation `requirements.md` §5 blamed. Accumulating over `d` in NumPy's
   own pairwise order keeps it bit-identical up to `d = 128`. `setup` on the
   d=2, 256+512 shape fell from 39% to 17% of wall time. What remains there
   is `_smart_init`'s query itself — see §9.
6. **The tuner's own sample.** `_tune` builds a `(256, 8192, d)` difference
   block — ~0.5 GB of traffic at n=50k. Seen in the callgrind profile
   (NumPy ufuncs, ~12% of that process), bounded by ESS's `setup` share.

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

## 9. Where the remaining ESS-side time is — and what does *not* buy speed

Four follow-up questions, each answered by measurement rather than by
reading the code.

### `_smart_init` is index-bound, and the index is near its floor for k=1

Split into its three parts (`pool = 15`, so the query has `15n` rows):

| shape | mode | total | sampler | index query | selection |
|---|---|---|---|---|---|
| d=2, 256→512 | brute | 26.0 ms | 0.4 ms | **25.6 ms (98%)** | 0.0 ms |
| d=8, 1024→2048 | lsh | 40.2 ms | 5.6 ms | **34.2 ms (85%)** | 0.4 ms |
| d=32, 10k→20k | lsh | 2696 ms | 363 ms | **2319 ms (86%)** | 13.6 ms |

The NumPy around the query is already negligible, so the only lever is the
query. On the index side there is little left: `k=1` costs 95% of `k=5`
(2370 vs 2504 ms at d=32; 36.9 vs 44.5 ms at d=8), so the heap is not the
cost — scanning the candidate set is, and that is the same work `k=5` does.
A `k=1` specialisation is worth ≤5%. The batched join already moved this
path 3366 → 2370 ms (1.42×).

Amdahl says stop here anyway: `setup` is 4.5% of the d=32 10k+20k run and
20% of the d=2 256+512 one. The remaining lever is ESS-side and algorithmic
— `pool = 15` fixes the query at 15 rows per new point, and nothing requires
those rows to be answered at full recall, since only their *rank within the
pool of 15* is used. Untested.

### The tuner is not over-provisioning tables — quality degrades immediately

Tempting after §4b (recall 0.69 is enough): if more recall buys almost
nothing, buy less of it. Measured at d=32, 0+10k, single batch:

| tables | total | query | CE | separation (L1) |
|---|---|---|---|---|
| **24 (tuned)** | 14.08 s | 8.45 s | **1.2326** | 5.055 |
| 16 | 10.47 s | 5.20 s | 1.2049 (−2.2%) | 4.576 |
| 12 | 10.12 s | 4.28 s | 1.1797 (−4.3%) | 4.750 |
| 8 | 5.27 s | 1.64 s | 1.1426 (−7.3%) | 4.266 |
| 4 | 5.81 s | 1.20 s | 1.0806 (−12.3%) | 3.913 |

The operating point sits on a knee: **+1.28% CE** for 3.6× more probes going
up (§4b), **−4.3% for 2×** fewer going down. There is no free speed in the
parameters — the speed has to come from the implementation.

### Batching is a memory knob, not a speed knob — still true after the join

ESS defaults to `batch_size=None` (all points relaxed together) because
freezing batches is greedy. The join changed the query economics, so the
trade was re-measured; the verdict did not move. d=32, 0+10k:

| batch | total | peak RSS | CE | separation (L1) | fill |
|---|---|---|---|---|---|
| 10 000 (default) | **14.08 s** | 1583 MB | **1.2326** | 5.055 | 5.430 |
| 5 000 | 15.28 s | 981 MB (−38%) | 1.2260 (−0.5%) | 5.160 | 5.431 |
| 2 500 | 25.26 s | 509 MB (−68%) | 1.2252 | 5.304 | 5.428 |
| 1 250 | 32.35 s | 273 MB (−83%) | 1.2205 (−1.0%) | 5.403 | 5.487 |

Smaller batches are monotonically *slower* — the epoch count explodes
(120 → 903) because each batch re-runs the plateau detector, and each still
queries an index that keeps growing. d=8, 1024+2048 agrees: 0.91 s → 0.86 s
(inside run-to-run spread) for CE 1.3324 → 1.2963 (−2.7%).

So: **no, batching cannot be re-introduced for speed.** As a memory knob it
is genuinely useful and cheap — `batch_size = n/2` gives 38% less peak RSS
for 0.5% CE and 8% time. Worth noting the quality trade is not uniform:
batching *improves* worst-pair separation (5.055 → 5.403) while lowering the
mean-based CE, because each frozen batch is placed against a denser index.

### CE and coverage are not the right panel above d = 2

Scored point sets whose voids are known by construction (`metric_probe.py`):
`uniform`, `void` (a ball emptied and its points redrawn outside),
`clustered` (half the points in a quarter-width box), `lhs`, `ess`.

* **Grid coverage fails first.** At d=2 it separates the variants weakly
  (ess 0.844, uniform 0.645, clustered 0.422). At d=8 it saturates and
  *inverts*: lhs 0.988 > ess 0.981 = uniform 0.981. At d ≥ 8 with 2 cells
  per dimension the grid is 2^d cells, so past d≈20 it cannot be built at
  all — it is `nan` at d=32. It should not be in the panel above d=2.
* **CE works but is blind to voids.** It ranks correctly everywhere
  (d=2 ess 2.081 vs uniform 0.982; d=32 ess 1.256 vs uniform 1.050, and
  clustered 0.707 at both). But on the d=2 `void` set it moves −3%
  (0.982 → 0.952) where the fill distance moves **+46%** (0.119 → 0.173):
  as a mean over nearest-neighbour distances it cannot see a hole that few
  points border. `toroidal_clark_evans`'s own docstring flags the d>32
  limit; the void blindness is the sharper limitation, and it applies at
  every d.
* **Separation is the metric that survives to d=32.** ess 5.524 vs uniform
  3.935 / lhs 3.918 — a 40% margin, while fill distance is flat
  (5.708 vs 5.750, 0.7%). At d=32 with n=4000 no design can cover the
  torus, so ESS is a *packing* method there, not a covering one, and only
  the packing metric registers what it achieved. Report it — but not with
  `calculate_min_pairwise_distance`, which is **Euclidean and ignores the
  wrap**: on four points with a pair straddling the seam it reports 0.633
  where the toroidal L1 separation is 0.020. It measures a different
  geometry from the one ESS optimizes.

**Panel settled at two metrics, both toroidal L1: CE and separation.**
`examples/bench_ess_quality.py` reports only those. Fill distance and mesh
ratio stay documented in that file's `metrics` docstring as the right choice
at low d, but they do not hold at d=32, and a metric that only works
sometimes costs more than it explains.

Both needed fixing in `ess.utils` first (ESS branch `metric-calibration`):

* **CE was not calibrated above d≈8.** Its whole meaning is "1.0 = uniform",
  and uniform samples scored 1.033 at d=16, 1.050–1.082 at d=32 and
  1.103–1.136 at d=64 (5 seeds, sd 0.001–0.007 — systematic, not noise). The
  normaliser was the Poisson asymptotic
  `E[r] = Γ(1+1/d)(d!/n)^{1/d}/2`, which assumes `exp(-nV)` for the survival
  function and `V(t) = (2t)^d/d!` for the ball volume. At d=32 the mean
  nearest-neighbour distance is ≈4.9, so the ball has wrapped around every
  coordinate and neither premise holds. The exact null is available in closed
  enough form to just compute: a toroidal coordinate distance to a uniform
  point is `U(0, 1/2)`, so the distance is Irwin-Hall/2, and
  `E[R] = ∫ (1-V(t))^{n-1} dt` with `V` obtained by FFT-convolving the
  one-coordinate density d times. Uniform now scores **0.999–1.003 at every
  d from 4 to 64**. The old d=32 figures were inflated ~4%: ESS 1.2326 →
  **1.1819**, with LHS and random landing on 0.999/1.000 as they should.
  Every *comparison* in §4b is unaffected — they were all against measured
  baselines, and CE is a ratio — but the absolute "+23% over random" reading
  at d=32 was really +18%.
* **Separation was measured in the wrong geometry.**
  `calculate_min_pairwise_distance` is Euclidean and ignores the wrap, so it
  is not the quantity ESS minimises: 0.176 vs 0.311 at d=8, 1.06 vs 4.18 at
  d=32, and on a pair straddling the seam 0.633 vs 0.020. Added
  `toroidal_separation`, which takes it from the same `exact_knn` scan CE
  already pays for — so the two-metric panel costs one scan, not two.

Also checked rather than assumed: the ceiling `2/Γ(1+1/d)` = 2.257 in 2D is
real, but it needs the **diagonal** lattice (L1 balls are diamonds and
diamonds tile) — measured 2.2574. The axis-aligned 16×16 grid reaches only
1.597, so a grid is not the reference packing to compare against.
