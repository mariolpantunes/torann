# torann: query-path optimization — requirements for a fresh session

Goal: make `ToroidalNN.query` substantially faster on the ESS workload
without changing a single returned result. Everything below is measured,
not assumed; where a plausible idea was tested and rejected, that is
recorded so it is not re-litigated.

---

## 1. Why this work exists

`ess` (branch `torann-backend`) spends **81–82% of its wall time in
`query`** on the re-exploration workload, measured with the timing sink
added in `ess` commit `92bcfa5`:

| d | anchors | new | total | query | force | step | update | setup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 10 000 | 10 000 | 32.7 s | **80.9%** | 6.5% | 2.1% | 3.0% | 4.7% |
| 32 | 50 000 | 50 000 | 151.2 s | **82.3%** | 8.1% | 1.1% | 1.8% | 5.9% |
| 32 | 500 | 200 | 0.8 s | 12.0% | 7.8% | 2.4% | 3.4% | **73.5%** |

Two separate problems. Query dominates at scale; at small `n` the cost
is `ess._smart_init`, which is a *torann* problem too — see §5.

Reproduce with `python examples/profile_torann.py` (index in isolation:
fit / query / update / recall) and the `stats` sink of `ess.esa`.

## 2. The finding to act on

`src/lib.rs:631`:

```rust
let _ = block;  // advisory; rayon schedules per query
```

`query_block_size` is accepted and discarded. Rayon parallelises **one
task per query** (`out_idx.par_chunks_mut(k)`), and each query calls
`gather_query` to walk its own `L × (1 + probes)` buckets, then
`refine_topk` over its private candidate list.

Consequence: a bucket probed by 50 queries is read from memory 50 times,
and a point appearing in several tables is distance-scored several times.
With the tuner's typical `L = 16–24` and `probes = 4–8`, that is
144+ bucket visits per query at ~32 points each.

## 3. Proposed change: bucket-centric batched join

1. Hash all queries in one vectorised pass per table.
2. Take a block of queries (this is what `query_block_size` was for) and
   group them by bucket — a counting sort on keys.
3. For each bucket the block touches, load its points **once** and score
   them against every query in the block that probes it. The inner
   operation is a dense `queries × points × d` tile: sequential in both
   operands, SIMD-friendly.
4. Keep per-query top-k in the block's own slab.

Parallelise over **query blocks**, so each thread owns its top-k
exclusively — no contention, no per-thread `n × k` allocation.

Two optimizations compose naturally once the loop is shaped this way:

- **Cross-table dedup** becomes structural: each bucket is visited once
  per block rather than once per query.
- **Partial-distance pruning.** L1 is a sum of non-negative terms, so a
  candidate can be abandoned mid-accumulation once the partial sum
  exceeds the current k-th best. With `k = 5` the threshold tightens
  fast; most candidates should die after a few dimensions.

This suits the ESS access pattern specifically: the default `query()` is
a **self-join** (the queries *are* the indexed candidates), so spatially
close queries are bucket-mates and blocks share buckets heavily.

## 4. Measure before rewriting

**Instrument candidates-scored-per-query** (and the duplicate fraction
across tables). If the ratio of candidates scored to the 5 kept is in the
hundreds, dedup and pruning are the whole prize and should be done first
— possibly without the full restructure. Size the prize before paying
for it.

Native profiling recipe (established, reuse verbatim): valgrind cannot
emulate Zen 5 `target-cpu=native` (SIGILL), so build an AVX2
`x86-64-v3` proxy with `CARGO_PROFILE_RELEASE_DEBUG=true` into a separate
`CARGO_TARGET_DIR`, swap the `.so`, run callgrind, restore and confirm
byte-identical with `cmp`. py-spy is useless here — rayon workers are
native-only and the main thread blocks in a futex.

Last full profile: `dist_l1_32` ~35% of instructions, `refine_topk`
bookkeeping ~30% (slice bounds checks ~8%, memset 3.3%), `gather_query`
~16% (memory-bound, so its wall share is higher than its instruction
share), `compute_keys` ~2%.

## 5. Second target: the small-`n` path

`ess._smart_init` is 38–74% of small runs and is almost entirely one
`torann.brute.pairwise_l1` call, which materialises an `(m, n, d)`
tensor — 3000 × 500 × 32 float64 is ~384 MB of traffic for what should
be a streaming min-reduction.

**Important:** a Rust brute kernel was rejected earlier, but that
verdict was about **many small calls**, where PyO3 overhead dominated.
Smart-init is **one large call**, where that overhead amortises to
nothing. Different case — re-test rather than inheriting the verdict.
Cheaper first step: chunk and fuse the NumPy reduction.

## 6. Hard constraints

- **Exactness is a contract.** Brute mode must return exactly what brute
  force returns; LSH mode must still guarantee `k` results via prefix
  relaxation. Verify new results are **bit-identical** to the current
  implementation across both backends before landing.
- Both backends stay working (`available_backends()` → rust, python).
- 60 existing tests stay green; `cargo fmt --check` and
  `cargo clippy --release -- -D warnings` stay clean.
- Commits: author `Mário Antunes <mario.antunes@ua.pt>`, never add Claude
  as author or co-author.

## 7. Settled — do not revisit

- **BLAS / OpenBLAS in the Rust core.** kNN only benefits via GEMM when
  the metric factors bilinearly (L2/IP). Toroidal L1 has no such
  decomposition; FAISS itself hand-scans `METRIC_L1`.
- **L^p (p<1) hash instead of L1.** The grid-LSH family provably covers
  exactly `p ∈ (0, 1]`, but L1 was chosen deliberately and stays.
- **Cached / stale neighbour lists.** Deprioritised by the user in favour
  of making the index genuinely fast. (Note for the record: it is *not*
  equivalent to taking bigger steps — caching preserves the number of
  force evaluations, and bigger steps were measured worse: `STEP_CAP`
  0.02 → 0.25 cost toroidal Clark-Evans 1.299 → 1.142 at d=32.)
- **Lowering recall to buy speed.** Measured: re-exploration quality
  tracks recall monotonically with no free plateau (void mean 5.93 →
  5.23 as recall falls 1.00 → 0.34), and low recall *also* triggers
  premature convergence — missing neighbours flatten the force EMA, so
  the early-stop fires while the relaxation is merely blind.

## 8. Open question inherited from this session

An apparent "free win" (`L=16, probes=8` at recall 1.000 costing 47
ms/epoch versus `L=24, probes=8` at 59) **did not survive scrutiny**: it
was measured against `probes=8`, which is not the default. Against the
actual default (`L=24, probes=4`, 45 ms/epoch) it is a wash. Single-seed
data; the mean-void metric was tight across equal-recall configs
(5.924–5.928) but worst-case void was noisy (4.90–5.55), so any tuner
change needs multiple seeds before being believed. **Not applied.**

## 9. Success criteria

- Query wall time at `d=32, 50 000 + 50 000` materially below the
  current 82% share, with end-to-end `ess` time to match.
- Results bit-identical; tests and lints green.
- `examples/profile_torann.py` in `ess` re-run before/after, and the
  `ess` timing decomposition re-measured at the same points.
