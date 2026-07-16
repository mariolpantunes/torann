# torann analysis — v3 (phase 8: recall-preserving kernel pass + the ESS main loop)

Third edition. The phase-8 addendum below documents a staged,
conformance-gated optimization pass over the Rust core and the first
end-to-end ESS main-loop simulation; the numbered sections that follow it
are the phase-7 (v2) measurements, kept because their conclusions stand
(the v2 file is `git show 1fadefd:ANALYSIS.md`, the phase-6 bake-off
`git show 64672f2:ANALYSIS.md`; the C backend lives at tag
`archive/backend-c`).

Machine: AMD Ryzen AI 7 PRO 350, 16 threads. Python 3.12, NumPy 2.5,
Rust 1.97 (`-C target-cpu=native`, LTO), FAISS 1.14. Workload shape:
`k = 2d`, 3 000-query batches (the ESS inner loop). All implementations
produce **byte-identical hash tables** (CONTRACT.md), so recall columns are
shared and the comparison is pure speed.

## TL;DR (after phase 8)

* **ESS main loop, simulated end to end** (15k anchors + 10 batches × 3k
  candidates × 32 epochs → 45k points, d=16): **61.5 ms/epoch at recall
  0.983–0.995** on the torus. FAISS Flat, rebuilt every epoch, takes
  106 ms/epoch and returns **0.25–0.28 recall** — exact for the wrong
  metric. The pure-Python reference: 3 729 ms/epoch.
* The phase-8 kernel pass bought a further **1.10–1.44×** on torus queries
  and **1.23–1.64×** on box queries over the phase-7 code, with recall
  provably unchanged (byte-identical tables at every stage).
* Against **FAISS on common ground**, the gap to the exact SIMD Flat scan
  narrowed again: 1.5–2.4× (phase 7) → **1.06–1.56×**, at equal ≈ 1.0
  recall, with 20× faster builds at n = 1M and millisecond updates.
* Brute force loses to the Rust LSH **from n = 500 up** at recall 1.00;
  against the python LSH it keeps winning to n ≈ 4–8k — both
  `brute_threshold` defaults (512 / 4096) stand.

## Phase 8 addendum — the recall-preserving kernel pass (v3)

Ground rule: recall is the product, so only changes that provably keep the
candidate set and results identical qualify — every stage had to pass the
conformance suite (byte-identical tables vs the NumPy reference) before
being benchmarked. Each idea was measured in isolation, on two workloads:
**box** data `[0.25, 0.75]^d` (the FAISS common ground of §6 — its best
case and torann's worst: half-range data over-fills buckets ~7× and makes
refine dominate) and **torus** data `[0, 1)^d` (the native regime).

### Kept (the current code)

1. **`mod 1` without the fmod libcall.** The hash's `(x + u) mod 1`
   compiled to a scalar `fmod` call per (table, dim). For non-negative s,
   `s - s.floor()` is *bit-identical* to `np.mod(s, 1)` (floor cancels the
   leading significand bits exactly; the fractional part is always
   representable) and `floor` lowers to one `vroundsd` instruction. The
   facade's mod-1 reduction guarantees the precondition (now stated in
   CONTRACT.md; `debug_assert`ed in the kernel). Worth ~1.2× at d=32.
2. **Allocation-free key computation.** `compute_keys` collected one heap
   `Vec` per point (1M allocations at a 1M build) plus a serial transpose;
   it now streams through per-chunk flat buffers with one contiguous copy
   per table. Chunk size adapts (`n / 4·threads`, clamped [16, 1024]) so
   the 3k-candidate per-epoch update still fans out over all workers — a
   fixed chunk regressed updates 2–3× before the fix caught it.
3. **Direct-address bucket offsets** — the big one. Keys live in
   `[0, B^K)` and the tuner keeps `B^K ≈ n/64`, so a per-table cumulative
   u32 array `offs[v] = lower_bound(sorted_keys, v)` (~a few MB total,
   L2-resident) answers every bucket lookup and every relaxation range in
   O(1), replacing the ~120 × log₂(n) dependent-load binary-search chains
   per query. Rebuilt in O(n + B^K) per table at `build`/`promote`; the
   candidate tier keeps its (cache-resident) binary search so `update` is
   untouched; a 64 MB budget falls back to the batched search if
   hand-forced parameters blow up `B^K`. Cut gather 12–19 % CPU even where
   scans dominate, and most of the query-time wins below.

Per-query timings (µs/q, best of 3, 3 000 queries, k = 2d):

| n · d | torus before | torus after | × | box before | box after | × | gap to Flat before → after |
|:--|--:|--:|:--|--:|--:|:--|:--|
| 20k·16 | 12.3 | **11.1** | 1.11 | 34.9 | **28.4** | 1.23 | 1.35 → **1.10** |
| 100k·16 | 20.8 | **16.0** | 1.30 | 201.3 | **139.4** | 1.44 | 1.53 → **1.06** |
| 100k·32 | 36.5 | **29.3** | 1.25 | 322.3 | **196.1** | 1.64 | 2.28 → **1.39** |
| 500k·16 | 102.8 | **93.3** | 1.10 | 1 174.8 | **899.0** | 1.31 | 1.82 → **1.39** |
| 1M·16 | 162.9 | **148.0** | 1.10 | 2 823.1 | **2 297.4** | 1.23 | 1.92 → **1.56** |

Recall identical to the reference at every row by construction. Builds
dropped ~15–20 % (1M box: 1.35 → 1.07 s); updates are at or below the old
numbers everywhere (0.9–2.1 ms).

### Rejected by measurement (recorded so nobody re-tries them)

* **Gather-time candidate-row prefetch** — issue the refine row fetch as
  each id passes dedupe. Slower on *both* workloads (box 500k:
  885 → 1055 µs/q): with 10⁴–10⁵ candidate rows per query the prefetches
  evict each other long before refine arrives. Same lesson as the v2
  prefetch rejection, now confirmed from the other side.
* **Exact early-abandonment in refine** — stop a distance once its partial
  sum reaches the current k-th (provably identical results). No gain: an
  L1 half-way lower bound only rejects candidates ≥ 2× the k-th distance,
  and distance concentration means essentially none are. The checkpoint
  overhead showed up instead (d=32: +7 %).
* **16-wide f32 accumulators** — AVX-512 width in the distance kernel.
  Slower everywhere measured (100k·16 box: 139 → 160 µs/q): this Zen 5
  variant double-pumps 512-bit ops, so the wider reduction only adds
  latency.

### The ESS main loop, simulated (`examples/ess_sim.py`)

The full lifecycle: 15k anchors, then batches of 3k candidates that query
`k = 2d` every epoch for 32 epochs (moving σ = 0.01 between epochs) and
freeze into anchors, until 45k points — 320 epochs, 960k queries, d=16.
Recall spot-checked per batch against exact toroidal truth. FAISS Flat is
rebuilt every epoch (`add()` is a memcpy — the cheapest way FAISS can
track a moving tier).

| system | data | epoch ms | wall s | recall per batch |
|:--|:--|--:|--:|:--|
| **torann rust (phase 8)** | torus | **61.5** | **27.4** | **0.983–0.995** |
| torann rust (phase 7) | torus | 63.3 | 28.0 | same (byte-identical) |
| faiss Flat, rebuilt/epoch | torus | 106.2 | 41.3 | **0.25–0.28** |
| torann python | torus | 3 729 | — | 0.983 |
| faiss Flat, rebuilt/epoch | box | 94.3 | 30.7 | 1.000 |
| torann rust (phase 8) | box | 145.1 | 47.4 | 1.000 |

Read it straight: on the metric ESS actually needs, torann is **1.7×
faster than FAISS Flat and returns 3.7× more of the true neighbours** —
Flat's misses are structural (seam-blind L1) and do not decay as points
accumulate. The box rows show the flip side honestly: where nothing wraps
and buckets over-fill, Flat's contiguous scan wins the epoch at this small
scale — that regime is FAISS's best case and not the workload. The
phase-8 epoch gain over phase 7 is modest (63.3 → 61.5 ms) because at
18–45k points the epoch is refine-bound; the offset table's value grows
with n (see the query table above).

## 1. What the optimization pass bought (phase 7 D)

The native class carries a phase profiler (`stats()` / `reset_stats()`:
gather / refine / relax nanoseconds + candidate counters — py-spy cannot
see rayon worker threads, so the index measures itself). It attributed
query time ~50/50 to *gather* (bucket lookups) and *refine* (distances),
relaxation 0%. Three changes survived measurement:

1. **SIMD refine** — the scalar `sum += min(t, 1−t)` loop has a serial
   float dependency LLVM must not reorder, so it never vectorized. Eight
   parallel accumulators make it one AVX lane set. Biggest single win
   (~2× at d = 32).
2. **Batched gather** — the ~120 bucket lookups of one query became a
   single level-synchronous branchless `lower_bound` over all keys: every
   cache-miss chain is in flight at once (memory-level parallelism — the
   transferable part of FAISS Flat's speed). Bucket ends by forward scan
   instead of a second binary search.
3. **Candidate-row prefetch + u32 visit stamps** in the dedupe/refine path.

Rejected by measurement: explicit prefetch inside the batched search — the
load buffers are already saturated; it made things *slower*. Recorded so
nobody re-adds it.

Single-thread effect (µs/query, before → after):

| n · d | 20k·8 | 20k·16 | 100k·16 | 100k·32 |
|:--|--:|--:|--:|--:|
| before | 189 | 257 | 391 | 811 |
| after | 152 | 172 | 210 | 288 |
| speedup | 1.24× | 1.5× | 1.9× | 2.8× |

## 2. Speed grid — 16 threads (`examples/out/grid_16t_v2.json`)

| n | d | tuned B/K/L | python q µs | rust q µs | python upd ms | rust upd ms | recall |
|--:|--:|:--|--:|--:|--:|--:|--:|
| 20,000 | 8 | 3/6/24 | 1251 | **13** | 13.5 | 1.3 | 0.870 |
| 20,000 | 16 | 2/9/24 | 1619 | **15** | 18.0 | 1.9 | 0.893 |
| 20,000 | 32 | 2/8/24 | 3339 | **26** | 17.3 | 2.5 | 0.884 |
| 100,000 | 8 | 3/7/24 | 3175 | **33** | 15.0 | 1.4 | 0.848 |
| 100,000 | 16 | 2/12/24 | 1889 | **23** | 22.5 | 2.2 | 0.852 |
| 100,000 | 32 | 2/11/24 | 3140 | **36** | 21.8 | 2.8 | 0.686 |
| 500,000 | 8 | 3/9/24 | 9175 | **171** | 18.4 | 1.5 | 0.817 |
| 500,000 | 16 | 2/14/24 | 4294 | **107** | 25.4 | 2.5 | 0.829 |
| 500,000 | 32 | 2/13/24 | 5919 | **143** | 24.9 | 3.1 | 0.555 |

Query speedup vs python: **93–126× at 20k, 83–95× at 100k, 40–54× at
500k**; updates 7–12×. Single-threaded (`grid_1t_v2.json`): 150–1241 µs/q —
threading buys a further 6–11×. The d=32 recall rows are the tuner's L-cap
(the open phase-5 set-recall vs distance-ratio question), identical across
implementations.

## 3. The ESS lifecycle (`examples/lifecycle_backends.py`)

15k anchors + 3 batches × 3k candidates, d=16, k=32, 5 epochs/batch, σ=0.01:

| backend | fit s | query s | maint s | total s | epoch ms | recall/batch |
|:--|--:|--:|--:|--:|--:|:--|
| rust | 0.47 | 0.61 | 0.04 | 1.12 | **43** | 0.985 · 0.988 · 0.987 |
| python | 0.56 | 66.68 | 0.36 | 67.60 | 4463 | 0.985 · 0.988 · 0.987 |

**104× per epoch, 60× end-to-end.** For scale: the phase-6 winning backend
ran this epoch in 80–83 ms; the phase-7 kernels roughly halved it again.

## 4. Brute force vs LSH — crossover (`examples/out/crossover.json`)

Exact blocked-NumPy scan vs each LSH implementation at tuned defaults,
1 000 explicit queries, best of three. Selected rows (d=16):

| n | brute µs/q | python LSH (recall) | rust LSH (recall) |
|--:|--:|--:|--:|
| 500 | 87 | 142 (1.00) | **3** (1.00) |
| 4,000 | 694 | 676 (1.00) | **7** (1.00) |
| 8,000 | 1396 | 934 (1.00) | **10** (1.00) |
| 64,000 | 11247 | 1657 (0.96) | **16** (0.96) |

| crossover n\* (recall ≥ 0.95) | d=8 | d=16 | d=32 | q\* at n\* |
|:--|--:|--:|--:|:--|
| python | 8,000 | 4,000 | 4,000 | 1.7k–12.7k queries |
| rust | ≤ 500 | ≤ 500 | ≤ 500 | 130–567 queries |

The picture from v1 holds and sharpens: brute force is genuinely the right
algorithm below ~4–8k points *against the python LSH* — and essentially
never against the Rust one (already 29× faster at n=500, recall 1.00, with
the build amortized inside one ESS epoch). The backend-aware
`brute_threshold` defaults (python 4096, rust 512) stand confirmed.

## 5. Complexity (unchanged — constants moved, asymptotics didn't)

n points, m tier size, q queries, d dims, k neighbours, B/K/L/P hash
params, T target bucket load, C gathered candidates:

| operation | brute (NumPy) | LSH, general | LSH, tuned K = log_B(n/T) |
|:--|:--|:--|:--|
| fit | O(nd) copy | O(L·n·K + L·n log n) | — |
| query, per q | **O(nd)** | O(L(1+P) log n + C·d + C log k) | C ≈ L(1+P)T constant ⇒ **O(log n)** in n |
| update | O(md) overwrite | O(L·m·K + L(moved·log moved + m)) | churn ≈ 1−(1−B·E\|Δ\|)^K |
| promote | O(1) concat | O(L(n+m)) linear merge | never a re-sort |
| relaxation, per short q | — | O(L·K·log n) | guarantees k results |
| memory | O(nd) | O(nd + L·n) | — |

The measured 11 247 → 16 µs/q separation at d=16, n=64k is the tuned-K
column's log n vs brute's n·d, not a constant-factor trick.

## 6. FAISS on common ground (`examples/out/compare_faiss_flat.json`)

FAISS cannot answer the toroidal problem (0.26 recall at d=16 — phase 4.1).
Throughput comparison on data in `[0.25, 0.75]^d`, where no neighbour pair
wraps and toroidal L1 = plain L1 exactly; 16 threads for all:

| index | n | d | build s | µs/q | recall |
|:--|--:|--:|--:|--:|--:|
| faiss Flat L1 (exact) | 100k | 16 | 0.00 | 80 | 1.000 |
| faiss HNSW32 L1 | 100k | 16 | 0.66 | 3 | 0.815 |
| **torann [rust]** | 100k | 16 | 0.50 | 134 | 1.000 |
| faiss Flat L1 (exact) | 100k | 32 | 0.00 | 101 | 1.000 |
| faiss HNSW32 L1 | 100k | 32 | 1.11 | 6 | 0.436 |
| **torann [rust]** | 100k | 32 | 0.87 | 186 | 0.999 |
| faiss Flat L1 (exact) | 500k | 16 | 0.00 | 470 | 1.000 |
| **torann [rust]** | 500k | 16 | 0.80 | 724 | 1.000 |
| faiss Flat L1 (exact) | 1M | 16 | 0.02 | 996 | 1.000 |
| faiss HNSW32 L1 | 1M | 16 | 24.07 | 13 | 0.743 |
| **torann [rust]** | 1M | 16 | 1.18 | 2363 | 1.000 |
| faiss Flat L1 (exact) | 1M | 32 | 0.04 | 1330 | 1.000 |
| faiss HNSW32 L1 | 1M | 32 | 32.35 | 19 | 0.269 |
| **torann [rust]** | 1M | 32 | 1.89 | 2660 | 0.989 |

Read it straight:

* **The gap to exact SIMD Flat narrowed from 3–4× (phase 6) to 1.5–2.4×**,
  at equal (≈ 1.0) recall on this data. Flat still wins raw large-n
  throughput — a contiguous AVX scan is the best-case memory pattern, and
  torann's random gathers can approach but not beat it. At the ESS
  operating scale (n ≲ 20–50k) the two are at parity (§2: 13–26 µs/q).
* **HNSW remains disqualified**, not slow: recall 0.27–0.44 at d=32 for
  L1, 24–32 s builds at 1M, no incremental updates.
* **What FAISS cannot do is still the product**: the toroidal metric
  (0.26 recall — the reason this library exists), 1–2 s builds at 1M
  (20× faster than HNSW's), and 1–3 ms selective updates of a moving tier.

## 7. Verdict (v2)

The phase-6 recommendation (merge Rust) was executed and then validated by
this round: the borrow-checker-safe core took an aggressive optimization
pass — SIMD reshaping, batched searches, prefetching — with **zero memory
incidents and byte-identical tables throughout**, roughly doubling again
over the bake-off winner. torann now delivers FAISS-class throughput at
ESS scale on a metric FAISS cannot serve, with a pure-Python fallback that
is itself exact-conformant.

Remaining open item (phase 5, unchanged): the tuner's L-cap trades recall
at d=32 — whether ESS needs set-recall or distance-ratio there decides how
hard to push L.
