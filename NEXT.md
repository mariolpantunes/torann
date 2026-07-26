# State of play, settled questions, and what to try next

Written to be read *first* in a fresh session. §1–§3 are results that should
not be re-derived; §4 is the list of things measured and rejected, which is
the expensive knowledge here; §5 is the proposed per-dimension grid LSH,
assessed against those measurements; §6 is the open question worth a real
experiment.

Detail behind every number: `OPTIMIZE.md`. Method rules: benchmarks must be
representative, optimization decisions come from profilers, and anything
else is speculation.

---

## 1. Where the code is

| | |
|---|---|
| branch `query-batched-join` (torann) | 6 commits, **not merged to main** |
| branch `metric-calibration` (ess) | 1 commit, **not merged** |
| tests | torann 60 pass; ess 48 pass |
| lint | `cargo fmt --check`, `cargo clippy --release -- -D warnings` clean |
| installed wheel | rebuilt from this branch — ESS imports the *installed* torann, so reinstall before any before/after claim |

Benchmarks, in the order you normally want them:

* `examples/benchmark.py` — the standing sweep: dimensions × empty/filled
  start × k-NN/radius × brute/LSH, reported **per epoch**. Runs with or
  without `ess` (`--driver mimic` gives a self-contained loop).
* `examples/bench_ess_suite.py` — the six representative ESS shapes, A/B
  against the per-query path, with a trajectory fingerprint check.
* `examples/bench_ess_quality.py` — CE and separation against LHS and
  uniform baselines, plus the high-recall comparison.
* `examples/bench_lifecycle.py` — index maintenance strategies
  (selective update vs tier-sort vs refit).

## 2. What the optimization actually bought

ESS end to end, six shapes, defaults, against the pre-work build:
**119.0 s → 74.2 s (1.60×)**; query 99.2 s → 54.1 s (1.83×); query fell from
83% to 73% of wall time. Epoch-normalised — the honest figure, since epoch
count swings ±40% with the seed — **1.47× total, 1.59× query**.

Where it came from: the batched bucket-centric join (a rayon task per *block*
of queries, bucket-major, each bucket loaded once), a 4-row ILP distance
kernel using safe `wide::f32x8`, and dropping the `(m, n, d)` block from the
brute path (2.4–4.3×, bit-identical).

Quality is unchanged: paired over 3 seeds at d=32, ΔCE **+0.0003** against a
between-seed sd of 0.0037.

**One trap.** d=32 ESS trajectories do *not* reproduce against pre-`0bdbe50`
builds. The index is bit-identical across builds at every dimension; the
divergence is an exact f32 distance tie, now broken canonically by
`(distance, id)` instead of by heap order. One swapped neighbour at epoch 6
of a 10k run is enough. Do not chase this as a bug.

## 3. Geometry that constrains every design (measured, not argued)

This is the section that decides most proposals.

| d | 5-NN distance | best possible cell-wall bound | mean coordinate gap to a 5-NN |
|---|---|---|---|
| 32 | 4.94 | 0.25 max, 0.019 mean | 4.94/32 = **0.154** |
| 8 | 0.719 | 0.167 max, 0.027 mean | 0.090 |
| 4 | 0.162 | 0.071 max, 0.023 mean | 0.041 |
| 2 | 0.0154 | 0.062 max, 0.014 mean | 0.008 |

Read the right-hand column as: *a true near neighbour differs from the query
in almost every coordinate, by about that much.* At d=32 it is 0.154 per
coordinate — larger than a cell of any grid fine enough to be selective.
Only at **d=2** is the wall bound comparable to the neighbour distance, and
that is the one dimension where geometric pruning can work at all.

Other standing facts: cross-table duplicates are 6%, not 3×; the machine is
execution-bound, not bandwidth-bound (linear to 8 cores, SMT +24%, 1.1 D1
misses per pair); ~4118 candidates are scored per 5 kept.

## 4. Measured and rejected — do not re-try

**Index / query**

1. **Bucket pruning and partial-distance early-abandon.** The bound is
   20–200× too small at d ≥ 4 (§3). At d=32 the whole distance is ~20 packed
   instructions, so a checkpoint costs more than the dimensions it skips.
2. **Fewer LSH tables to buy speed.** d=32, 0+10k: L=24 → CE 1.2326;
   L=16 → 1.2049 (−2.2%); L=12 → 1.1797 (−4.3%); L=8 → 1.1426 (−7.3%). The
   tuner sits on a knee — +1.28% CE for 3.6× more probes going up, −4.3% for
   2× fewer going down. There is no free speed in the parameters.
3. **Instruction-count reduction in the kernel.** Made it *worse*
   (450 → 565 ms). The limit was dependency-chain latency, fixed by ILP
   (ROWS=4), not by fewer instructions. A cheaper threshold test and an
   inlined row copy were both neutral.
4. **`core::arch` intrinsics** — 320 ms vs 388 ms for safe `wide::f32x8`.
   The 21% is real and deliberately declined: no `unsafe` in a library.
   `std::simd` is nightly-only, so not an option for a published wheel.
5. **Keeping the query tile L2-resident** via small blocks. Reuse wins
   instead: 782 → 531 ms, 1563 → 445 ms, one-block-per-worker → 404 ms.

**ESS side**

6. **Batching for speed.** Monotonically slower — d=32 0+10k: 14.1 s at one
   batch, 15.3 s at n/2, 25.3 s at n/4, 32.4 s at n/8, because the epoch
   count explodes (120 → 903). It *is* a good memory knob: n/2 gives −38%
   peak RSS for −0.5% CE.
7. **Optimizing `_smart_init` further.** 85–98% of it is the index query,
   and `k=1` already costs 95% of `k=5`, so ≤5% is left. It is 4.5% of a
   d=32 run.

**Metrics**

8. **Grid coverage.** Saturates and inverts at d=8 (LHS 0.988 > ESS 0.981 =
   uniform 0.981) and cannot be built past d≈20 (2^d cells). Dropped.
9. **Fill distance / coverage radius above d≈8.** The metric that matches
   the name "empty space", and correct at low d (+46% on a set with a ball
   emptied out, where CE moves −3%) — but flat at d=32 (ESS 5.708 vs uniform
   5.750). Kept in the docs, out of the panel.
10. **The Poisson CE null.** Read 1.05 at d=32 and 1.14 at d=64 for uniform
    samples, i.e. the "1.0 = random" baseline was wrong by up to 14%. Fixed
    with the exact fixed-n null (Irwin-Hall halved, FFT convolution).
11. **`calculate_min_pairwise_distance`** is Euclidean and non-toroidal — a
    fragment of the tagged L2 version. Use `toroidal_separation`.

**Reference point:** the tagged ESS used FAISS, measured at 0.27 recall and
106 ms/epoch where this index did 61.5 ms/epoch at far higher recall. Any new
design has to beat *this* index, not that baseline.

## 5. Proposal: one hash per dimension, sparse hyper-grid, D-pad expansion

The idea: hash each dimension separately into a full grid of `B^d` cells,
look in the query's own cell, then open adjacent cells (±1 per axis)
recursively until `k` neighbours are found, or until the ring distance
exceeds the search radius.

Note first that the current design **already is** grid LSH — `code =
min(floor(B·frac(x+u)), B-1)` per sampled dimension, `K` dimensions per
table, `L` tables. The proposal is its `K = d, L = 1` limit, with ring
expansion replacing multi-probe.

### At d ≥ 16 the arithmetic says no, and it is the same arithmetic as §3

With `n = 10 000` at `d = 32`, `B = 2`: `4.3e9` cells, occupancy `2.3e-6`.
The query's own cell is empty almost surely, so expansion is not an
optimization, it is the whole algorithm. To reach an expected 5 points you
must visit `5 / 2.3e-6 ≈ 2.2e6` cells. Ring `r` holds roughly `(2d)^r / r!`
cells — 64, 2048, 43k, 690k, 8.8e6 for r = 1…5 — so you need **ring 5, of
order 10^6 hash lookups per query**, against 120 bucket lookups and ~4100
scored candidates today.

Worse, §3 says where the neighbours actually are: a 5-NN differs by ~0.154
per coordinate, so with any `B ≥ 8` it sits **more than one cell away in
nearly all 32 dimensions at once** — ring distance of order `d`, not order 1.
Ring expansion assumes near neighbours are near in *few* coordinates. In high
dimension they are near in *all* coordinates and far in the sum. That is
exactly why pruning failed, and it is not fixable by a different cell layout.

### At d ≤ 4 it is genuinely attractive — and could be *exact*

Everything above inverts at low dimension. Ring `r` in 2D holds `4r` cells,
occupancy can be tuned to a few points per cell, and the ring distance gives
a **correctness bound**: once the nearest `k` found are closer than
`(r-1)/B`, no further ring can improve them. That is exact k-NN with no
recall parameter — the classic grid/spiral search — and §3 shows d=2 is the
one dimension where the bound is in range (0.062 wall vs 0.0154 5-NN).

This matters for ESS because the d=2 shapes are served by **brute force**
today (below the 512-point crossover, and `brute` is 92% of that shape's wall
time). A grid search with an exactness bound would compete with brute force
there, not with LSH.

**So the experiment worth running is narrow and cheap:** implement ring-search
for `d ≤ 4` only, as a third engine alongside brute and LSH, and put it
against brute on the d=2 shapes of `examples/benchmark.py`. Success criterion:
faster than brute at n = 256…4096 while returning *identical* results (it
should — it is exact). Do not build the d=32 path; §3 already answers it.

## 6. The open question: does ESS need recall, or only plausible repellers?

The interesting hypothesis in this session, and still untested: the force
kernel needs `k` neighbours that push the candidate in roughly the right
direction, not the true `k` nearest. If so, recall is the wrong objective and
a much cheaper index would do.

What is known: recall 0.69 at d=32 costs only 1.28% CE against perfect
recall — consistent with the hypothesis. But dropping tables to *lower*
recall costs 4.3% CE at L=12 — which looks inconsistent, until you notice
that fewer tables degrades recall *and* locality together, so the experiment
confounds them.

**The clean test** separates the two. Drive ESS with an exact index, then
corrupt the neighbour list in controlled ways before it reaches the force
kernel:

1. exact `k`-NN (control),
2. `k` points sampled uniformly from the true `2k` nearest — same locality,
   worse recall,
3. `k` points sampled from everything within `2×` the k-th distance —
   plausible directions, recall ~0,
4. `k` uniformly random points — the null; CE should collapse.

If (2) and (3) hold CE within noise of (1), recall is not the objective and
the index can be redesigned around cost per *plausible local neighbour*. If
(3) collapses but (2) holds, what matters is a distance *ranking*, and the
current design is already the right shape. Either answer redirects the work,
and it costs one afternoon with no index changes at all.
