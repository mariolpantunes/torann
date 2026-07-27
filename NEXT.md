# State of play, settled questions, and what to try next

Written to be read *first* in a fresh session. §1–§3 are results that should
not be re-derived; §4 is the list of things measured and rejected, which is
the expensive knowledge here; §5 is the proposed per-dimension grid LSH,
assessed against those measurements; §6 is the recall question, now
**answered**; **§7 is the newest work and the one to read before planning** —
above `d ~ 8` the ESS force law stops discriminating between neighbours,
which changes what the index is being asked to do, and §7.4 is the brief for
the next session (a hash family designed natively for `p < 1`).

Detail behind every number: `OPTIMIZE.md`. Method rules: benchmarks must be
representative, optimization decisions come from profilers, and anything
else is speculation.

---

## 1. Where the code is

| | |
|---|---|
| branch `query-batched-join` (torann) | 8 commits, **not merged to main** |
| branch `metric-calibration` (ess) | 1 commit, **not merged** |
| tests | torann 60 pass; ess 48 pass |
| lint | `cargo fmt --check`, `cargo clippy --release -- -D warnings` clean |
| installed wheel | rebuilt from this branch — ESS imports the *installed* torann, so reinstall before any before/after claim |

**Read §7 before planning anything.** It is the newest work and it changes
what the index is being asked to do: above `d ~ 8` the ESS force law stops
discriminating between its neighbours, which is a different problem from
the ones §2–§5 were solving.

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
* `examples/bench_recall_ablation.py` — drives ESS with an exact index and
  corrupts the neighbour list before the force kernel; answers §6.
  `examples/report_recall_ablation.py` renders its JSON as a standalone
  HTML report.
* `examples/bench_refine_rounds.py` — **written, never run.** Repeated
  `esa` calls over an accumulating set; see §7.5.

Exploration (`exploration/`, all one-shot analyses behind §7):
`exp_cellsize.py` (grid feasibility + force weight by rank),
`exp_metric_contrast.py` (contrast and required force sharpness vs `p`),
`exp_lp_recall.py` (can the L1 hash retrieve `L^p` neighbours).

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

## 6. ANSWERED: recall has a floor, not a price

`examples/bench_recall_ablation.py`, commit `964cd9b`. The index is exact
throughout, so recall is *imposed*, not measured: the neighbour list is
corrupted between the query and the force kernel and the true toroidal-L1
distance of every substituted point is passed through, so only the
selection changes. 3 seeds, paired.

| arm | recall | ratio | CE d=32 | ΔCE | CE d=8 | ΔCE |
|---|---|---|---|---|---|---|
| exact | 1.000 | 1.00 | 1.2074 | — | 1.4741 | — |
| top2k | 0.500 | 1.00 | 1.2019 | −0.45% | 1.4183 | −3.78% |
| rank1-4k | 0.000 | 1.05 | 0.9993 | −17.2% | 0.8253 | −44.0% |
| rank8-16k | 0.000 | 1.16 | 0.9830 | −18.6% | 0.9440 | −36.0% |
| ratio2x | 0.001 | 1.45 | 1.0008 | −17.1% | 1.0083 | −31.6% |
| uniform | 0.001 | 1.45 | 1.0008 | −17.1% | 1.0042 | −31.9% |

**Half the true neighbours can go almost free; all of them cannot go at
all.** This is the "(2) holds, (3) collapses" branch: a distance *ranking*
is the product, the current design is the right shape, and redesigning
around cost per *plausible local neighbour* is ruled out — that arm is
indistinguishable from random. There is real slack in *how many* neighbours
are true, which is roughly where the tuner already sits.

Notes for whoever reads this next:

* §6's original arm 3 cannot work as specified. At d=32 `ratio2x` and
  `uniform` agree to four decimals on every column, because the ball of
  radius 2× the k-th distance has swallowed the whole point set. The
  rank-window arms replace it and pin recall at exactly zero while varying
  locality independently.
* "Worse than random" is a d=8 effect, not a general one. At d=8 a
  near-miss list scores CE 0.8253 against a null of 1.0042 — it holds
  points together while pushing on the wrong pairs. By d=32 every
  zero-recall arm simply lands on the null.
* **The filled start is uniformly *less* sensitive**, and it is the case
  ESS is actually for. 3 seeds, `recall_ablation_anchored.json`:

  | arm | d=32 empty | d=32 filled | d=8 empty | d=8 filled |
  |---|---|---|---|---|
  | top2k (recall 0.5) | −0.45% | **−0.29%** | −3.78% | **−2.51%** |
  | rank1-4k (recall 0) | −17.2% | **−10.4%** | −44.0% | **−28.3%** |

  So the slack measured from an empty start is a *lower bound* on the slack
  in production. Anchors also pin `separation` — at both filled shapes the
  `exact` and `top2k` arms score identically (4.1058 at d=32), because the
  minimum pairwise distance is set by the static tier, which no arm can
  move. Read separation as saturated for the good arms in the filled case;
  it still discriminates the collapsed ones (3.4952 for `rank1-4k`).

### 6a. The contradiction this exposes — the best speed lead on the list

Imposed recall 0.5 costs **0.45%** CE at d=32. But cutting LSH tables to
L=12 costs **4.3%** and L=8 costs **7.3%** (§4). Those cannot both describe
"less recall", so cutting tables must be doing something else, and finding
out what is the highest-value measurement outstanding:

* `_tune` sets `L = ceil(log(0.10) / log(1 - p1))`, clamped `[4, 24]`. That
  `0.10` is a hardcoded **90% collision target for a true k-NN** — a recall
  objective that was never derived from what ESS needs. Dropping the target
  to 50% gives L≈7 where the current model gives 22, i.e. **~3× less
  candidate work**, and query is 68–85% of ESS wall time.
* Two candidate explanations, both testable in ~20 minutes: (a) LSH misses
  are replaced by *far* points rather than omitted — the `rank8-16k`
  regime, −18%; or (b) `requirements.md` §7 already records that low recall
  *flattens the force EMA and fires the early-stop prematurely*, so the CE
  loss is a convergence artifact, not a force-quality one. The ablation's
  own epoch counts are consistent with (b): corrupted arms stopped at 40
  epochs against 151 for exact.
* **The measurement:** run ESS at L=24 and L=8 and record the *true rank*
  of every returned neighbour. If the L=8 substitutes sit at rank 6–20 the
  tuner is simply over-provisioned and the 3× is nearly free; if they sit
  at rank 500+, fix the fallback (prefix relaxation) and then take it.

What is missing is not a knob — `num_tables`, `resolution`,
`dims_per_table`, `target_bucket_size`, `probes`, `query_block_size` and
`brute_threshold` all override the tuner — but an **objective**: no way to
say "I need recall 0.5, not 0.9". Make the `0.10` a parameter.

## 6b. Superseded phrasing of the original question

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

---

## 7. The high-dimensional wall, restated: the *force law* stops working

This is the newest and most consequential section. Everything above treats
the index's job as "return the true k nearest". §7 is about the fact that
above `d ~ 8` the thing consuming those neighbours can no longer tell them
apart, which changes what "good enough" means.

Measured with `exploration/exp_cellsize.py` on ESS-converged points, using
the `gaussian_force` law `esa` actually defaults to:

| | rank 1 | rank 8 | rank 64 | share of vote from ranks 1–8 |
|---|---|---|---|---|
| d=2 | 0.159 | 0.021 | 3.7e-27 | **99.95%** |
| d=8 | 0.124 | 0.102 | 0.019 | 23.6% |
| d=32 | 0.349 | 0.330 | 0.299 | 13.3% |

At d=32 the 64th neighbour pushes **86% as hard as the nearest**. All 64
distances lie in [6.13, 6.31] — a 3% spread — so the Gaussian evaluates to
nearly the same weight for all of them. Consequences:

1. **Magnitude carries almost no information at high d; the entire signal
   is in *which* points are returned.** The force law cannot down-weight a
   wrongly-returned far neighbour — it votes at nearly full strength. This
   is the mechanism behind §6's collapse.
2. **Radius mode cannot work at d=32 either**, because "inside R" stops
   being a meaningful distinction. Related: at convergence **zero**
   neighbours lie inside R at every shape tested (the nearest sits at
   1.15–1.4·R). ESS pushes to equilibrium just outside the interaction
   radius, so a radius query sized at R returns empty sets.
3. **ESS destroys its own contrast as it converges.** Relative spread
   `(r64-r1)/r1` at d=32 is 24.2% on uniform random points but 2.9% on
   converged ones — making points equidistant is precisely its job. Any
   benchmark on random data (both FAISS scripts) measures an easier
   problem than the real one.

### 7.1 Grid cells sized from the force law: exact at d≤4, impossible at d≥8

The proposal (query's own cell plus the face-adjacent ones — von Neumann,
`1 + 2d` cells; the L1 ball makes diagonal cells pointless) needs two things
at once: **capture** (cell width `w >= R`, since `L∞ <= L1`) and
**selectivity** (occupancy `n / B^d` small). They cross between d=4 and d=8:

| d | n | R | cells/dim allowed (capture) | needed (occupancy) | verdict |
|---|---|---|---|---|---|
| 2 | 512 | 0.0442 | 22.6 | 10.1 | **works** — 5 cells, ~1 pt/cell |
| 4 | 2048 | 0.1956 | 5.1 | 4.5 | **works** — 9 cells |
| 8 | 2048 | 0.7355 | 1.4 | 2.1 | impossible |
| 32 | 10000 | 5.1095 | 0.20 | 1.27 | impossible |

At d≥16 `B_capture < 1`: the cell that would capture R is larger than the
whole domain. This is the same wall as §3 and §5, now in the force law's
own terms, and it is why §5's d≤4 ring engine remains the right scope.

### 7.2 Lower `p` restores contrast — worth about two octaves of dimension

`exploration/exp_metric_contrast.py`, contrast `(r64-r1)/r1` on
ESS-converged points:

| d | p=1.0 | p=0.5 | p=0.25 |
|---|---|---|---|
| 8 | 24.0% | 49.0% | 74.8% |
| 16 | 7.3% | 21.6% | 36.7% |
| **32** | **2.9%** | **11.5%** | **20.1%** |
| 64 | 1.8% | 6.4% | 11.5% |

p=0.25 at d=32 (20.1%) beats L1 at d=16 (7.3%) and approaches L1 at d=8
(24.0%). The required force sharpness relaxes with it (σ/r₁ 0.113 → 0.309),
so the two fixes reinforce rather than compete.

### 7.3 But the L1 hash cannot retrieve `L^p` neighbours — measured

The grid hash collides with probability `1 - B*delta` per sampled
dimension, so `log P(collide) ~ -B * sum_j delta_j`: **a function of L1 and
nothing else.** Two points with equal L1 and very different `L^0.5` are
retrieved with equal probability. Ranking candidates by `L^p` is therefore
a *rerank of a set that was never selected for it*. Measured against the
real LSH (d=32, n=4000, `exploration/exp_lp_recall.py`):

| L1 retrieval width | recall vs true `L^0.5` top-5 |
|---|---|
| k=5 (current) | 16.0% |
| k=50 | 74.3% |
| k=200 | 93.2% |
| k=500 | **93.2% — plateau** |

The plateau is not retrieval width, it is the hash's own L1 recall (0.689
here): the missing ~7% never collide, so no rerank reaches them. The
mitigating fact is that the `L^p`-nearest are not *far* in L1 — median L1
rank 18, 90th pct 55, max 147 — and the refine kernel already scores ~4118
candidates per query, so the rerank itself costs one `vsqrtps` per lane
plus a bigger heap, not extra retrieval.

**Do not ship the rerank as if it were an `L^p` index.** It is capped at
~93% and the cap is structural.

### 7.4 Q2 brief: a hash family designed natively for `p < 1`

This is the next session's task, to be done **in torann**. (Q1, the sharper
force law — `sigma` 0.5 → ~0.11 at d=32, no index change — is ESS's, not
torann's.)

The target: per-dimension collision probability `~ exp(-c * delta^p)`
instead of the current `1 - B*delta`.

* A fixed-width grid with random shift gives collision `max(0, 1 - delta/w)`
  — linear in delta, which sums to L1. That is not incidental; it is why
  the family reproduces L1.
* **Randomising the cell width** is the lever: `f(delta) = E_w[max(0, 1 -
  delta/w)]` can represent any convex decreasing `f` with `f(0)=1`, and
  `exp(-c*delta^p)` is convex for `p < 1`. This is presumably what
  `requirements.md` §7's "the grid-LSH family provably covers exactly
  `p in (0,1]`" means — the *family* covers it; torann instantiates the
  `p=1` member.
* **The obstacle is toroidal.** The wrap requires `w = 1/B` with *integer*
  B, so the mixing distribution is confined to `{1/2, 1/3, 1/4, ...}` with
  a hard ceiling at `w = 0.5`, rather than a continuum. Whether a discrete,
  capped mixture approximates the `p=0.5` kernel well enough is a theory
  question, not a measurement — settle it before writing code.
* Validation, once built: recall against true `L^p` k-NN must beat the
  93.2% rerank ceiling in §7.3, or the construction has bought nothing.

Numerics to watch: `(sum u^p)^(1/p)` reaches 1.4e5 at d=32, p=0.25, so the
force normalisation and log-sum-exp path need range checks, and ESS's
radius heuristic inverts an exact **L1** ball-volume formula that would
have to be re-derived per `p`.

### 7.5 Two things the benchmarks do not cover

* **ESS is a refinement method.** It is called repeatedly, each call handed
  the points it produced before, so the filled start is the *normal* case
  and the static tier grows without bound. Every benchmark in `examples/`
  measures a single `esa` call. Each call re-fits the index from scratch
  and re-tunes `(B,K,L)` on the new `n` — a full build over all accumulated
  points to add a fixed-size batch. `bench_refine_rounds.py` is written for
  exactly this and **has never been run**; run it on a quiet machine.
* **There is no FAISS comparison on the ESS workload.** `compare_faiss.py`
  uses uniform random data at `k=2*D=32`; `compare_faiss_flat.py` is a
  throughput reference in a non-toroidal box. Neither runs the ESS loop
  (`k=5`, every point querying every epoch, index mutated in place). The
  "61.5 vs 106 ms/epoch" figure comes from the tagged ESS's own FAISS path,
  not a controlled comparison. **Treat "competitive with FAISS on ESS" as
  unestablished.**

### 7.6 Measured and rejected this session

* **Hubness.** Ruled out as an explanation and as a target for
  hubness-corrected metrics (mutual proximity, local scaling). k-occurrence
  at k=5, n=2000: skew +0.32…+0.47 with max 13–15 against a mean of 5, for
  uniform *and* ESS-converged points, at d=8 and d=32, under p=1 and p=0.5.
  No runaway hubs anywhere.
* **A power-law force as the fix for §7.** Reaching 10:1 discrimination
  between rank 1 and rank 64 needs `r^-81` at d=32 (`r^-133` at d=64). The
  Gaussian's `r^2` in the exponent is what makes the same job reachable at
  `sigma/r1 = 0.113`, so `softened_inverse_force` is not the route.
