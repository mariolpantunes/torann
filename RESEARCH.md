# Past the `d ~ 8` wall: the research programme

Read `NEXT.md` first — §3 (measured geometry), §4 (measured and rejected)
and §7 (the force-law collapse) are the standing facts this programme is
built on and must not be re-derived. This document is the plan that follows
from them.

**The output is a paper, targeting GECCO 2027, and its subject is OBLESA.**
That fixes the shape of everything below. The stack is
`pyBlindOpt/OBLESA -> ess -> torann`: OBLESA combines opposition-based
learning with ESS expansion to jump-start a population-based optimizer, ESS
does the empty-space filling, and torann is the toroidal index that makes
ESS tractable. For an evolutionary-computation audience the claim has to be
about **optimizer performance**, so ESS and torann are machinery — necessary,
and interesting in their own right, but a section rather than the spine.
§1's `L^p` family is machinery for the machinery. Phase order in §2 reflects
that: the end-to-end OBLESA evaluation is on the critical path and every
index result is subordinate to it.

Method rules are unchanged and non-negotiable: benchmarks must be
representative of the ESS workload, optimization decisions come from
profilers, hypotheses are settled by measurement, and anything else is
speculation.

---

## 0. The problem, stated once

torann indexes points on the unit torus under toroidal L1 and serves ESS,
which serves OBLESA in `pyBlindOpt`. Above `d ~ 8` two things fail together:

* **The metric stops separating.** On ESS-converged points at d=32 the 64
  nearest neighbours lie within a 3% spread in L1 (`NEXT.md` §7).
* **The force law therefore stops discriminating.** Rank 64 pushes 86% as
  hard as rank 1, so a wrongly-returned far neighbour votes at nearly full
  strength, and magnitude carries almost no information — the entire signal
  is *which* points come back.

ESS makes this worse by construction: relative spread at d=32 is 24.2% on
uniform random points but 2.9% on converged ones. Making points equidistant
is precisely its job, so **any benchmark on random data measures an easier
problem than the real one.**

Lower `p` is the measured fix for the metric half (§7.2: `p=0.25` at d=32
beats L1 at d=16), and §7.3 measures that the current L1 hash *cannot*
deliver it — its collision probability is a function of L1 and nothing else,
capping any `L^p` rerank at 93.2% recall, structurally.

---

## 1. Settled this session: a toroidal `L^p` family, exactly

This supersedes `NEXT.md` §7.4, which posed the construction as an open
theory question ("whether a discrete, capped mixture approximates the p=0.5
kernel well enough is a theory question, not a measurement — settle it
before writing code"). It is now settled, and the answer is better than the
question assumed.

### 1.1 The projection route is closed on the torus

The textbook answer for `L^p`, `0 < p <= 1` is Datar–Immorlica–Indyk–
Mirrokni (SoCG'04), the first provably efficient ANN scheme for `p < 1`:
draw `a` from a `p`-stable law, hash `floor((a.x + b)/w)`.

**It cannot be used here, for a structural reason rather than a numerical
one.** The continuous homomorphisms `T^d -> T^1` are exactly the characters
`x -> sum_j n_j x_j mod 1` with *integer* `n_j`. A real-valued `a` has no
toroidal meaning; an integer-valued one wraps repeatedly and destroys
locality. torann's collision law is exactly `max(0, 1 - B*delta)` with no
boundary defect **because** it never projects. Seamlessness and projection
are mutually exclusive on the torus.

### 1.2 The stable law belongs in the cell rate

For `p in (0, 1]`, `exp(-c*delta^p)` is completely monotone, hence a Laplace
mixture:

```
E_s[ exp(-s*delta) ] = exp(-c*delta^p)     for   s = c^(1/p) * S_p
```

with `S_p` the one-sided `p`-stable subordinator — sampled by the same
Chambers–Mallows–Stuck method the projection route would have used. And
`exp(-s*delta)` is exactly the void probability of a rate-`s` Poisson
process on a circle. So:

> **Per `(table, dimension)`: draw `s ~ c^(1/p) * S_p`, scatter `Poisson(s)`
> breakpoints on `[0,1)`, hash `x_j` to the index of the arc containing it.**

Per-coordinate collision is `exp(-c*delta_j^p)` **exactly**, seamlessly
toroidal, with no integer-`B` constraint anywhere. Concatenating `K`
dimensions gives `exp(-c * sum_j delta_j^p) = exp(-c * ||delta||_p^p)`: an
exact `L^p` LSH family on the torus. The `p`-stable law is still the answer
to "which distribution do we draw" — it just goes into the grid rate, not
into a direction.

### 1.3 What it costs, measured

`exploration/exp_lp_family.py`, calibrated from `NEXT.md` §3 so a true 5-NN
collides with probability `q0` in a typical coordinate:

| d | p | c | rate cap | max kernel err | E[cells/dim] | P(drop) | eff. dims |
|---|---|---|---|---|---|---|---|
| 32 | 1.0 | 2.410 | any | 0.00000 | 2.41 | 0.090 | 0.910 |
| 32 | 0.5 | 0.946 | none | 0.00022 | 6.4e5 | 0.389 | 0.611 |
| 32 | 0.5 | 0.946 | 200 | **0.01499** | 14.65 | 0.389 | 0.611 |
| 32 | 0.5 | 0.946 | 50 | 0.03203 | 7.11 | 0.389 | 0.611 |
| 32 | 0.25 | 0.592 | none | 0.00025 | 1.4e17 | 0.553 | 0.447 |
| 32 | 0.25 | 0.592 | 200 | **0.05974** | 31.65 | 0.553 | 0.447 |
| 8 | 0.5 | 0.744 | 200 | 0.01181 | 11.59 | 0.476 | 0.524 |
| 8 | 0.25 | 0.407 | 200 | 0.04169 | 22.30 | 0.665 | 0.335 |

Four consequences that shape the build:

1. **Uncapped, the family is exact** (2e-4 is Monte-Carlo noise at 2e6
   samples). Every approximation in the design lives in the rate cap.
2. **Capping is mandatory, not a tuning choice.** `E[S_p] = inf` for
   `p < 1`, so the expected cell count per dimension is infinite —
   `1.4e17` at `p=0.25` uncapped. A cap at 200 gives a workable 15–32
   cells/dim.
3. **`p=1` is a degenerate member** (`S_1 = 1` deterministically): exact at
   any cap, `E[cells] = 2.41`. Note this is *not* today's index — it is a
   Poisson grid where today's is a uniform grid. The two share a collision
   law only in the small-`delta` limit.
4. **`P(drop) = exp(-c)` in closed form.** Dimension subsampling stops
   being a separate mechanism; it falls out of the family, and it prices
   `p` directly. At `p=0.25`, d=32 only 45% of sampled dimensions
   discriminate, so a table needs ~2.2× as many for equal selectivity.

**Verdict on the target.** `p=0.5` is comfortable (1.5% kernel error at a
usable cap, 61% effective dims). `p=0.25` is reachable but pays twice (6%
error, 45% effective dims) — build the family parameterised over `p`,
validate at 0.5, and treat 0.25 as a measured data point rather than a
target. `p(d)` decreasing with dimension is then a tuner rule, not a
redesign, since `p` and `c` are per-table scalars fixed at `fit()`.

### 1.4 The uniform-grid control

`exploration/exp_lp_gridfit.py` fits the same kernel with **uniform** cells
only — the torus-legal widths `1/B` plus a drop atom, by NNLS. It reaches
1.8% at `p=0.5` and 7.9% at `p=0.25` (d=32). Kept because the uniform hash
is one `floor()` per dimension where the breakpoint hash is a
`searchsorted`, so if the gap is small the cheaper kernel may win on speed —
and because it is the honest control for any claim that the Poisson
construction was *necessary* rather than merely elegant.

---

## 2. Phase order

Ordered by what unblocks what, not by appeal.

### P0 — Literature

Partly done. Two corpora matter and they are very different.

**The EC corpus — and the gap OBLESA sits in.** This is the one that decides
the paper, and it is more favourable than expected. Kazimipour, Li & Qin,
*A Review of Population Initialization Techniques for Evolutionary
Algorithms* (CEC 2014, 358 citations) closes with open questions that name
this work almost exactly:

* "Nearly all previous studies have been done on low dimensional single
  objective problems (less than 60 dimensions)."
* **"There has been little agreement on validation of those findings in
  high dimensional spaces. For example, [one] claimed that the desirable
  effect of uniformity of initial population is more significant in high
  dimensions (up to 50 dimensions) while [another], in contrast, claimed
  that uniform initialization techniques (e.g., QRS) lose their
  effectiveness in problems of 12 or more dimensions."**
* "Most comparison studies on population initialization are limited to a
  few (mostly less than four) techniques."

The disagreement is live: Tharwat & Schenck report *no* significant
difference among five initialization methods; Agushaka et al. find an
algorithm-dependent effect over 11 methods × 10 metaheuristics; other work
reports Sobol/Halton as *particularly* effective in high dimension.

**That contradiction is the paper's opening, and this stack already holds a
mechanism for it.** The EC literature disputes whether space-filling
initialization keeps helping above `d ~ 12`. `NEXT.md` independently
measured, from the geometry side, that the *metrics which define* space
filling stop discriminating in the same place: grid coverage saturates and
inverts at d=8 (§4.8), fill distance goes flat at d=32 (§4.9), and §7 shows
distance concentration collapses the contrast that "uniform coverage" is
defined by. The reason the field disagrees about uniformity above ~12
dimensions may simply be that it is measuring uniformity with instruments
that stop working there. A paper that (a) demonstrates the instrument
failure, (b) supplies `L^p` and a metric panel that survive it, and (c)
shows the effect on real optimizer performance, is a coherent GECCO
contribution — and notably one where the *negative* results are still
publishable.

Direct baselines to beat, not merely cite: Rahnamayan, Tizhoosh & Salama,
*A novel population initialization method for accelerating evolutionary
algorithms* (OBL initialization — the "OBL" half of OBLESA), and
Kazimipour, Li & Qin, *Effects of Population Initialization on Differential
Evolution for Large Scale Optimization* (CEC 2014), which supplies the
methodological template and one finding to design around: **initialization
matters more at smaller population sizes.**

**The ANN corpus.** Incomplete — Semantic Scholar rate-limited throughout
and arXiv does not index it (Datar SoCG'04, Gan SIGMOD'12, Aggarwal ICDT'01
are DB/theory conference papers). Established:

* **Datar et al. (SoCG'04)** — canonical `p`-stable LSH, explicitly covers
  `p < 1`. The reference §1.1's argument must engage with.
* **C2LSH, Gan et al. (SIGMOD'12, 330+ citations)** — *collision counting*
  with dynamic compound hash functions and virtual rehashing. Overlaps P3
  below; read before claiming novelty there.

Still to check, each able to invalidate a claim: whether §1.2's
rate-mixture construction is known (the completely-monotone step and grid
LSH are both textbook; the *toroidal* combination may not be); Chierichetti
& Kumar on which similarity functions are LSHable, which bounds what
non-distance scores can achieve; Aggarwal, Hinneburg & Keim (ICDT'01) on
fractional norms, the standing citation for §7.2; and any prior ANN work
under periodic boundary conditions.

### P1 — OBLESA end to end: the experiment the paper is made of

**Missing from every plan so far, and it is the spine.** Everything measured
to date — CE, toroidal separation, recall, ms/epoch — is an *intrinsic*
metric of the sampler or the index. Not one measurement in either repo shows
that an ESS-initialized optimizer finds better optima. For GECCO that is the
only claim that counts, and it is the one that can still fail.

Design, following the Kazimipour CEC'14 template so the result is
comparable to the literature it argues with:

* **Optimizers**: several from `pyBlindOpt` (DE, PSO, GWO at minimum) —
  the survey's own criticism is that comparisons use too few, and the
  reported effect is algorithm-dependent, so a single optimizer proves
  nothing.
* **Initializers**: random, LHS, Sobol, OBL (Rahnamayan), ESS alone, and
  OBLESA. Six arms, against a literature where "most comparison studies
  are limited to fewer than four".
* **Problems**: a standard suite (CEC'13 LSGO and/or CEC'17), swept over
  dimension **through the `d ~ 8-12` region where both literatures
  disagree** — that sweep is the paper's central figure.
* **Population size** as an explicit factor: CEC'14 found initialization
  matters more when the population is small, and OBLESA's cost is paid per
  initial individual.
* **Budget**: fixed function evaluations, *and* separately fixed wall time —
  ESS is not free, and a GECCO reviewer will ask what the initialization
  cost bought. This is where torann's speed work becomes a result rather
  than an appendix.
* **Statistics**: >= 30 runs, Wilcoxon signed-rank pairwise plus
  Friedman/Nemenyi across arms. Non-negotiable given that the field's
  existing answers contradict each other.

> **This gate outranks all the others.** If OBLESA does not beat OBL and
> Sobol on optimizer performance, then no amount of `L^p` index work
> rescues the paper, and the honest paper becomes "why space-filling
> initialization stops paying above `d ~ 8`, measured three ways" — which
> §4.8, §4.9 and §7 already largely support, and which the literature
> explicitly asks for.

Run this early and run it dirty first: current defaults, current L1 index,
one optimizer, to find out whether the effect exists at all before
investing in the full design.

#### 2.1 Three methodology rules, each learned by getting it wrong

`pyBlindOpt/examples/bench_init_oblesa.py` (branch `init-benchmark`).
These are recorded because the first two versions of this benchmark
produced confident, publishable-looking tables that were pure noise.

1. **The same seed must hit the same code path.** The arms consume very
   different amounts of randomness while initializing — ESS draws for every
   particle of every epoch, plain sampling draws once. Passing the
   optimizer whatever generator state the initializer left behind gives
   each arm a different DE trajectory, so two things vary per seed and both
   get attributed to the initializer. Seed the optimizer's generator
   *identically across arms*. Re-running six cells with this fix **flipped
   five of the six winners.**
2. **Compare convergence curves at matched evaluation counts, not final
   scores.** The initializer acts on generation zero and DE washes it out
   long before the budget ends, so an endpoint is mostly optimizer
   variance. Record best-so-far per generation and read every arm at 10 /
   25 / 50 / 100% of budget, accounting for the fact that arms which paid
   more to initialize reach a given budget later.
3. **Sanity-check the invariant the methods themselves guarantee.** OBLESA
   selects greedily from OBL's candidate pool *plus* the ESS batch, so its
   chosen population must be pointwise no worse than OBL's at every rank.
   It is — verified on every seed. Any result contradicting a structural
   invariant is a bug in the harness, and that check is what located both
   defects above.

The dispersion is the reason all three matter: a single cell (rastrigin,
d=32, n_pop=512) ranged 24.9–186.9 across eight seeds. At 8–20 seeds the
winner changed at every budget checkpoint. **Plan for >= 100 seeds at
small `n_pop`**, and treat any table without them as a smoke test.

### P2 — Index baseline and the metric gate, run concurrently

**P2a (torann): does the index scale the way we think?** Nothing in
`OPTIMIZE.md` measures scaling in `n` — every benchmark is at fixed sizes,
so both standing complexity claims are unverified. `bench_scaling.py`:
`n` in {1k … 1M} × `d` in {4, 8, 32} × {brute, LSH}; report single-query
latency, self-join wall time, and self-join per query as log-log slopes with
CIs, with the tuned `(B, K, L)` alongside. Criteria fixed in advance:
single-query slope <= 0.2, self-join slope < 1.15. If `L(n)` is not flat,
that is the finding — `_tune` sets `L` from a recall target, so a growing
`L` would silently add a factor to a claimed `O(log n)`.

Also here: property tests on prefix relaxation (least-covered code, most
invariants), cross-backend equality at sizes the current suite never
reaches, and `bench_refine_rounds.py` — written, never run, and the only
benchmark that measures ESS's *actual* repeated-call mode.

**P2b (ess): the force-law gate.** §7.2's chain has an untested link:
contrast is a property of the metric, not the index. A perfect `L^p` index
only pays if the force law consumes `L^p` distances. Brute path only, no
index changes, d in {8, 16, 32}, 3 seeds, paired, both starts: swap the
force kernel's distance to `L^0.5` / `L^0.25` with `_l1_radius_heuristic`
re-derived per `p`. Run as a 2×2 against `sigma in {0.5, tuned}` — §7.2
shows required sharpness relaxes with `p`, so without the second factor
"lower `p` helps" and "sharper `sigma` helps" are inseparable, and the
`sigma` fix is far cheaper.

> **Decision rule.** If `L^0.5` at tuned `sigma` does not beat L1 at tuned
> `sigma` by more than the between-seed sd (0.0037 CE at d=32), P4 is
> descoped from "make ESS better" to "make a better index", and the paper's
> claim narrows accordingly.

**P2c: the `0.10` collision target.** `_tune` sets
`L = ceil(log(0.10)/log(1-p1))` — a hardcoded 90% collision target for a
true k-NN, never derived from what ESS needs, while §6 measured that imposed
recall 0.5 costs 0.45% CE. Dropping to 50% gives `L≈7` where the model gives
22: ~3× less candidate work on a path that is 68–85% of ESS wall time.
Independent of everything else and the largest unclaimed speed item on the
list. The measurement that decides it is in `NEXT.md` §6a.

### P3 — Scores that are not distances

The `p -> 0` limit of `sum_j delta_j^p` is the *count of coordinates that
differ at all* — and the hash already computes it, as digit agreements
across `K*L` digits. Its contrast is binomial, spread `~sqrt(KL)`, so unlike
a distance it **does not concentrate as `d` grows**, which is exactly §7's
failure mode. Cost is near zero: it is the cross-table multiplicity the join
already produces (§3 measures 6% duplicates today; mixed-resolution tables
would raise it deliberately).

**Honest scoping.** C2LSH already uses collision counting — as a *candidate
selection* threshold, still refined by exact Euclidean distance afterwards.
The proposal here is different in intent: collision count as the **final
ranking**, justified by the high-`d` contrast property and by §7's finding
that magnitude carries no information anyway. The mechanism is known; the
contribution would be the analysis and the space-filling application. P0
decides how much of that survives.

Testable with **no index changes** through `bench_recall_ablation.py`'s
existing substitution hook. Two further candidates on the same hook:

* **Angular coverage.** §7.1 says the signal is *which* points come back.
  What the force step wants is a set that *surrounds* the query — return
  `k` points chosen for angular spread (greedy MMR/DPP over the candidate
  list) rather than the `k` nearest.
* **Occupancy-aware ranking.** The index knows every bucket's load;
  "nearest point in a sparse region" is free information that speaks
  directly to empty-space filling, and is the bridge to P4.

Deprioritised: local scaling, mutual proximity, SNN. §7.6 ruled out
hubness-corrected metrics on measurement (no runaway hubs anywhere, `p=1`
or `p=0.5`, d=8 or d=32). Their contrast rationale is formally separate but
strictly dominated by P3 on principle and by collision counting on cost.

### P4 — Build the family

Gated by P2b, and subordinate to P1. Now a build, not a research question.

* Per-`(table, dim)` breakpoint arrays; `code = searchsorted(bp, frac(x+u))`;
  mixed-radix keys `sum_j code_j * prod_{i<j} B_i`.
* **Digit order is a free design choice — use it.** Prefix relaxation drops
  low-order digits, so place the largest-`B` (most discriminative) digits
  low-order and relaxation widens gently.
* Multi-probe survives: "digit closest to a cell boundary" is still well
  defined under variable widths.
* The tuner must be re-derived. With variable radix and a drop atom, bucket
  load is a random variable; `K = round(log_B(n/target))` breaks, and the
  key-fits-in-int64 guard becomes a constraint on a random product
  `prod_j B_j <= 2^62`.
* `p=1.0` ships as the regression anchor. Note it reproduces today's
  *collision law* only in the small-`delta` limit, not bit-for-bit — a
  Poisson grid is not a uniform grid. Bit-identity is a P2a baseline
  concern, not a P3 constraint.
* **Validation criterion, fixed in advance:** recall against true `L^p` k-NN
  must beat **93.2%**, §7.3's structural rerank ceiling. Below that the
  construction bought nothing and gets recorded as rejected.
* Numerics: `(sum u^p)^(1/p)` reaches 1.4e5 at d=32, p=0.25 — range-check
  the force normalisation and the log-sum-exp path.

### P5 — Autoencoders / matrix factorization

Last, highest variance, and decidable by one measurement.

**The gate is intrinsic dimension.** Any reduction method buys something
only if the points lie near a manifold of dimension `< d`. Estimate ID
(TWO-NN and MLE) on **two** populations, because they should differ sharply
and only one of them has ever been benchmarked:

* *ESS-converged sets* — engineered to be as close to uniform as the method
  can manage, so ID should read `~= d` and every reduction method is closed
  by measurement rather than by opinion.
* *Mid-optimization OBLESA populations* — real `pyBlindOpt` runs on its
  benchmark functions, where the population clusters around optima and ID
  may read well below `d`.

That split is not drawn anywhere in `NEXT.md` and it decides whether P4 is
a dead end or the most interesting item here: reduction may be closed for
the *initialization* use case and open for the *refinement* one.

If the gate opens, two genuinely different products — do not conflate them:

* **Learned hash.** Train the quantiser so `P(collide) ~ exp(-c*delta_p)`
  directly, without needing a stable law — the ML answer to P3. Cost stated
  up front: it makes the backend non-deterministic, and ESS re-fits from
  scratch on every call (§7.5), so training lands on every `esa`.
* **Void model.** Low-rank / NMF factorization of the (bucket × table)
  **occupancy** matrix — modelling emptiness, not points. Its output is not
  a k-NN query but a new API, `query_voids(k) -> sparsest cells`, serving
  ESS's proposal step directly. This is the one that matches the original
  framing, and it is additive rather than contract-breaking.

---

## 3. Decisions taken

* **`p` lives in the `ToroidalNN` constructor**, alongside `num_tables`,
  `resolution` and the rest, with `None`/`"auto"` selecting a measured
  `p(d)` heuristic. Explicit arguments win, as they already do everywhere.
* **There is no standing normative contract document.** `CONTRACT.md`
  specified guarantees particular to the L1 grid hash — byte-identical
  tables, the `max(0, 1-B*delta)` law, the digit structure prefix
  relaxation relies on. A backend on a different family answers to its own
  specification. `lsh.py` remains the executable reference for the L1
  backend; P4's family will document its own.
* **Both repos merged to `main`** (torann `7538844`, ess `f9c22da`) before
  this branch, so the metric work does not stack on an unmerged
  optimization branch.
* **Venue GECCO 2027**, no hard deadline. That is roughly a year, which is
  enough for P1 to fail and be re-aimed once — so P1 runs first and dirty.

## 4. Still open

* P0's ANN half is incomplete. **No novelty claim until it is.**
* Whether the `p=1` Poisson member should replace the uniform grid in the
  shipped L1 path, or coexist. Decided by P4's speed measurement, not now.
* The FAISS comparison remains unestablished on the ESS workload (§7.5).
  If it appears in the paper it has to be a controlled run on the real loop
  — `k=5`, self-join, index mutated in place, ESS-converged data — not the
  uniform-random scripts in `examples/`.
* **Why the torus exists** — recorded because it was nearly mis-framed here
  as a modelling assumption. It is not. OBLESA optimizes over a bounded box
  and delivers into one; the torus lives *inside* the relaxation. ESS moves
  particles by physical repulsion, and with hard walls the wall behaves as
  an enormous particle: it perturbs far too many particles at once, and the
  resulting wall-particle bouncing never settles, which is what broke the
  early-stop mechanism. The toroidal domain removes the wall rather than
  softening it — particles navigate off one edge and back on the other, so
  they neither clamp to the boundary nor collapse toward the centre. So the
  wrap is the fix for a diagnosed failure, and the open question is only the
  empirical one P1 answers anyway: whether points relaxed under a toroidal
  metric are good starting points for a box-bounded objective.
