# Past the `d ~ 8` wall: the research programme

Read `NEXT.md` first — §3 (measured geometry), §4 (measured and rejected)
and §7 (the force-law collapse) are the standing facts this programme is
built on and must not be re-derived. This document is the plan that follows
from them. The primary output is a **paper**; the index is the artefact that
supports it.

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

### P0 — Literature, before any Rust

The paper is the primary output, so prior art is a blocking dependency, and
**this session did not complete it.** Semantic Scholar rate-limited
throughout; arXiv does not index the relevant corpus (Datar SoCG'04, Gan
SIGMOD'12, Aggarwal ICDT'01 are all DB/theory conference papers). What was
established:

* **Datar et al. (SoCG'04)** — canonical `p`-stable LSH, explicitly covers
  `p < 1`. The reference the §1.1 argument must engage with.
* **C2LSH, Gan et al. (SIGMOD'12, 330+ citations)** — uses *collision
  counting* over hash functions, with dynamic compound hash functions and
  virtual rehashing. **This overlaps P2 below and must be read before P2
  is claimed as novel** (see the honest scoping note there).

Still to check, and each one can invalidate a claim:

* Whether the rate-mixture / Poisson-breakpoint construction of §1.2 is
  known. The completely-monotone-to-Laplace-mixture step is textbook and
  grid LSH is textbook; the *toroidal* combination is what may be new.
* Chierichetti & Kumar on which similarity functions are LSHable —
  directly bounds what item 3 (non-distance scores) can hope for.
* Aggarwal, Hinneburg & Keim (ICDT'01) on fractional norms in high
  dimension — the standing citation for §7.2's contrast argument.
* Prior work on ANN under periodic boundary conditions at all.

Retry Semantic Scholar (429 is transient) or use institutional access.

### P1 — Baseline and gate, run concurrently

**P1a (torann): does the index scale the way we think?** Nothing in
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

**P1b (ess): the force-law gate.** §7.2's chain has an untested link:
contrast is a property of the metric, not the index. A perfect `L^p` index
only pays if the force law consumes `L^p` distances. Brute path only, no
index changes, d in {8, 16, 32}, 3 seeds, paired, both starts: swap the
force kernel's distance to `L^0.5` / `L^0.25` with `_l1_radius_heuristic`
re-derived per `p`. Run as a 2×2 against `sigma in {0.5, tuned}` — §7.2
shows required sharpness relaxes with `p`, so without the second factor
"lower `p` helps" and "sharper `sigma` helps" are inseparable, and the
`sigma` fix is far cheaper.

> **Decision rule.** If `L^0.5` at tuned `sigma` does not beat L1 at tuned
> `sigma` by more than the between-seed sd (0.0037 CE at d=32), P3 is
> descoped from "make ESS better" to "make a better index", and the paper's
> claim narrows accordingly.

**P1c: the `0.10` collision target.** `_tune` sets
`L = ceil(log(0.10)/log(1-p1))` — a hardcoded 90% collision target for a
true k-NN, never derived from what ESS needs, while §6 measured that imposed
recall 0.5 costs 0.45% CE. Dropping to 50% gives `L≈7` where the model gives
22: ~3× less candidate work on a path that is 68–85% of ESS wall time.
Independent of everything else and the largest unclaimed speed item on the
list. The measurement that decides it is in `NEXT.md` §6a.

### P2 — Scores that are not distances

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

### P3 — Build the family

Gated by P1b. Now a build rather than a research question.

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
  Poisson grid is not a uniform grid. Bit-identity is a P1a baseline
  concern, not a P3 constraint.
* **Validation criterion, fixed in advance:** recall against true `L^p` k-NN
  must beat **93.2%**, §7.3's structural rerank ceiling. Below that the
  construction bought nothing and gets recorded as rejected.
* Numerics: `(sum u^p)^(1/p)` reaches 1.4e5 at d=32, p=0.25 — range-check
  the force normalisation and the log-sum-exp path.

### P4 — Autoencoders / matrix factorization

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
  backend; P3's family will document its own.
* **Both repos merged to `main`** (torann `7538844`, ess `f9c22da`) before
  this branch, so the metric work does not stack on an unmerged
  optimization branch.

## 4. Still open

* P0 is incomplete — see the list there. **No novelty claim until it is.**
* Whether the `p=1` Poisson member should replace the uniform grid in the
  shipped L1 path, or coexist. Decided by P3's speed measurement, not now.
* The FAISS comparison remains unestablished on the ESS workload (§7.5).
  If it appears in the paper it has to be a controlled run on the real loop
  — `k=5`, self-join, index mutated in place, ESS-converged data — not the
  uniform-random scripts in `examples/`.
