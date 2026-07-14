# Plan — a toroidal, L1-sensitive hash and the ANN built on it

Goal: a custom ANN for ESS — points in `[0,1)^d` with wrap-around, L1 (ideally
L0.5) distances, index rebuilt every epoch, each point querying `k = 2d`
neighbours. The centrepiece is the *hash*: a stable, toroidal, L1-mimicking
hash function, validated in low dimension before any implementation.

## Process

| Phase | What | Output | Gate |
|---|---|---|---|
| 1 | Candidate hash families, on paper | this document, §Candidates | agree on candidates |
| 2 | 1D experiments: collision law, seam invariance, stability | `exploration/exp_1d.py`, figures | disqualify broken candidates |
| 3 | 2D experiments: concatenation law, bucket geometry, mini-ANN | `exploration/exp_2d.py`, figures | **concept approval** |
| 4 | Implementation: single NN class (brute + LSH), batch queries, range post-filter | `src/` — **done 2026-07-14** | code review |
| 5 | Benchmark & tuning at ESS scales (n, d, k = 2d) | `examples/benchmark.py` — first results in README | defaults locked |

Phase 4 implemented the gate outcomes as specified: L1-only `ToroidalNN` with
the two-tier lifecycle (`fit(static, candidates)` / `query()` /
`update()` / `promote()`), drift-budgeted updates using the experiment-H
closed form (`staleness` property exposes the prediction), tuning of
(B, K, L) from the fit-time `k=`/`radius=` hints, and the exact fallback.
First lifecycle benchmark (d=16, k=32): two-tier maintenance ~11× cheaper
than per-epoch refits at equal recall (~0.99); the deferred budget skips 60 %
of tier rehashes at zero recall cost.

## What we require of the hash

For a bucket hash `h` drawn from a family H, with `δ` the toroidal distance:

1. **Toroidal**: `P[h(x)=h(y)]` depends only on `δ(x,y)` — position-invariant,
   in particular invariant across the 0/1 seam.
2. **L1-mimicking**: collision probability strictly decreasing in `δ`
   (monotone), ideally with a closed form we can reason about.
3. **Stable**: deterministic given its parameters, and *incrementally* stable —
   a point moving by a small step `s` changes bucket with probability O(s),
   so per-epoch index churn is proportional to how far points actually moved.
4. **Cheap & vectorisable**: one pass of NumPy arithmetic per table.

## Candidates

**H1 — rotated integer grid.** Per dimension, integer resolution `B ≥ 2`,
offset `u ~ U[0,1)`:

```
h(x) = floor(B · ((x + u) mod 1))
```

Claim: on the circle, `P[h(x)=h(y)] = max(0, 1 − Bδ)` **exactly** — B equal
arcs under a uniform random rotation; no seam, no approximation. Multi-dim by
concatenating K sampled dimensions ⇒ collision `∏ᵢ max(0, 1 − Bδᵢ)`
≈ `exp(−B·L1)`. Also: for a uniform random pair the per-dimension collision
probability is exactly `1/B`, giving closed-form bucket-load control.

**H2 — integer-projection grid** (the family the old attempts used):

```
h(x) = floor(B · ((z·x + u) mod 1)),   z a random integer vector
```

Integer `z` makes `z·x mod 1` well-defined on the torus, so it *is* toroidal.
Suspicion: it is **not monotone** in δ. In 1D, `z(x+δ) − zx = zδ`, so the
collision probability is `max(0, 1 − B·d_circ(zδ))` — a function of `zδ`
wrapped around the circle. For `|z| = 2`, points at the *maximal* toroidal
distance δ = 0.5 collide with probability 1. Experiment 1D-A/1D-D tests this.

**H3 — shifted grid without wrap (naive control)**: same as H1 but the grid
lives on the line, not the circle (`floor((x + u)/w)`, no mod). This isolates
exactly what toroidal correctness buys: it should match H1 away from the seam
and fail for pairs straddling it. (The classic Cauchy p-stable L1 family is in
this class too — real-valued projections cannot wrap; rejected on paper, the
control stands in for it.)

**H4 — binary case, B = 2** ("toroidal hyperplanes"): H1 at its coarsest;
included in experiments because high-dimensional L1 concentration favours
coarse grids (contrast argument, to be verified at phase 5 scale).

## Experiment protocol

**1D** (`exploration/exp_1d.py`):
- **A. Collision law**: measured `P[collision]` vs toroidal δ for H1(B=2),
  H1(B=4), H2(B=4, z ~ round N(0,2)), H3(B=4). Overlay `max(0,1−Bδ)`.
  *Accept H1 if measured = theory within MC error; reject H2 if non-monotone;
  reject H3 if below theory (seam leakage).*
- **B. Winding defect**: H2 with fixed z ∈ {1,2,3}: collision vs δ against
  the predicted `max(0, 1 − B·d_circ(zδ))`.
- **C. Seam invariance**: δ fixed at 0.05, pair midpoint swept across the
  seam. *Accept only position-flat curves.*
- **D. Stability**: P[bucket changes] when a point moves by step s; H1 theory
  is `min(1, Bs)` per dimension. *Matters for epoch updates: churn ∝ movement.*

**2D** (`exploration/exp_2d.py`):
- **E. Concatenation law**: K=2, B=3 — measured collision vs
  `∏ max(0,1−Bδᵢ)`; accept if it sits on y = x.
- **F. Bucket geometry** (visual): single-table cell maps for H1 vs H2 (H2
  should show winding stripes), and the union-of-tables candidate region
  around a query at a corner — H1's region must wrap all four corners, H3's
  must clip at the border.
- **G. Mini-ANN**: n = 4000 uniform points, k = 8: candidate-set recall and
  candidate count vs number of tables L. *Concept approved if recall ≥ 0.95
  with modest L and candidate sets ≪ n.*
- **H. Staleness** (`exploration/exp_update.py`): recall of a never-rehashed
  index as points drift, probes on/off, plus the stale-key fraction vs its
  closed form. *Informs the epoch-update strategy (gate outcome 2).*

## Carried requirements for phase 4 (unchanged from the brief)

Single `NN` class — exact brute force (NumPy) below a size threshold, LSH
above; batch queries; per-epoch rebuild must be one vectorised pass; range
queries as a distance post-filter on the candidate set; pure Python + NumPy
until the concept is settled. The hash filter factorises over per-dimension
distances, so one index serves L1, L2 and L0.5 refinement — only the refine
metric changes.

## Gate outcomes (settled 2026-07-13)

1. **L1-only contract.** H1 is not projection-based, so no p-stable family is
   involved; its guarantee (`∏ max(0,1−Bδᵢ)` ≈ `exp(−B·L1)`) is an L1
   guarantee. Refining an L1-tuned candidate set with L2/L0.5 carries no
   guarantee, so L2/L0.5 are dropped from the public API. The hash *is* the
   metric contract.
2. **Monotone collisions suffice** (ESS mainly needs "move away from the
   closest points"; small errors self-correct next epoch). Updates are
   **selective**: recompute keys of the moved points (vectorised), re-place
   only the ones whose key changed, by delete + merge — the index is exact
   after every update. *(Revised 2026-07-14: an optional deferred mode —
   skip rehashing while predicted staleness `1−(1−B·E|Δ|)^K` stays under a
   budget, probes covering the drift per experiment H — was implemented,
   benchmarked at zero recall cost, and then removed at Mário's request:
   accuracy is non-negotiable and the selective path is cheap enough.)*
3. **Tuning is driven by ESS's own heuristics** (its per-query k / range
   rules): B from the target radius (per-dim collision reaches zero at
   δ = 1/B, probes extend reach to 2/B — pick B so the expected per-dim
   neighbour distance sits well inside), K from the bucket-load closed form
   `n/B^K`, L from the recall target. Expose a `tune(k= / radius=)` hook that
   estimates the neighbour scale on a sample and sets (B, K, L).
4. **Two-tier lifecycle**, mirroring ESS exactly: a *static tier*
   (`[0, n]`, hashed and sorted once; grows only by promotion) and a
   *candidate tier* (`(n, m]`, small, rebuilt/updated each epoch). Queries
   come **from candidates only** and search both tiers.
   `promote(new_batch)` merges the candidate tier into the static tier
   (linear merge of sorted arrays, no re-sort) and installs the new batch.

## Phase 6 — native backend (agreed direction)

The NumPy pipeline is memory-bound: refinement materialises (pairs × d)
temporaries and every stage round-trips RAM. FAISS on the same data
(`examples/compare_faiss.py`) shows SIMD C++ is ~40–150× faster, which puts
millisecond epochs in reach. Chosen backend: **C++ via pybind11** (or plain
C + cffi) — one extension module exposing three batch kernels:

1. fused gather+refine: bucket ranges → toroidal L1 distances without
   materialising candidate matrices (AVX-friendly: `|a−b|`, `min(x, 1−x)`,
   horizontal add);
2. per-query top-k (heap per query, OpenMP over queries);
3. key computation + selective merge for updates.

Rationale: matches the NumPy buffer protocol zero-copy, OpenMP for the
embarrassingly parallel query batches, and the same stack FAISS itself uses.
Rust/PyO3 was considered and set aside (owner's preference; binding overhead
is irrelevant for batched calls, but there is no upside here either). Numba
is the cheap intermediate step (~10–20× on the refine kernel with `prange`)
if a pure-pip install must be preserved. The Python class stays as the
orchestrator and reference implementation; kernels are swapped per
availability.

## Remaining open question

- Phase 5: at scale, is the acceptance metric set-recall or the distance
  ratio of returned neighbours? (Concentration in high d says they diverge;
  point 2 suggests distance ratio is what ESS actually needs.)
