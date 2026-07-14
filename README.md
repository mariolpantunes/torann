# torann — TORoidal Approximate Nearest Neighbours

Exact + approximate k-NN and range search on the unit torus `[0, 1)^d` under
toroidal **L1** — the metric is the contract (see below). One Python package,
three interchangeable implementations of one interface (`torann/base.py`):
exact brute force (`brute.py`), the pure-Python LSH reference (`lsh.py`), and
the Rust core (`torann._native`, built with maturin) — see `torann/CONTRACT.md`
and `ANALYSIS.md` for the bake-off; ~80 ms per ESS epoch, 25–75× NumPy.

Built for [ESS](https://github.com/mariolpantunes/ess): anchors that never
move, candidate points that move a little every epoch, each candidate asking
for `k = 2d` neighbours, batches of candidates freezing into anchors over
time. The design was validated concept-first: `PLAN.md` records the process
and the gate decisions; `exploration/` holds the 1D/2D experiments behind
every claim.

## The hash

An LSH built from **randomly rotated integer grids** — a family that is
*exactly* L1-sensitive on the torus. Per sampled dimension, with integer
resolution `B ≥ 2` and offset `u ~ U[0,1)`:

```
c(x) = floor(B · ((x + u) mod 1))          P[c(x)=c(y)] = max(0, 1 − Bδ)
```

The collision law is exact — B equal arcs under a uniform random rotation, no
seam, no approximation (experiments A, C). A table concatenates K sampled
dimensions, so collisions decay as `∏ max(0,1−Bδᵢ) ≈ exp(−B·L1)`
(experiment E), a uniformly random pair collides per dimension with
probability exactly `1/B` (closed-form bucket load `n/B^K`), and a point that
moves by `s` changes a cell with probability `B·s` (experiment D).

**Why L1-only:** this family needs no p-stable projections — its guarantee is
inherently an L1 guarantee. Refining an L1-tuned candidate set under L2 or
L0.5 carries no guarantee, so those metrics are deliberately not offered
(gate outcome 1). The rejected alternatives are documented and measured in
`exploration/`: integer projections alias far points onto near ones
(winding, experiment B); real-valued projections cannot wrap (experiment C).

## The index

- **Two tiers** (gate outcome 4): a *static tier*, hashed and key-sorted once,
  that grows only by `promote()` — a linear `searchsorted + insert` merge,
  never a re-sort — and a small *candidate tier* refreshed per epoch.
  Queries default to "each candidate against everything", the ESS inner loop.
- **Selective updates** (gate outcome 2): `update()` re-places only the
  points that moved far enough to change a cell (`P[change] = B·step` per
  dimension, experiment D), by delete + merge — never a full re-sort. The
  index is exact after every update: fast *and* accurate.
- **No brute-force fallback**: an under-filled query widens its buckets by
  *prefix relaxation* — keys are digit concatenations, so dropping low-order
  digits turns a bucket into a contiguous range of the sorted key array (two
  `searchsorted` calls per table per level). Level K spans the whole table,
  so k results are structurally guaranteed without a distance scan.
- **Self-tuning** (gate outcome 3): `fit(..., k=...)` or `radius=...` feeds
  ESS's own query heuristic into the index — B from the measured neighbour
  scale, K from the bucket-load closed form, L from a recall target.
  Explicit constructor arguments always override tuning.
- **Exact below `brute_threshold`** total points (blocked NumPy brute force),
  and an exact fallback guarantees every query returns k true results when
  the dataset allows it.

## Usage

```python
import numpy as np
from torann import ToroidalNN

d = 16
static = np.random.rand(15_000, d)      # anchors: never move
batch = np.random.rand(3_000, d)        # candidates: move each epoch

nn = ToroidalNN(seed=42)
nn.fit(static, batch, k=2*d)            # build + tune from zero

for epoch in range(epochs):
    idx, dist = nn.query()              # each candidate vs everything, k=2d
    new = force_step(nn.candidates, idx, dist)   # ESS physics
    nn.update(new)                      # drift-budgeted refresh

nn.promote(next_batch)                  # candidates freeze into the static
                                        # tier (linear merge); new batch in

nn.query_radius(0.25)                   # range query as a post-filter
nn.query(k=8, queries=Q)                # arbitrary external queries
```

Knobs (all optional — tuning fills them in): `num_tables`, `resolution`,
`dims_per_table`, `target_bucket_size` shape the recall/speed trade;
`probes`; `brute_threshold` (default: the backend's measured brute/LSH
crossover — 4096 python, 512 native); `backend` (`"auto"` prefers the
fastest installed of `c`, `rust`, `python`).

## Performance

ESS-workload benchmark (`examples/benchmark.py`): 15 000 anchors + 3 batches
of 3 000 candidates, d=16, k=32, 5 epochs per batch, σ=0.01 drift per epoch,
single core. Three maintenance strategies over the identical workload:

| strategy | query | maintenance | keys re-placed | recall/batch |
|:--|--:|--:|--:|:--|
| selective (default) | 66.7 s | **0.9 s** | 13 % / epoch | 0.985 · 0.988 · 0.987 |
| full tier re-sort each epoch | 66.1 s | 0.9 s | 100 % | 0.985 · 0.988 · 0.987 |
| full refit each epoch | 59.9 s | 9.2 s | 100 % | 0.985 · 0.988 · 0.969 |

Two-tier maintenance is ~10× cheaper than refitting at identical recall. At
σ=0.01 only ~13 % of keys change per epoch, so the selective path does 7×
less re-placement work than a tier re-sort (the wall-clock gap is hidden by
NumPy overhead at this tier size; it materialises at larger tiers and in the
native backend). Query time dominates and is the recall/speed knob
(`num_tables`, `target_bucket_size`).

The prefix-relaxation fallback costs nothing when the index is tuned (0 of
3 000 queries under-fill) and replaces the old brute-force fallback's worst
case: on a deliberately starved index it answers ~90× faster (0.14 s vs
12.4 s per epoch) by degrading recall instead of speed.

### vs FAISS (`examples/compare_faiss.py`)

n=24 000, d=16, 3 000 queries, k=32; recall measured against the **true
toroidal** L1 k-NN; FAISS on 16 threads, ToroidalNN mostly single-threaded:

| index | query | toroidal recall |
|:--|--:|--:|
| ToroidalNN (toroidal L1, tuned B=2 K=10 L=24) | 1142 µs/q | **0.968** |
| FAISS Flat L1 — exact but seam-blind | 27 µs/q | 0.259 |
| FAISS HNSW L1 — seam-blind | 7 µs/q | 0.255 |

Two conclusions. *Correctness*: even exact seam-blind search misses ~74 % of
the true toroidal neighbours at d=16 — with concentrated distances almost
every true neighbour pair wraps in at least one dimension, so FAISS is not
an option for this problem, only a throughput reference. *Throughput*: SIMD
C++ on the same data is ~40–150× faster than the NumPy pipeline, which is
the headroom target for a native backend (see PLAN.md phase 6).

## Development

```
python -m unittest discover -s test    # conformance suite, per installed backend
python examples/benchmark.py           # maintenance strategies (python backend)
python examples/lifecycle_backends.py  # the ESS lifecycle, per backend
python examples/bench_backends.py      # per-op grid over (n, d, backend)
python examples/crossover.py           # brute vs LSH crossover n*(d, backend)
python examples/compare_faiss_flat.py  # non-toroidal throughput vs FAISS
python exploration/exp_1d.py           # regenerate the concept experiments
python exploration/exp_2d.py
python exploration/exp_update.py
```

The project is a maturin mixed Rust/Python package: `maturin build --release`
produces the complete wheel (`Cargo.toml` + `src/lib.rs` are the native core;
`torann/` is the Python package). Without the compiled module everything
still runs on the pure-Python reference. The C contender from the phase-6
bake-off is preserved at tag `archive/backend-c`. Benchmarks, crossover
measurements, complexity tables and the code-quality comparison are in
`ANALYSIS.md`.
