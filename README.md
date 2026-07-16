# torann — TORoidal Approximate Nearest Neighbours

Exact + approximate k-NN and range search on the unit torus `[0, 1)^d` under
toroidal **L1** — the metric is the contract (see below). One Python package,
three interchangeable implementations of one interface (`torann/base.py`):
exact brute force (`brute.py`), the pure-Python LSH reference (`lsh.py`), and
the Rust core (`torann._native`, built with maturin) — see `torann/CONTRACT.md`
and `ANALYSIS.md`; ~60 ms per simulated ESS epoch at recall ≈ 0.99, 60× the
NumPy pipeline, within 1.06–1.56× of FAISS's exact SIMD Flat scan on data
FAISS can actually answer.

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
crossover — 4096 python, 512 rust); `backend` (`"auto"` prefers the
fastest installed of `rust`, `python`).

## Performance

Full analysis (kernel pass, grids, crossover, complexity, FAISS) in
`ANALYSIS.md`. Headlines from the v3 measurements (AMD Ryzen AI 7 PRO 350,
16 threads):

ESS main loop, simulated end to end (`examples/ess_sim.py`; 15 000 anchors
+ 10 batches of 3 000 candidates × 32 epochs → 45 000 points, d=16, k=32,
σ=0.01, exact toroidal truth spot-checked per batch):

| system | epoch | total | recall per batch |
|:--|--:|--:|:--|
| **torann [rust]** | **61.5 ms** | **27.4 s** | **0.983–0.995** |
| faiss Flat L1, rebuilt each epoch | 106.2 ms | 41.3 s | 0.25–0.28 |
| torann [python] | 3 729 ms | — | 0.983 |

FAISS Flat is exact — for the wrong metric: seam-blind L1 misses ~73 % of
the true toroidal neighbours here, and the misses are structural. On the
workload this library exists for, torann is 1.7× faster *and* correct.

Queries run 11–148 µs at 16 threads across n ∈ [20k, 1M] (d=16, torus
data) — 60–120× the NumPy pipeline — with updates at 0.9–2.1 ms; identical
recall by construction (byte-identical tables, `torann/CONTRACT.md`). The
Rust LSH beats exact brute force from n = 500 up at recall 1.00; the
pure-Python LSH crosses at n ≈ 4–8k, which is what the `brute_threshold`
defaults encode. At σ=0.01 only ~13 % of keys change per epoch, so the
selective update re-places a fraction of the tier; prefix relaxation costs
nothing when the index is tuned (0 of 3 000 queries under-fill).

### vs FAISS

FAISS cannot answer the toroidal query — exact seam-blind L1 misses ~74 %
of true toroidal neighbours at d=16 (`examples/compare_faiss.py`), so it is
a throughput reference only. On wrap-free common-ground data
(`examples/compare_faiss_flat.py` conditions), torann sits within
**1.06–1.56×** of FAISS's exact SIMD Flat scan at equal ≈ 1.0 recall
(1.06× at n=100k — parity at ESS scales), builds 1M-point indexes in ~1 s
(HNSW: 24–32 s), and updates incrementally in milliseconds — while HNSW's
L1 recall collapses in high d (0.27 at d=32, n=1M).

Two conclusions. *Correctness*: even exact seam-blind search misses ~74 % of
the true toroidal neighbours at d=16 — with concentrated distances almost
every true neighbour pair wraps in at least one dimension, so FAISS is not
an option for this problem, only a throughput reference. *Throughput*: SIMD
C++ on the same data is ~40–150× faster than the NumPy pipeline, which is
the headroom target for a native backend (see PLAN.md phase 6).

## Development

```
python -m unittest discover -s test    # conformance suite, per installed backend
python examples/ess_sim.py             # the ESS main loop end to end, vs FAISS
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
