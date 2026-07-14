"""Benchmark the ESS lifecycle: fit -> (query, update)* -> promote -> ...

Three maintenance strategies over the same workload:
  * selective — the default: update() re-places only the points whose key
                changed (delete + merge), index exact after every epoch.
  * tier-sort — full candidate-tier rebuild (argsort) every epoch.
  * refit     — the phase-3 baseline: full index rebuild every epoch.

Reported per strategy: query time, maintenance time (update/promote/refit),
fraction of keys the selective path re-placed, and recall vs exact search on
a sample at the last epoch of each batch.
Run from the repository root:  python examples/benchmark.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ann import ToroidalNN  # noqa: E402

D = 16
K = 2 * D
N_STATIC = 15_000
BATCH = 3_000
BATCHES = 3
EPOCHS = 5
SIGMA = 0.01  # per-dim step per epoch
SAMPLE = 300


def sample_recall(nn, rng):
    """Recall of nn.query() against exact search, on a candidate sample."""
    idx, _ = nn.query()
    pts = np.vstack([nn._arena[:nn.n_static], nn.candidates])
    sample = rng.choice(nn.n_candidates, SAMPLE, replace=False)
    q = nn.candidates[sample]
    diff = np.abs(q[:, None, :] - pts[None, :, :])
    Dm = np.minimum(diff, 1.0 - diff).sum(-1)
    Dm[np.arange(SAMPLE), nn.n_static + sample] = np.inf
    exact = np.argsort(Dm, axis=1)[:, :K]
    hits = sum(len(np.intersect1d(idx[s], exact[i])) for i, s in enumerate(sample))
    return hits / exact.size


def run(strategy):
    rng = np.random.default_rng(0)  # same workload for every strategy
    static = rng.random((N_STATIC, D))
    batch = rng.random((BATCH, D))

    nn = ToroidalNN(seed=1)
    t0 = time.perf_counter()
    nn.fit(static, batch, k=K)
    t_maint = time.perf_counter() - t0
    t_query, recalls, moved = 0.0, [], []

    for b in range(BATCHES):
        for e in range(EPOCHS):
            t0 = time.perf_counter()
            nn.query()
            t_query += time.perf_counter() - t0

            step = rng.normal(0, SIGMA, (nn.n_candidates, D))
            new = np.mod(nn.candidates + step, 1.0)
            keys_before = nn._keys_c.copy() if strategy == "selective" else None
            t0 = time.perf_counter()
            if strategy == "refit":
                pts = nn._arena[:nn.n_static].copy()
                nn = ToroidalNN(seed=1)
                nn.fit(pts, new, k=K)
            elif strategy == "tier-sort":
                nn._arena[nn.n_static:] = new
                nn._pts32[nn.n_static:] = new.astype(np.float32)
                nn._build_candidates()
            else:
                nn.update(new)
                moved.append(float((nn._keys_c != keys_before).mean()))
            t_maint += time.perf_counter() - t0
        recalls.append(sample_recall(nn, rng))

        batch = rng.random((BATCH, D))
        t0 = time.perf_counter()
        nn.promote(batch)
        t_maint += time.perf_counter() - t0

    return {
        "strategy": strategy,
        "query_s": round(t_query, 2),
        "maint_s": round(t_maint, 2),
        "total_s": round(t_query + t_maint, 2),
        "keys_replaced": round(float(np.mean(moved)), 3) if moved else 1.0,
        "recall_per_batch": [round(r, 3) for r in recalls],
    }


def main():
    print(f"workload: static={N_STATIC} +{BATCHES}x{BATCH} candidates, "
          f"d={D}, k={K}, {EPOCHS} epochs/batch, sigma={SIGMA}")
    header = (f"{'strategy':>10} {'query(s)':>9} {'maint(s)':>9} {'total(s)':>9} "
              f"{'keys moved':>10}  recall/batch")
    print(header)
    print("-" * len(header))
    results = []
    for strategy in ("selective", "tier-sort", "refit"):
        r = run(strategy)
        results.append(r)
        print(f"{r['strategy']:>10} {r['query_s']:>9.2f} {r['maint_s']:>9.2f} "
              f"{r['total_s']:>9.2f} {r['keys_replaced']:>10.3f}  "
              f"{r['recall_per_batch']}")

    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults saved to {out_dir}/benchmark_results.json")


if __name__ == "__main__":
    main()
