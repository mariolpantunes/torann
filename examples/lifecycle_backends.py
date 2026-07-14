"""The exact ESS lifecycle, timed per backend.

Workload: 15k anchors + 3 batches of 3k candidates, d=16, k=32, 5 epochs
per batch, sigma=0.01 drift. Reports wall time of fit / query / maintenance
(update+promote), the mean epoch latency (query+update — the number ESS
feels), and recall on a sample at the last epoch of each batch.

Run from the repository root:
  python examples/lifecycle_backends.py [--backends python,c,rust]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torann import ToroidalNN  # noqa: E402
from torann import available_backends  # noqa: E402

D = 16
K = 2 * D
N_STATIC = 15_000
BATCH = 3_000
BATCHES = 3
EPOCHS = 5
SIGMA = 0.01
SAMPLE = 300


def sample_recall(nn, rng):
    idx, _ = nn.query()  # fresh query against current positions
    pts = np.vstack([nn._arena[:nn.n_static], nn.candidates])
    sample = rng.choice(nn.n_candidates, SAMPLE, replace=False)
    q = nn.candidates[sample]
    diff = np.abs(q[:, None, :] - pts[None, :, :])
    Dm = np.minimum(diff, 1.0 - diff).sum(-1)
    Dm[np.arange(SAMPLE), nn.n_static + sample] = np.inf
    exact = np.argsort(Dm, axis=1)[:, :K]
    hits = sum(len(np.intersect1d(idx[s], exact[i]))
               for i, s in enumerate(sample))
    return hits / exact.size


def run(backend):
    rng = np.random.default_rng(0)
    static = rng.random((N_STATIC, D))
    batch = rng.random((BATCH, D))

    nn = ToroidalNN(seed=1, backend=backend)
    t0 = time.perf_counter()
    nn.fit(static, batch, k=K)
    t_fit = time.perf_counter() - t0

    t_query = t_maint = 0.0
    epoch_ms, recalls = [], []
    for _ in range(BATCHES):
        for _ in range(EPOCHS):
            t0 = time.perf_counter()
            nn.query()
            tq = time.perf_counter() - t0
            step = rng.normal(0, SIGMA, (nn.n_candidates, D))
            new = np.mod(nn.candidates + step, 1.0)
            t0 = time.perf_counter()
            nn.update(new)
            tu = time.perf_counter() - t0
            t_query += tq
            t_maint += tu
            epoch_ms.append(1e3 * (tq + tu))
        recalls.append(sample_recall(nn, rng))
        t0 = time.perf_counter()
        nn.promote(rng.random((BATCH, D)))
        t_maint += time.perf_counter() - t0

    return {
        "backend": backend,
        "fit_s": round(t_fit, 3),
        "query_s": round(t_query, 2),
        "maint_s": round(t_maint, 2),
        "total_s": round(t_fit + t_query + t_maint, 2),
        "epoch_ms": round(float(np.mean(epoch_ms)), 1),
        "recall_per_batch": [round(r, 3) for r in recalls],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default=",".join(available_backends()))
    args = ap.parse_args()

    print(f"ESS workload: {N_STATIC} anchors +{BATCHES}x{BATCH} candidates, "
          f"d={D}, k={K}, {EPOCHS} epochs/batch, sigma={SIGMA}")
    header = (f"{'backend':>8} {'fit(s)':>7} {'query(s)':>9} {'maint(s)':>9} "
              f"{'total(s)':>9} {'epoch(ms)':>10}  recall/batch")
    print(header)
    print("-" * len(header))
    results = []
    for backend in args.backends.split(","):
        r = run(backend)
        results.append(r)
        print(f"{r['backend']:>8} {r['fit_s']:>7.3f} {r['query_s']:>9.2f} "
              f"{r['maint_s']:>9.2f} {r['total_s']:>9.2f} "
              f"{r['epoch_ms']:>10.1f}  {r['recall_per_batch']}")

    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "lifecycle_backends.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults saved to {out_dir}/lifecycle_backends.json")


if __name__ == "__main__":
    main()
