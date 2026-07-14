"""Head-to-head backend benchmark: python vs rust.

For every installed backend (or ``--backends``), over an (n, d) grid, times
each lifecycle operation of the ESS workload and measures recall on a query
sample against blocked exact search:

  fit      build both tiers from zero        (n static + batch candidates)
  query    candidate self-join, k = 2d       (reported as us/query)
  update   sigma=0.01 drift step, selective
  promote  merge tiers + install a new batch

Threading: the python backend is single-threaded by design; native backends
parallelise over queries. Control native threads with OMP_NUM_THREADS /
RAYON_NUM_THREADS before running.

Run from the repository root:
  python examples/bench_backends.py [--sizes 20000,100000] [--dims 8,16,32]
                                    [--backends python,rust] [--epochs 3]
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

BATCH = 3_000
SIGMA = 0.01
SAMPLE = 200


def exact_knn_sample(pts, Q, k, exclude_ids, block=100_000):
    """Blocked exact toroidal-L1 k-NN (reference at any n)."""
    m = Q.shape[0]
    best_i = np.full((m, 0), -1, dtype=np.int64)
    best_d = np.full((m, 0), np.inf)
    for s in range(0, pts.shape[0], block):
        e = min(pts.shape[0], s + block)
        diff = np.abs(Q[:, None, :] - pts[None, s:e, :])
        D = np.minimum(diff, 1.0 - diff).sum(-1)
        ids = np.broadcast_to(np.arange(s, e), D.shape).copy()
        mask = (ids == exclude_ids[:, None])
        D[mask] = np.inf
        best_d = np.concatenate([best_d, D], axis=1)
        best_i = np.concatenate([best_i, ids], axis=1)
        if best_d.shape[1] > k:
            part = np.argpartition(best_d, k - 1, axis=1)[:, :k]
            best_d = np.take_along_axis(best_d, part, axis=1)
            best_i = np.take_along_axis(best_i, part, axis=1)
    order = np.argsort(best_d, axis=1, kind="stable")
    return np.take_along_axis(best_i, order, axis=1)


def run_scenario(backend, n, d, epochs):
    k = 2 * d
    rng = np.random.default_rng(0)
    static = rng.random((n, d))
    cands = rng.random((BATCH, d))

    nn = ToroidalNN(seed=1, backend=backend)
    t0 = time.perf_counter()
    nn.fit(static, cands, k=k)
    t_fit = time.perf_counter() - t0

    t_query, t_update = [], []
    idx = np.empty((0, 0), dtype=np.int64)
    for _ in range(max(1, epochs)):
        t0 = time.perf_counter()
        idx, _ = nn.query()
        t_query.append(time.perf_counter() - t0)
        step = rng.normal(0, SIGMA, (nn.n_candidates, d))
        new = np.mod(nn.candidates + step, 1.0)
        t0 = time.perf_counter()
        nn.update(new)
        t_update.append(time.perf_counter() - t0)

    sample = rng.choice(BATCH, SAMPLE, replace=False)
    pts = np.vstack([nn._arena[:nn.n_static], nn.candidates])
    truth = exact_knn_sample(pts, nn.candidates[sample], k,
                             nn.n_static + sample)
    hits = sum(len(np.intersect1d(idx[s], truth[i]))
               for i, s in enumerate(sample))

    t0 = time.perf_counter()
    nn.promote(rng.random((BATCH, d)))
    t_promote = time.perf_counter() - t0

    return {
        "backend": backend, "n": n, "d": d, "k": k,
        "tuned": {"B": nn._B, "K": nn._K, "L": nn._L, "probes": nn.probes},
        "fit_s": round(t_fit, 4),
        "query_us_per_q": round(1e6 * float(np.median(t_query)) / BATCH, 1),
        "update_s": round(float(np.median(t_update)), 4),
        "promote_s": round(t_promote, 4),
        "recall": round(hits / truth.size, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="20000,100000")
    ap.add_argument("--dims", default="8,16,32")
    ap.add_argument("--backends", default=",".join(available_backends()))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--tag", default="bench_backends")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    dims = [int(x) for x in args.dims.split(",")]
    backends = args.backends.split(",")

    header = (f"{'backend':>8} {'n':>8} {'d':>3} {'B/K/L':>8} {'fit(s)':>8} "
              f"{'query(us/q)':>12} {'update(s)':>10} {'promote(s)':>10} "
              f"{'recall':>7}")
    print(header)
    print("-" * len(header))
    results = []
    for n in sizes:
        for d in dims:
            for backend in backends:
                r = run_scenario(backend, n, d, args.epochs)
                results.append(r)
                tuned = f"{r['tuned']['B']}/{r['tuned']['K']}/{r['tuned']['L']}"
                print(f"{r['backend']:>8} {r['n']:>8} {r['d']:>3} {tuned:>8} "
                      f"{r['fit_s']:>8.3f} {r['query_us_per_q']:>12.1f} "
                      f"{r['update_s']:>10.4f} {r['promote_s']:>10.4f} "
                      f"{r['recall']:>7.3f}")

    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults saved to {path}")


if __name__ == "__main__":
    main()
