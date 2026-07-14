"""Compare ToroidalNN against FAISS on the ESS query workload.

FAISS cannot answer the toroidal query — its metrics live in R^d, so pairs
straddling the 0/1 seam are scored wrong (exploration experiment C). The
comparison therefore has two axes:

* throughput — what a SIMD C++ backend achieves on the same data, as the
  target for a future native backend;
* correctness — recall of seam-blind exact L1 against the true toroidal
  k-NN, quantifying what using FAISS directly would cost ESS.

Run from the repository root:  python examples/compare_faiss.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ann import ToroidalNN  # noqa: E402

try:
    import faiss
except ImportError:
    sys.exit("faiss not installed (pip install faiss-cpu)")

D = 16
K = 2 * D
N_STATIC = 21_000
N_CAND = 3_000
SAMPLE = 500

rng = np.random.default_rng(0)
static = rng.random((N_STATIC, D))
cands = rng.random((N_CAND, D))
all_pts = np.vstack([static, cands]).astype(np.float32)


def toroidal_truth(queries, sample_ids):
    diff = np.abs(queries[:, None, :] - np.vstack([static, cands])[None, :, :])
    Dm = np.minimum(diff, 1.0 - diff).sum(-1)
    Dm[np.arange(len(queries)), N_STATIC + sample_ids] = np.inf
    return np.argsort(Dm, axis=1)[:, :K]


def bench(name, build, search, truth, sample_ids):
    t0 = time.perf_counter()
    index = build()
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx = search(index)
    t_query = time.perf_counter() - t0
    hits = sum(len(np.intersect1d(idx[s][idx[s] >= 0], truth[i]))
               for i, s in enumerate(sample_ids))
    recall = hits / truth.size
    print(f"{name:>28} build {t_build:7.3f}s  query {t_query:7.3f}s "
          f"({1e6 * t_query / N_CAND:7.1f} us/q)  toroidal-recall {recall:.3f}")


def main():
    sample_ids = rng.choice(N_CAND, SAMPLE, replace=False)
    truth = toroidal_truth(cands[sample_ids], sample_ids)
    print(f"workload: n={N_STATIC + N_CAND} d={D} queries={N_CAND} k={K} "
          f"(recall vs true toroidal L1 k-NN, {SAMPLE}-query sample)")
    print(f"faiss threads: {faiss.omp_get_max_threads()}")

    def build_ours():
        nn = ToroidalNN(seed=1).fit(static, cands, k=K)
        print(f"{'':>28} [ToroidalNN tuned: B={nn._B} K={nn._K} L={nn._L} "
              f"probes={nn.probes} -> {nn._L * (1 + nn.probes)} lookups/query]")
        return nn

    bench("ToroidalNN (toroidal L1)", build_ours,
          lambda nn: nn.query()[0], truth, sample_ids)

    def flat_search(index):
        _, idx = index.search(cands.astype(np.float32), K + 1)
        return np.array([row[row != N_STATIC + i][:K]
                         for i, row in enumerate(idx)])

    def build_flat():
        index = faiss.IndexFlat(D, faiss.METRIC_L1)
        index.add(all_pts)
        return index

    bench("faiss Flat L1 (seam-blind)", build_flat, flat_search,
          truth, sample_ids)

    def build_hnsw():
        index = faiss.IndexHNSWFlat(D, 32, faiss.METRIC_L1)
        index.add(all_pts)
        return index

    bench("faiss HNSW L1 (seam-blind)", build_hnsw, flat_search,
          truth, sample_ids)


if __name__ == "__main__":
    main()
