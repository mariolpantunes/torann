"""Non-toroidal throughput reference: native backends vs FAISS.

FAISS cannot answer toroidal queries, so this comparison levels the field
the other way: data is drawn in [0.25, 0.75]^d, where no coordinate pair
can be more than 0.5 apart — toroidal L1 equals plain L1 *exactly*, and
both systems answer the same ground truth (computed by exact FAISS Flat).

Run from the repository root:
  python examples/compare_faiss_flat.py [--sizes 100000,500000,1000000]
                                        [--dims 16,32] [--backends c,rust]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ann import ToroidalNN            # noqa: E402
from src.backends import available_backends  # noqa: E402

try:
    import faiss
except ImportError:
    sys.exit("faiss not installed (pip install faiss-cpu)")

M_Q = 3_000


def bench(name, n, d, k, build, search, truth, results):
    t0 = time.perf_counter()
    index = build()
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx = search(index)
    t_query = time.perf_counter() - t0
    us = 1e6 * t_query / M_Q
    if truth is None:  # this run *is* the truth
        recall = 1.0
    else:
        hits = sum(len(np.intersect1d(idx[i][idx[i] >= 0], truth[i]))
                   for i in range(M_Q))
        recall = hits / truth.size
    results.append({"index": name, "n": n, "d": d, "k": k,
                    "build_s": round(t_build, 2), "us_per_q": round(us, 1),
                    "recall": round(recall, 3)})
    print(f"{name:>22} n={n:>8} d={d:>3} build {t_build:7.2f}s "
          f"query {us:8.1f} us/q  recall {recall:.3f}")
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100000,500000,1000000")
    ap.add_argument("--dims", default="16,32")
    ap.add_argument("--backends",
                    default=",".join(b for b in available_backends()
                                     if b != "python"))
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    dims = [int(x) for x in args.dims.split(",")]
    backends = [b for b in args.backends.split(",") if b]

    print(f"box data in [0.25, 0.75]^d (toroidal L1 == plain L1); "
          f"queries={M_Q}; faiss threads {faiss.omp_get_max_threads()}")
    results = []
    for n in sizes:
        for d in dims:
            k = 2 * d
            rng = np.random.default_rng(0)
            pts = (0.25 + 0.5 * rng.random((n, d)))
            Q = (0.25 + 0.5 * rng.random((M_Q, d)))
            pts32, Q32 = pts.astype(np.float32), Q.astype(np.float32)

            def build_flat():
                ix = faiss.IndexFlat(d, faiss.METRIC_L1)
                ix.add(pts32)
                return ix

            truth = bench("faiss Flat L1 (exact)", n, d, k, build_flat,
                          lambda ix: ix.search(Q32, k)[1], None, results)

            def build_hnsw():
                ix = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L1)
                ix.add(pts32)
                return ix

            bench("faiss HNSW32 L1", n, d, k, build_hnsw,
                  lambda ix: ix.search(Q32, k)[1], truth, results)

            for backend in backends:
                bench(f"ToroidalNN [{backend}]", n, d, k,
                      lambda b=backend: ToroidalNN(seed=1, backend=b).fit(pts, k=k),
                      lambda nn: nn.query(k=k, queries=Q)[0], truth, results)

    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "compare_faiss_flat.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults saved to {out_dir}/compare_faiss_flat.json")


if __name__ == "__main__":
    main()
