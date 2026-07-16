"""ESS main-loop simulation — the whole lifecycle, timed end to end.

The Empty-Space-Search loop this library exists for:

  1. generate A static anchors
  2. generate K candidates; for N epochs each candidate asks for its 2d
     nearest neighbours (the self-join) and then moves (selective update)
  3. after N epochs the K candidates are promoted to anchors
  4. back to 2. until the target number of explored points is reached

Two data regimes:

  --data torus (default)   uniform [0, 1)^d — the real workload. FAISS
      indexes the raw coordinates under seam-blind L1, and its recall is
      measured against exact *toroidal* truth: this is what "FAISS cannot
      answer the metric" means in numbers.
  --data box   [0.25+m, 0.75-m]^d with movement clamped inside — no
      neighbour pair wraps, toroidal L1 == plain L1 exactly, so FAISS Flat
      is exact and the comparison is pure throughput (its best case: box
      data over-fills LSH buckets).

Systems: rust / python (torann backends: fit once, selective update per
epoch, linear-merge promote) and flat (faiss IndexFlat L1 rebuilt every
epoch — add() is a memcpy, so a rebuild is the cheapest way FAISS can
track a moving tier). Per batch, recall is spot-checked on 128 candidate
queries against exact truth; the check is not timed.

Run from the repository root:
  python examples/ess_sim.py --system rust
  python examples/ess_sim.py --system flat --data box --dims 32
  python examples/ess_sim.py --system python --max-batches 1
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torann import ToroidalNN                      # noqa: E402
from torann.brute import exact_knn                 # noqa: E402

try:
    import faiss
except ImportError:
    sys.exit("faiss not installed (pip install faiss-cpu)")

R_CHECK = 128  # recall spot-check queries per batch


def exact_truth(arena, Qi, k, ex_ids, mode):
    """Exact top-k of arena rows Qi against arena, excluding self. On box
    data plain L1 == toroidal L1 (FAISS Flat computes it); on the torus
    the truth is the exact toroidal scan."""
    if mode == "torus":
        return exact_knn(arena, arena[Qi], k, exclude_ids=ex_ids)[0]
    ix = faiss.IndexFlat(arena.shape[1], faiss.METRIC_L1)
    ix.add(arena.astype(np.float32))
    idx = ix.search(arena[Qi].astype(np.float32), k + 1)[1]
    out = np.empty((len(Qi), k), dtype=np.int64)
    for r, (row, ex) in enumerate(zip(idx, ex_ids)):
        out[r] = row[row != ex][:k]
    return out


def recall_of(approx, exact):
    hits = sum(len(np.intersect1d(a[a >= 0], e)) for a, e in zip(approx, exact))
    return hits / exact.size


class FlatSim:
    """FAISS Flat driven through the ESS lifecycle: full rebuild whenever
    the arena changes, k+1 search with the self-hit filtered by id."""

    def __init__(self, d):
        self.d = d
        self.arena = np.empty((0, d))
        self.n_static = 0
        self.ix = None

    def _rebuild(self):
        self.ix = faiss.IndexFlat(self.d, faiss.METRIC_L1)
        self.ix.add(self.arena.astype(np.float32))

    def fit(self, anchors, cands):
        self.arena = np.vstack([anchors, cands])
        self.n_static = anchors.shape[0]
        self._rebuild()

    def query(self, k):
        Q = self.arena[self.n_static:].astype(np.float32)
        idx = self.ix.search(Q, k + 1)[1]
        ex = np.arange(self.n_static, self.arena.shape[0])
        out = np.empty((Q.shape[0], k), dtype=np.int64)
        for r in range(Q.shape[0]):
            row = idx[r]
            out[r] = row[row != ex[r]][:k]
        return out

    def update(self, coords):
        self.arena[self.n_static:] = coords
        self._rebuild()

    def promote(self, new_cands):
        self.n_static = self.arena.shape[0]
        if new_cands is not None and len(new_cands):
            self.arena = np.vstack([self.arena, new_cands])
        self._rebuild()


class TorannSim:
    """torann through the same lifecycle, via the public wrapper."""

    def __init__(self, d, backend):
        self.backend = backend
        self.k_hint = 2 * d
        self.nn = None

    def fit(self, anchors, cands):
        self.nn = ToroidalNN(seed=1, backend=self.backend)
        self.nn.fit(anchors, cands, k=self.k_hint)

    def query(self, k):
        return self.nn.query(k=k)[0]

    def update(self, coords):
        self.nn.update(coords)

    def promote(self, new_cands):
        self.nn.promote(new_cands)

    @property
    def arena(self):
        return self.nn._arena

    @property
    def n_static(self):
        return self.nn.n_static


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="rust",
                    choices=["rust", "python", "flat"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--anchors", type=int, default=15_000)
    ap.add_argument("--batch", type=int, default=3_000)
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--target", type=int, default=45_000)
    ap.add_argument("--dims", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--data", default="torus", choices=["torus", "box"])
    ap.add_argument("--max-batches", type=int, default=0,
                    help="stop after this many batches (0 = run to target)")
    args = ap.parse_args()

    d, k = args.dims, 2 * args.dims
    m = 4 * args.sigma  # movement margin keeps the box wrap-free
    lo, hi = 0.25 + m, 0.75 - m
    rng = np.random.default_rng(0)

    def gen(n):
        if args.data == "torus":
            return rng.random((n, d))
        return lo + (hi - lo) * rng.random((n, d))

    def move(coords):
        step = rng.normal(0.0, args.sigma, coords.shape)
        if args.data == "torus":
            return np.mod(coords + step, 1.0)
        return np.clip(coords + step, lo, hi)

    n_batches = max(1, int(np.ceil((args.target - args.anchors) / args.batch)))
    if args.max_batches:
        n_batches = min(n_batches, args.max_batches)

    sim = FlatSim(d) if args.system == "flat" else TorannSim(d, args.system)
    t = {"build": 0.0, "query": 0.0, "update": 0.0, "promote": 0.0}
    recalls = []
    t_wall = time.perf_counter()

    def timed(phase, fn, *a):
        t0 = time.perf_counter()
        out = fn(*a)
        t[phase] += time.perf_counter() - t0
        return out

    timed("build", sim.fit, gen(args.anchors), gen(args.batch))
    for b in range(n_batches):
        for e in range(args.epochs):
            idx = timed("query", sim.query, k)
            if e == 0:  # recall spot-check, not timed
                qi = np.arange(sim.n_static, sim.arena.shape[0])[:R_CHECK]
                truth = exact_truth(sim.arena, qi, k, qi, args.data)
                recalls.append(recall_of(idx[:R_CHECK], truth))
            timed("update", sim.update, move(sim.arena[sim.n_static:]))
        nxt = gen(args.batch) if b + 1 < n_batches else np.empty((0, d))
        timed("promote", sim.promote, nxt)

    wall = time.perf_counter() - t_wall
    n_epochs = n_batches * args.epochs
    row = {
        "label": args.label or f"{args.system}-{args.data}-d{d}",
        "system": args.system, "data": args.data, "d": d, "k": k,
        "anchors": args.anchors, "batch": args.batch, "epochs": args.epochs,
        "batches": n_batches, "final_points": int(sim.arena.shape[0]),
        "build_s": round(t["build"], 3), "query_s": round(t["query"], 3),
        "update_s": round(t["update"], 3),
        "promote_s": round(t["promote"], 3), "wall_s": round(wall, 3),
        "epoch_ms": round(1e3 * (t["query"] + t["update"]) / n_epochs, 2),
        "recall_per_batch": [round(r, 4) for r in recalls],
    }
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ess_sim.json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    rows.append(row)
    json.dump(rows, open(path, "w"), indent=1)
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
