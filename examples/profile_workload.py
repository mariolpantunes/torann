"""The torann calls ESS and OBLESA make, replayed without them — for profiling.

`bench_ess_loop.py` times the ESS self-join; this replays *every* call one
`ess.esa` run makes into torann, in order, with the arguments ESS's defaults
choose (ess 0.7, `search_mode="auto"`), so a profiler attached to this
process sees the library and nothing else:

  1. ``fit(anchors)``; with an attraction field, a second index over the
     anchors (``ToroidalNN().fit``), as `ess.attraction` builds it.
  2. Placement: ``query(k=1, queries=n * pool candidates)``, then
     ``promote(init)``.
  3. Per epoch: the self-join -- ``query(k)`` below d = 10, else
     ``query_radius(r, pad=True)`` with the radius that holds the neighbour
     target max(2d, 5) -- then ``weighted_directions`` over the neighbour
     list (``w = 2`` with attraction: repulsion and pull share one geometry),
     the field lookup ``query(k=8, queries=active)``, a fixed pseudo-random
     L1-capped step and ``update(active)``.

The epoch count is fixed and the step does not depend on the neighbours, so
two builds do the same work and only speed differs (the reason
`bench_ess_loop.py` exists). Two shape families:

* ``oblesa`` -- `pyBlindOpt.init.oblesa` at the ``root`` population rule
  (n_pop = ceil(10 sqrt d)): ``2 n_pop`` anchors, ``n_pop`` new points,
  ``pool = 64``, three rounds, each round's anchors grown by the previous
  round's points and their opposites. Small n: this is the brute path.
* ``ess`` -- the `bench_ess_suite.py` shapes (n up to 30 000): the LSH path.

Run from the repository root::

    python examples/profile_workload.py --family oblesa
    python examples/profile_workload.py --family ess --epochs 20 --json out/prof.json
    perf record -g -o /tmp/p.data python examples/profile_workload.py --family ess
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torann import ToroidalNN, rust

STEP_CAP = 0.02  # ess.ess.STEP_CAP: L1 travel per epoch, in radii
LOW_DIM = 10  # ess.geometry.LOW_DIM: radius mode from here up
K_LOCAL = 5  # ess.ess.K_LOCAL: default k in k-NN mode
K_ATT = 8  # ess.esa default k_att
POOL = 64  # OBLESA's k_cand, forwarded to esa as init_pool


def oblesa_shapes(dims):
    """(label, d, anchors, new, rounds) at the `root` population rule."""
    for d in dims:
        n_pop = math.ceil(10 * math.sqrt(d))
        yield f"oblesa d={d} n_pop={n_pop}", d, 2 * n_pop, n_pop, 3


def ess_shapes(dims):
    """The bench_ess_suite shapes, one esa call each."""
    table = {2: (256, 512), 8: (1024, 2048), 32: (10_000, 20_000), 100: (10_000, 20_000)}
    for d in dims:
        if d in table:
            anchors, new = table[d]
            yield f"ess d={d} a={anchors} c={new}", d, anchors, new, 1


def radius_for(points, target, rng):
    """L1 radius holding `target` neighbours: median target-th NN distance."""
    sample = points[rng.choice(len(points), min(256, len(points)), replace=False)]
    _, dist = ToroidalNN(backend="auto").fit(points).query(k=target + 1, queries=sample)
    return float(np.median(dist[:, target]))


def esa_call(anchors, n, epochs, rng, timers, attraction=True):
    """One ess.esa run's torann calls. Returns the new points."""
    d = anchors.shape[1]
    radius_mode = d >= LOW_DIM
    target = min(max(2 * d, K_LOCAL), max(1, (len(anchors) + n) // 2))
    k = min(2 * d + 1, K_LOCAL)
    radius = radius_for(np.vstack([anchors, rng.random((n, d))]), target, rng)

    def timed(name, fn, *args, **kw):
        t = time.perf_counter()
        out = fn(*args, **kw)
        timers[name] = timers.get(name, 0.0) + time.perf_counter() - t
        return out

    index = ToroidalNN(backend="auto")
    timed("fit", index.fit, anchors, k=k, radius=radius if radius_mode else None)
    field = timed("field_fit", ToroidalNN(backend="auto").fit, anchors) if attraction else None

    candidates = rng.random((n * POOL, d))
    _, novelty = timed("placement", index.query, k=1, queries=candidates)
    best = novelty.reshape(n, POOL).argmax(axis=1)
    active = candidates.reshape(n, POOL, d)[np.arange(n), best].copy()
    timed("promote", index.promote, active)
    all_data = np.vstack([anchors, active])

    wd = rust.weighted_directions
    for _ in range(epochs):
        if radius_mode:
            ids, dists = timed("self_join", index.query_radius, radius, pad=True)
        else:
            ids, dists = timed("self_join", index.query, k=k)
        weights = np.where(ids >= 0, 1.0 / np.maximum(dists, 1e-9), 0.0)
        stacked = np.stack((weights, weights)) if attraction else weights[None]
        if wd is not None:
            timed("forces", wd, all_data, active, ids, stacked)
        if field is not None:
            timed("field_query", field.query, k=min(K_ATT, len(anchors)), queries=active)
        step = rng.standard_normal(active.shape)
        step *= STEP_CAP * radius / np.abs(step).sum(axis=1, keepdims=True)
        active = np.mod(active + step, 1.0)
        all_data[len(anchors) :] = active
        timed("update", index.update, active)
    return active, type(index._impl).__name__ if index._impl is not None else "?"


def run(shapes, epochs, seed):
    rows = []
    for label, d, n_anchor, n_new, rounds in shapes:
        rng = np.random.default_rng(seed)
        anchors = rng.random((n_anchor, d))
        timers: dict[str, float] = {}
        t0 = time.perf_counter()
        impl = "?"
        for _ in range(rounds):
            new, impl = esa_call(anchors, n_new, epochs, rng, timers)
            # OBLESA measures the new points and their opposites; both become anchors.
            anchors = np.vstack([anchors, new, np.mod(1.0 - new, 1.0)])
        total = time.perf_counter() - t0
        rows.append({"shape": label, "d": d, "impl": impl, "total_s": total, **timers})
        parts = "  ".join(f"{k}={v * 1e3:.1f}" for k, v in sorted(timers.items(), key=lambda kv: -kv[1]))
        print(f"{label:<34} {impl:<14} total {total * 1e3:8.1f} ms | {parts}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Replay the torann calls of ESS and OBLESA for profiling.")
    ap.add_argument("--family", choices=["oblesa", "ess", "both"], default="both")
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 8, 10, 32, 100])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write the rows here")
    args = ap.parse_args()
    shapes = []
    if args.family in ("oblesa", "both"):
        shapes += list(oblesa_shapes(args.dims))
    if args.family in ("ess", "both"):
        shapes += list(ess_shapes(args.dims))
    rows = run(shapes, args.epochs, args.seed)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"native": rust.AVAILABLE, "epochs": args.epochs, "rows": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
