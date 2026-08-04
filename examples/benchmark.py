"""The standing benchmark: dimension sweep x start x search mode x engine.

One entry point for "is this change good?", covering the axes that were each
measured separately during the query-path work:

* **dimension sweep** — `d = 2 … 64`. The cost structure changes
  qualitatively across it: below `d = 8` the eight-lane distance kernel has
  no full vector to fill, and above `d ~ 16` concentration of measure makes
  every candidate roughly equidistant, which is what kills bucket pruning
  (measured: the cost is locality and dependency latency, not pruning).
* **start** — `empty` is the from-scratch first batch (no anchors, no
  `_smart_init`); `filled` has a static tier, so it pays for initialization
  against existing points, which is the re-exploration case ESS exists for.
* **search mode** — `knn` and `radius`. Radius mode had *no* measurements at
  all before this file, while being a first-class
  `ess.esa` argument.
* **engine** — `brute` (exact, and the ground truth recall is measured
  against) and `lsh`, forced with `brute_threshold` so the same `n` runs both
  ways; `auto` reports the size-based choice a caller actually gets.

Wall time per *run* is a poor metric here: under ESS's plateau stop the epoch
count swings ±40% with the seed, so a run can look 1.5× faster for reasons
that have nothing to do with the code. Everything is therefore reported **per
epoch**, over a fixed epoch count.

Quality is toroidal Clark-Evans and toroidal-L1 separation — the two metrics
that hold at every dimension (`bench_ess_quality.py` documents why coverage
and fill distance are not here). Both are implemented locally so this file
runs without `ess`; when `ess` *is* importable they are cross-checked against
`ess.utils` on the first call, which keeps the two copies honest.

The full default sweep is 48 configurations and takes minutes, most of it in
the forced-`brute` cells at high `d` — narrow it while iterating.

Run from the repository root::

    python examples/benchmark.py                      # the standard sweep
    python examples/benchmark.py --dims 2 8 32 --n 4096
    python examples/benchmark.py --modes radius --engines lsh
    python examples/benchmark.py --driver mimic       # ignore ess if present
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torann import ToroidalNN  # noqa: E402
from torann.brute import exact_knn, exact_radius  # noqa: E402

try:
    import ess
    import ess.utils
    HAVE_ESS = True
except ImportError:  # pragma: no cover - the benchmark runs standalone too
    HAVE_ESS = False

OUT = os.path.join(os.path.dirname(__file__), "out")

DIMS = (2, 4, 8, 16, 32, 64)
K = 5                      # ess.K_LOCAL: what the force kernel asks for
EPOCHS = 30
LR = 0.2
STEP_CAP = 0.02
RECALL_QUERIES = 128
_CHECKED: list[bool] = []


# --------------------------------------------------------------------- #
# Metrics — local, so this file has no hard dependency on ess
# --------------------------------------------------------------------- #

def expected_nn(n: int, dim: int) -> float:
    r"""Mean nearest-neighbour toroidal-$L_1$ distance of $n$ uniform points.

    Exact for fixed $n$, not the Poisson asymptotic
    $\Gamma(1+1/d)(d!/n)^{1/d}/2$ — which reads 5–14% high above $d = 16$,
    because by then the $L_1$ ball has wrapped around every coordinate and
    neither $\exp(-nV)$ nor $V(t) = (2t)^d/d!$ holds. A toroidal coordinate
    distance to a uniform point is $U(0, 1/2)$, so the distance is an
    Irwin-Hall variable halved and

    $$ \mathbb{E}[R] = \int_0^{d/2} (1 - V(t))^{n-1}\,dt $$

    with $V$ obtained by convolving the one-coordinate density $d$ times via
    FFT — stable where the closed-form alternating sum is not.

    Args:
        n: Number of points.
        dim: Dimensionality.

    Returns:
        The expected nearest-neighbour distance under uniformity.
    """
    poisson = (math.gamma(1.0 + 1.0 / dim)
               * math.exp((math.lgamma(dim + 1) - math.log(n)) / dim) / 2.0)
    step = min(2.0e-4, poisson / 400.0)
    m = int(round(dim * 0.5 / step)) + 1
    size = 1 << int(math.ceil(math.log2(2 * m)))
    half = int(round(0.5 / step)) + 1
    dens = np.zeros(size)
    dens[:half] = 2.0                    # U(0, 1/2) has density 2
    dens[0] = dens[half - 1] = 1.0       # trapezoid end weights
    dens *= step
    pdf = np.fft.irfft(np.fft.rfft(dens) ** dim, size)[:m]
    cdf = np.clip(np.cumsum(pdf), 0.0, 1.0)
    return float(np.trapezoid(np.power(1.0 - cdf, n - 1), dx=step))


def quality(points: np.ndarray) -> tuple[float, float]:
    """Toroidal Clark-Evans and separation, from one exact k-NN scan.

    Args:
        points: ``(n, d)`` design in ``[0, 1)``.

    Returns:
        ``(clark_evans, separation)``, both in toroidal L1.
    """
    pts = np.mod(np.ascontiguousarray(points, dtype=np.float64), 1.0)
    n, dim = pts.shape
    if n < 2:
        return 0.0, 0.0
    nn = exact_knn(pts, pts, 2)[1][:, 1]
    ce = float(nn.mean() / expected_nn(n, dim))
    sep = float(nn.min())
    if HAVE_ESS and not _CHECKED:        # the two copies must agree
        _CHECKED.append(True)
        assert abs(ce - ess.utils.toroidal_clark_evans(pts)) < 1e-9, "CE drift"
        assert abs(sep - ess.utils.toroidal_separation(pts)) < 1e-12
    return ce, sep


def recall(index: ToroidalNN, mode: str, k: int, radius: float,
           seed: int = 0) -> float:
    """Fraction of the exact answer the index returns, self excluded.

    Args:
        index: Index as left by the run.
        mode: ``"knn"`` or ``"radius"``.
        k: Neighbours per query in k-NN mode.
        radius: Cutoff in radius mode.
        seed: Sub-sample seed.

    Returns:
        Recall in ``[0, 1]``, or nan when the index is exact anyway.
    """
    if not index.is_approximate:
        return float("nan")
    arena = index._arena
    rng = np.random.default_rng(seed)
    take = min(RECALL_QUERIES, arena.shape[0])
    rows = rng.choice(arena.shape[0], take, replace=False)
    q = np.ascontiguousarray(arena[rows])
    ids = np.ascontiguousarray(rows, dtype=np.int64)
    if mode == "knn":
        got, _ = index.query(k=k, queries=q, exclude_ids=ids)
        want, _ = exact_knn(arena, q, k, exclude_ids=ids)
        hits = sum(len(set(got[i].tolist()) & set(want[i].tolist()))
                   for i in range(take))
        return hits / float(take * k)
    approx = index.query_radius(radius, queries=q)
    indptr, want_ids, _ = exact_radius(arena, q, radius, exclude_ids=ids)
    hits = total = 0
    for i in range(take):
        want = set(want_ids[indptr[i]:indptr[i + 1]].tolist()) - {int(ids[i])}
        got = set(approx[i][0].tolist()) - {int(ids[i])}
        hits += len(want & got)
        total += len(want)
    return hits / float(total) if total else float("nan")


# --------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------- #

def radius_for(dim: int, n: int, target: int = K) -> float:
    r"""Interaction radius holding ``target`` neighbours on average.

    Mirrors `ess._l1_radius_heuristic`: solve $n(2r)^d/d! = $ ``target``
    while the $L_1$ ball still fits inside the torus, else fall back to the
    normal approximation of the coordinate sum.

    Args:
        dim: Dimensionality.
        n: Points in the index.
        target: Desired neighbour count.

    Returns:
        Radius in toroidal L1 units.
    """
    count = max(min(target, max(n - 1, 1)), 1)
    dense = 0.5 * math.exp(
        (math.lgamma(dim + 1) + math.log(count) - math.log(max(n, 2))) / dim)
    if dense <= 0.5:
        return dense
    z = statistics.NormalDist().inv_cdf(count / max(n, 2))
    return min(max(dim / 4.0 + z * math.sqrt(dim / 48.0), 1e-6), dim / 2.0)


def _pad(results: list) -> tuple[np.ndarray, np.ndarray]:
    """Pack ragged radius results into dense ``(m, w)`` arrays."""
    width = max(max((r[0].shape[0] for r in results), default=0), 1)
    ids = np.full((len(results), width), -1, dtype=np.int64)
    dst = np.full((len(results), width), np.inf)
    for i, (row_ids, row_dst) in enumerate(results):
        ids[i, :row_ids.shape[0]] = row_ids
        dst[i, :row_dst.shape[0]] = row_dst
    return ids, dst


def _forces(active, arena, ids, dists, radius):
    """Toroidal inverse-square repulsion from the returned neighbours.

    A stand-in for `ess._compute_forces`, not a reimplementation of it: the
    job is to move candidates the way ESS does, so the index sees a realistic
    sequence of `update` calls. Direction is the shortest way round the
    torus.

    Args:
        active: ``(m, d)`` candidate coordinates.
        arena: ``(n, d)`` every indexed point.
        ids: ``(m, w)`` neighbour ids, ``-1`` where missing.
        dists: ``(m, w)`` neighbour distances.
        radius: Force scale.

    Returns:
        ``(m, d)`` force vectors.
    """
    valid = ids >= 0
    nb = arena[np.where(valid, ids, 0)]
    delta = active[:, None, :] - nb
    delta -= np.round(delta)                       # shortest way round
    dd = np.maximum(dists, 1e-9)[:, :, None]
    w = np.where(valid[:, :, None], (radius / dd) ** 2, 0.0)
    return (delta / dd * w).sum(axis=1)


def run(dim: int, n: int, start: str, mode: str, engine: str,
        epochs: int = EPOCHS, seed: int = 0, driver: str = "auto") -> dict:
    """One configuration, driven through ESS when it is importable.

    Args:
        dim: Dimensionality.
        n: Candidate points to place.
        start: ``"empty"`` (no anchors) or ``"filled"`` (``n`` anchors).
        mode: ``"knn"`` or ``"radius"``.
        engine: ``"auto"``, ``"brute"`` or ``"lsh"``.
        epochs: Fixed epoch count — comparable across configurations.
        seed: Seed for the sample, the index and the driver.
        driver: ``"auto"``, ``"ess"`` or ``"mimic"``.

    Returns:
        Timings per epoch, quality, recall and index state.
    """
    rng = np.random.default_rng(seed)
    anchors = n if start == "filled" else 0
    A = rng.random((anchors, dim)) if anchors else np.empty((0, dim))
    radius = radius_for(dim, anchors + n)
    thresh = {"brute": 1 << 30, "lsh": 0}.get(engine)
    index = ToroidalNN(seed=seed, brute_threshold=thresh)
    use_ess = HAVE_ESS and driver in ("auto", "ess")
    keys = ("setup_s", "query_s", "force_s", "step_s", "update_s")

    t0 = time.perf_counter()
    if use_ess:
        stats: dict = {}
        pts = ess.esa(A, np.array([[0.0, 1.0]] * dim), n=n, index=index,
                      seed=seed, stats=stats, k=K, radius=radius,
                      search_mode="k_nn" if mode == "knn" else "radius",
                      epochs=epochs, patience=epochs + 1)
        total = time.perf_counter() - t0
        used = stats.get("epochs_total") or epochs
        phases = {p: stats.get(p, 0.0) for p in keys}
        full = np.vstack([A, pts]) if anchors else pts
        fanout = float("nan")
    else:
        init = rng.random((n, dim))
        index.fit(A if anchors else np.empty((0, dim)), init, k=K,
                  radius=radius)
        arena = index._arena
        active = arena[anchors:]
        phases = dict.fromkeys(keys, 0.0)
        lr, fan = LR, []
        for _ in range(epochs):
            t = time.perf_counter()
            if mode == "knn":
                ids, dists = index.query(k=K)
            else:
                res = index.query_radius(radius)
                fan.append(float(np.mean([r[0].shape[0] for r in res])))
                ids, dists = _pad(res)
            phases["query_s"] += time.perf_counter() - t

            t = time.perf_counter()
            f = _forces(active, arena, ids, dists, radius)
            phases["force_s"] += time.perf_counter() - t

            t = time.perf_counter()
            step = f * (lr * radius)
            norm = np.abs(step).sum(axis=1, keepdims=True)
            step *= np.minimum(1.0, STEP_CAP * radius / np.maximum(norm, 1e-12))
            active += step
            np.mod(active, 1.0, out=active)
            phases["step_s"] += time.perf_counter() - t

            t = time.perf_counter()
            index.update(active)
            phases["update_s"] += time.perf_counter() - t
            lr *= 0.98
        total = time.perf_counter() - t0
        used, full = epochs, arena
        fanout = float(np.mean(fan)) if fan else float("nan")

    ce, sep = quality(full)
    return {"dim": dim, "n": n, "start": start, "mode": mode,
            "engine": engine, "driver": "ess" if use_ess else "mimic",
            "epochs": used, "total_s": total,
            "ms_per_epoch": 1000.0 * total / max(used, 1),
            "query_ms_per_epoch": 1000.0 * phases["query_s"] / max(used, 1),
            **phases,
            "impl": "lsh" if index.is_approximate else "brute",
            "backend": index.backend_name or "brute",
            "tables": index.n_tables, "radius": radius, "fanout": fanout,
            "recall": recall(index, mode, K, radius),
            "clark_evans": ce, "separation": sep,
            "fingerprint": int(np.asarray(full * 1e9, dtype=np.int64).sum())}


def table(rows: list[dict]) -> None:
    """Print one line per configuration."""
    head = ["d", "n", "start", "mode", "engine", "impl", "L", "ms/epoch",
            "query ms/ep", "q share", "recall", "CE", "separation", "fanout"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        share = (100.0 * r["query_ms_per_epoch"] / r["ms_per_epoch"]
                 if r["ms_per_epoch"] else 0.0)
        print("| " + " | ".join([
            str(r["dim"]), str(r["n"]), r["start"], r["mode"], r["engine"],
            r["impl"], str(r["tables"]),
            f"{r['ms_per_epoch']:.2f}", f"{r['query_ms_per_epoch']:.2f}",
            f"{share:.0f}%",
            "exact" if np.isnan(r["recall"]) else f"{r['recall']:.3f}",
            f"{r['clark_evans']:.4f}", f"{r['separation']:.4f}",
            "-" if np.isnan(r["fanout"]) else f"{r['fanout']:.1f}",
        ]) + " |")


def group(rows: list[dict], label: str = "suite") -> None:
    """Aggregate a set of configurations: a change must win on the whole."""
    if not rows:
        return
    q = sum(r["query_ms_per_epoch"] for r in rows)
    t = sum(r["ms_per_epoch"] for r in rows)
    geo = statistics.geometric_mean([max(r["ms_per_epoch"], 1e-6)
                                     for r in rows])
    print(f"{label}: {len(rows):2} configs, {t:9.1f} ms/epoch summed "
          f"({100 * q / t:3.0f}% query), geometric mean {geo:8.2f} ms/epoch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    ap.add_argument("--n", type=int, default=2048,
                    help="candidates per run (anchors match it when filled)")
    ap.add_argument("--starts", nargs="+", default=["empty", "filled"],
                    choices=["empty", "filled"])
    ap.add_argument("--modes", nargs="+", default=["knn", "radius"],
                    choices=["knn", "radius"])
    ap.add_argument("--engines", nargs="+", default=["brute", "lsh"],
                    choices=["auto", "brute", "lsh"])
    ap.add_argument("--driver", default="auto",
                    choices=["auto", "ess", "mimic"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=os.path.join(OUT, "benchmark.json"))
    args = ap.parse_args()

    driver = "ess" if HAVE_ESS and args.driver != "mimic" else "mimic"
    print(f"driver: {driver}  epochs: {args.epochs}  n: {args.n}\n")
    rows = []
    for dim in args.dims:
        for start in args.starts:
            for mode in args.modes:
                for engine in args.engines:
                    r = run(dim, args.n, start, mode, engine,
                            epochs=args.epochs, seed=args.seed,
                            driver=args.driver)
                    rows.append(r)
                    print(f"  [d={dim:<3} {start:<6} {mode:<6} {engine:<5} "
                          f"{r['ms_per_epoch']:9.2f} ms/epoch  "
                          f"recall {r['recall']:.3f}]", flush=True)

    print()
    table(rows)
    print()
    for engine in args.engines:
        group([r for r in rows if r["engine"] == engine],
              f"engine={engine:<6}")
    for mode in args.modes:
        group([r for r in rows if r["mode"] == mode], f"mode={mode:<8}")
    group(rows, "all          ")

    os.makedirs(os.path.dirname(args.json) or OUT, exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n[saved {args.json}]")
