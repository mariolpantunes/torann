"""Paired A/B benchmark across compiled builds: the instrument every change is judged by.

`bench_ess_suite.py` and `benchmark.py` measure one build. A change has to be
judged against another build, and on this host that comparison is easy to get
wrong in three ways, each of which has already produced a retracted number:

* **Across sessions.** Measured hours apart this machine drifts 20-29% under
  sustained load (`.cargo/config.toml`), which swamps most effects. So arms
  are interleaved inside one session, ABBA, pair by pair.
* **Against the wrong build.** ESS imports the *installed* torann, and an
  editable install points at whatever `.so` was last built. Every worker
  asserts that the `torann` it imported, and the hash of its extension, are
  the arm it was launched for.
* **As a single ratio.** One run per arm cannot tell an effect from noise.
  Each case is repeated in pairs, and the report gives the median paired ratio
  with a bootstrap interval, next to an A/A run of the control against itself
  that says how wide "no change" is on that shape.

Arms are wheels unpacked into a directory (``build``); cases run in fresh
subprocesses with that directory first on ``PYTHONPATH`` (``run``); the JSON is
compared per case (``report``). Every case records latency, peak RSS, the
native ``stats()``, recall where it applies, and a fingerprint of its output,
so a change declared bit-identical can be held to it.

The workload matrix::

    W1  ESS loop, k-NN     the six `bench_ess_suite.py` shapes, both query paths
    W2  ESS loop, radius   the same shapes in radius mode, both paths
    W3  standing sweep     d x empty/filled x knn/radius x brute/lsh, per epoch
    W4  kernels            brute_knn / brute_radius / weighted_directions, and
                           the LSH query over d x k x m x path
    W5  lifecycle          fit / update / promote and peak RSS
    W6  python backend     the pure-NumPy reference, for changes to lsh.py
    WQ  quality            full-length ESS runs scored with the dispersion panel

Run from the repository root (``ess`` on the path)::

    python examples/bench_ab.py build base --rev b06c7df
    python examples/bench_ab.py build work
    python examples/bench_ab.py run --arms base work --suites W1 W4 --pairs 7 \\
        --out examples/out/ab_work.json
    python examples/bench_ab.py report examples/out/ab_work.json --aa examples/out/ab_aa.json
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARMS = os.environ.get("TORANN_ARMS", os.path.join(HERE, "out", "arms"))

ESS_SHAPES = ((2, 0, 256), (2, 256, 512), (8, 0, 1024), (8, 1024, 2048),
              (32, 0, 10000), (32, 10000, 20000))
LOOP_EPOCHS = 20
SWEEP_DIMS = (2, 8, 16, 32, 64, 100)
SWEEP_N = 2048
SWEEP_EPOCHS = 10
LSH_N = 16384
RECALL_QUERIES = 128


# --------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------- #

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _native_so(pkg_dir):
    for name in os.listdir(pkg_dir):
        if name.startswith("_native") and name.endswith(".so"):
            return os.path.join(pkg_dir, name)
    return None


def build_arm(name, rev=None, rustflags=None, profile=None, features=None):
    """Build a release wheel of `rev` (default: the working tree) into an arm.

    A revision is checked out into a temporary worktree so the working tree is
    never touched. The cargo target directory is shared between arms, so only
    what changed is recompiled. `rustflags` and `profile` feed PGO arms;
    `features` builds an instrumented arm (``stats``) for profiling -- never
    for timing against a release arm, which has no instrumentation.
    """
    dest = os.path.join(ARMS, name)
    src = ROOT
    tmp = None
    if rev:
        tmp = tempfile.mkdtemp(prefix=f"arm-{name}-")
        subprocess.run(["git", "-C", ROOT, "worktree", "add", "--detach", tmp, rev],
                       check=True)
        src = tmp
    env = dict(os.environ)
    env.setdefault("CARGO_TARGET_DIR", os.path.join(ARMS, "_target"))
    if rustflags:
        env["RUSTFLAGS"] = rustflags
    wheels = tempfile.mkdtemp(prefix="wheels-")
    try:
        cmd = ["maturin", "build", "--release", "-o", wheels]
        if features:
            cmd += ["--features", features]
        subprocess.run(cmd, cwd=src, env=env, check=True)
        (wheel,) = [f for f in os.listdir(wheels) if f.endswith(".whl")]
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest)
        with zipfile.ZipFile(os.path.join(wheels, wheel)) as zf:
            zf.extractall(dest)
        # Resolved in the real repository, not in the temporary worktree: on a
        # shared filesystem the worktree registration can be pulled out from
        # under a running build (an rsync --delete of `.git` will do it), and
        # the arm's identity must not depend on that.
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", rev or "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT if rev else src, "status", "--porcelain",
                                "src", "torann", "Cargo.toml", ".cargo"], capture_output=True,
                               text=True, check=False).stdout.strip() if not rev else ""
        so = _native_so(os.path.join(dest, "torann"))
        meta = {"name": name, "rev": rev or "worktree", "head": head, "dirty": bool(dirty),
                "so_sha256": _sha(so), "rustflags": rustflags, "profile": profile,
                "features": features,
                "built": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(dest, "arm.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print(json.dumps(meta, indent=1))
    finally:
        shutil.rmtree(wheels, ignore_errors=True)
        if tmp:
            subprocess.run(["git", "-C", ROOT, "worktree", "remove", "--force", tmp], check=False)


def _check_arm(arm_dir):
    """Refuse to measure anything but the arm we were launched for."""
    import torann
    with open(os.path.join(arm_dir, "arm.json")) as fh:
        meta = json.load(fh)
    pkg = os.path.dirname(os.path.realpath(torann.__file__))
    if pkg != os.path.realpath(os.path.join(arm_dir, "torann")):
        raise SystemExit(f"arm mismatch: imported {pkg}, expected {arm_dir}")
    so = _native_so(pkg)
    if so is None or _sha(so) != meta["so_sha256"]:
        raise SystemExit(f"extension hash mismatch in {pkg}")
    return meta


# --------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------- #

def _fp(*arrays):
    import numpy as np
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _stats(index):
    impl = getattr(index, "_impl", index)
    get = getattr(impl, "stats", None)
    return dict(get()) if get else {}


def _timeit(fn, min_time=0.25, min_reps=5, max_reps=200):
    """Median seconds of `fn` after one warm-up, and its last return value."""
    out = fn()
    times = []
    start = time.perf_counter()
    while len(times) < min_reps or (time.perf_counter() - start < min_time and len(times) < max_reps):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), out


def _knn_recall(arena, got_fn, k, seed=0):
    import numpy as np

    from torann.brute import exact_knn
    rng = np.random.default_rng(seed)
    take = min(RECALL_QUERIES, arena.shape[0])
    rows = rng.choice(arena.shape[0], take, replace=False)
    q = np.ascontiguousarray(arena[rows])
    ids = np.ascontiguousarray(rows, dtype=np.int64)
    got, _ = got_fn(q, k, ids)
    want, _ = exact_knn(arena, q, k, exclude_ids=ids)
    return sum(len(set(got[i].tolist()) & set(want[i].tolist())) for i in range(take)) / (take * k)


def case_ess(dim, anchors, cands, mode, block, epochs=LOOP_EPOCHS, full=False):
    """W1/W2/WQ: one ESS run through a ToroidalNN."""
    import ess
    import ess.utils
    import numpy as np

    from torann import ToroidalNN
    sys.path.insert(1, HERE)
    import benchmark as bm
    rng = np.random.default_rng(0)
    A = rng.random((anchors, dim)) if anchors else np.empty((0, dim))
    index = ToroidalNN(seed=0, query_block_size=block)
    stats = {}
    # Timing runs use a fixed epoch count, so per-epoch cost is compared at
    # equal work; quality runs (`full`) let the plateau detector stop.
    run_epochs, patience = (1024, 25) if full else (epochs, epochs + 1)
    t0 = time.perf_counter()
    pts = ess.esa(A, np.array([[0.0, 1.0]] * dim), n=cands, index=index, seed=0, stats=stats,
                  search_mode="k_nn" if mode == "knn" else "radius",
                  epochs=run_epochs, patience=patience)
    total = time.perf_counter() - t0
    used = max(1, stats.get("epochs_total") or epochs)
    row = {"total_s": total, "epochs": used,
           "ms_per_epoch": 1e3 * (total - stats.get("setup_s", 0.0)) / used,
           "query_ms_per_epoch": 1e3 * stats.get("query_s", 0.0) / used,
           "update_ms_per_epoch": 1e3 * stats.get("update_s", 0.0) / used,
           "setup_s": stats.get("setup_s", 0.0),
           "impl": "lsh" if index.is_approximate else "brute",
           "native": _stats(index), "fingerprint": _fp(pts)}
    if index.is_approximate:
        k = stats.get("k") or 5
        radius = bm.radius_for(dim, index.n_points, k)
        row["recall"] = bm.recall(index, mode, k, radius)
    if full:
        full_pts = np.vstack([A, pts]) if anchors else pts
        row["projection_discrepancy"] = ess.utils.projection_discrepancy(full_pts)
        row["wrap_around_discrepancy"] = ess.utils.wrap_around_discrepancy(full_pts)
    return row


def case_sweep(dim, start, mode, engine):
    """W3: `benchmark.run`, fixed epoch count, per epoch."""
    sys.path.insert(1, HERE)
    import benchmark as bm
    r = bm.run(dim, SWEEP_N, start, mode, engine, epochs=SWEEP_EPOCHS, seed=0)
    return {"ms_per_epoch": r["ms_per_epoch"], "query_ms_per_epoch": r["query_ms_per_epoch"],
            "recall": r["recall"], "impl": r["impl"], "fingerprint": str(r["fingerprint"]),
            "separation": r["separation"]}


def case_brute(kernel, n, d, k=5):
    """W4: the three brute kernels on the `.cargo/config.toml` shapes."""
    import numpy as np

    from torann import rust
    sys.path.insert(1, HERE)
    import benchmark as bm
    if rust.brute_knn is None or rust.brute_radius is None or rust.weighted_directions is None:
        raise RuntimeError("arm has no compiled brute kernels")
    brute_knn, brute_radius, directions = rust.brute_knn, rust.brute_radius, rust.weighted_directions
    rng = np.random.default_rng(0)
    pts = np.ascontiguousarray(rng.random((n, d)))
    ex = np.arange(n, dtype=np.int64)
    if kernel == "brute_knn":
        sec, out = _timeit(lambda: brute_knn(pts, pts, k, ex))
    elif kernel == "brute_radius":
        r = bm.radius_for(d, n, k)
        sec, out = _timeit(lambda: brute_radius(pts, pts, r, ex))
    else:
        ids, _ = brute_knn(pts, pts, k, ex)
        w = np.ascontiguousarray(rng.random((2, n, k)))
        sec, out = _timeit(lambda: (directions(pts, pts, ids, w),))
    return {"us_per_call": 1e6 * sec, "fingerprint": _fp(*out)}


_LSH = {}


def _lsh_index(d):
    """A two-tier LSH index on `LSH_N` points, forced past the brute threshold."""
    import numpy as np

    from torann import ToroidalNN
    if d not in _LSH:
        rng = np.random.default_rng(d)
        pts = rng.random((LSH_N, d))
        nn = ToroidalNN(seed=0, brute_threshold=0)
        nn.fit(pts[: LSH_N // 2], pts[LSH_N // 2:], k=2 * d)
        _LSH[d] = nn
    return _LSH[d]


def case_lsh(op, d, k, m, block):
    """W4: one LSH query shape; `m = 0` is the ESS self-join over candidates."""
    import numpy as np
    sys.path.insert(1, HERE)
    import benchmark as bm
    nn = _lsh_index(d)
    if m:
        rng = np.random.default_rng(1000 + m)
        rows = rng.choice(nn.n_points, m, replace=m > nn.n_points)
        q = np.ascontiguousarray(nn._arena[rows])
        ex = np.ascontiguousarray(rows, dtype=np.int64)
    else:
        q = np.ascontiguousarray(nn._arena[nn.n_static:])
        ex = np.arange(nn.n_static, nn.n_points, dtype=np.int64)
    # Rebuild the native index with this block size: `block` is fixed at
    # construction on builds that predate the setter.
    from torann import rust
    if rust.RustLshIndex is None:
        raise RuntimeError("arm has no compiled LSH index")
    native = rust.RustLshIndex(nn._B, nn._K, nn._L, nn.probes, nn._S, nn._U, block)
    native.build(np.ascontiguousarray(nn._arena), nn.n_static)
    row = {}
    if op == "knn":
        sec, out = _timeit(lambda: native.query_knn(q, k, ex))
        row["recall"] = _knn_recall(nn._arena, lambda qq, kk, ii: native.query_knn(qq, kk, ii), min(k, 64))
    else:
        r = bm.radius_for(d, nn.n_points, k)
        sec, out = _timeit(lambda: native.query_radius(q, r, ex))
    if hasattr(native, "stats"):   # instrumented builds only
        native.reset_stats()
        if op == "knn":
            native.query_knn(q, k, ex)
        else:
            native.query_radius(q, r, ex)
        row["native"] = dict(native.stats())
    row.update({"us_per_query": 1e6 * sec / q.shape[0], "ms_per_call": 1e3 * sec,
                "fingerprint": _fp(*out)})
    return row


def case_lifecycle(d, n):
    """W5: fit, update and promote cost, and the process's peak RSS."""
    import numpy as np

    from torann import ToroidalNN
    rng = np.random.default_rng(d + n)
    pts = rng.random((n, d))
    half = n // 2

    def fit():
        nn = ToroidalNN(seed=0, brute_threshold=0)
        nn.fit(pts[:half], pts[half:], k=2 * d)
        return nn

    fit_s, nn = _timeit(fit, min_time=0.0, min_reps=3, max_reps=3)
    drift = np.mod(pts[half:] + rng.normal(0.0, 0.01, (n - half, d)), 1.0)
    upd_s, _ = _timeit(lambda: nn.update(drift), min_time=0.0, min_reps=5, max_reps=5)
    q_s, (ids, dst) = _timeit(lambda: nn.query(k=2 * d), min_time=0.0, min_reps=3, max_reps=3)
    batch = rng.random((max(1, n // 8), d))
    prom = []
    for _ in range(3):
        cur = fit()
        t = time.perf_counter()
        cur.promote(batch)
        prom.append(time.perf_counter() - t)
    return {"fit_ms": 1e3 * fit_s, "update_ms": 1e3 * upd_s, "query_ms": 1e3 * q_s,
            "promote_ms": 1e3 * statistics.median(prom),
            "fingerprint": _fp(ids, dst, *[np.asarray(v) for v in nn._impl.tables().values()])}


def case_python(d, mode):
    """W6: the pure-NumPy reference backend."""
    import numpy as np

    from torann import ToroidalNN
    sys.path.insert(1, HERE)
    import benchmark as bm
    rng = np.random.default_rng(d)
    pts = rng.random((4096, d))
    nn = ToroidalNN(seed=0, brute_threshold=0, backend="python")
    nn.fit(pts[:2048], pts[2048:], k=2 * d)
    if mode == "knn":
        sec, out = _timeit(lambda: nn.query(k=2 * d), min_time=0.5, min_reps=3)
    else:
        r = bm.radius_for(d, 4096, 2 * d)
        sec, out = _timeit(lambda: nn.query_radius(r, pad=True), min_time=0.5, min_reps=3)
    return {"ms_per_call": 1e3 * sec, "fingerprint": _fp(*out)}


def cases():
    """Every case id, mapped to (callable, kwargs). Ids sort into suites."""
    reg = {}
    for dim, a, c in ESS_SHAPES:
        for mode, suite in (("knn", "W1"), ("radius", "W2")):
            for block, path in ((1, "perquery"), (None, "batched")):
                reg[f"{suite}/d{dim}-a{a}-c{c}/{path}"] = (
                    case_ess, {"dim": dim, "anchors": a, "cands": c, "mode": mode, "block": block})
        reg[f"WQ/d{dim}-a{a}-c{c}/knn"] = (
            case_ess, {"dim": dim, "anchors": a, "cands": c, "mode": "knn", "block": None, "full": True})
    for dim in SWEEP_DIMS:
        for start in ("empty", "filled"):
            for mode in ("knn", "radius"):
                for engine in ("brute", "lsh"):
                    reg[f"W3/d{dim}-{start}-{mode}-{engine}"] = (
                        case_sweep, {"dim": dim, "start": start, "mode": mode, "engine": engine})
    for kernel in ("brute_knn", "brute_radius", "weighted_directions"):
        for n, d in ((400, 8), (400, 100), (4096, 8), (4096, 100)):
            reg[f"W4/{kernel}/n{n}-d{d}"] = (case_brute, {"kernel": kernel, "n": n, "d": d})
    reg["W4/brute_knn/n4096-d100-k64"] = (case_brute, {"kernel": "brute_knn", "n": 4096, "d": 100, "k": 64})
    for d in (8, 32, 100):
        for m in (20, 63, 64, 1024, 10000, 0):
            for k in sorted({5, 2 * d, 200}):
                for block, path in ((1, "perquery"), (0, "batched")):
                    reg[f"W4/lsh_knn/d{d}-k{k}-m{m}/{path}"] = (
                        case_lsh, {"op": "knn", "d": d, "k": k, "m": m, "block": block})
            for block, path in ((1, "perquery"), (0, "batched")):
                reg[f"W4/lsh_radius/d{d}-t5-m{m}/{path}"] = (
                    case_lsh, {"op": "radius", "d": d, "k": 5, "m": m, "block": block})
        for block, path in ((1, "perquery"), (0, "batched")):
            reg[f"W4/lsh_knn/d{d}-k512-m1024/{path}"] = (
                case_lsh, {"op": "knn", "d": d, "k": 512, "m": 1024, "block": block})
    for d in (8, 32):
        for n in (20000, 100000, 1000000):
            reg[f"W5/d{d}-n{n}"] = (case_lifecycle, {"d": d, "n": n})
    for d in (8, 32):
        for mode in ("knn", "radius"):
            reg[f"W6/d{d}-{mode}"] = (case_python, {"d": d, "mode": mode})
    return reg


# One subprocess per case, except the cheap W4 LSH cases, which share a
# process per dimension so the fixture is built once.
def _batches(ids):
    out, lsh = [], {}
    for cid in ids:
        m = re.match(r"W4/lsh_\w+/d(\d+)-", cid)
        if m:
            lsh.setdefault(m.group(1), []).append(cid)
        else:
            out.append([cid])
    return out + list(lsh.values())


def _host():
    """Host and CPU model of this machine, recorded on every row.

    A paired comparison is only valid within one machine, and the effects
    this harness chases are micro-architectural (see the alignment finding
    of 2026-09-17), so a merged run from several cluster nodes has to be
    able to prove that both arms of a case met on the same CPU.
    """
    model = ""
    try:
        with open("/proc/cpuinfo", encoding="ascii", errors="ignore") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"host": os.uname().nodename, "cpu": model}


def worker(arm_dir, ids):
    meta = _check_arm(arm_dir)
    reg = cases()
    for cid in ids:
        fn, kw = reg[cid]
        t = time.perf_counter()
        try:
            row = fn(**kw)
            row["error"] = None
        except Exception as e:  # noqa: BLE001 -- recorded in the row, never dropped
            row = {"error": f"{type(e).__name__}: {e}"}
        row.update({"case": cid, "arm": meta["name"], "so_sha256": meta["so_sha256"], **_host(),
                    "wall_s": time.perf_counter() - t,
                    "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
        print("@@ROW " + json.dumps(row, default=float), flush=True)


def run(arms, suites, pattern, pairs, out, threads, shard=None):
    """Both arms of every case, interleaved ABBA, `pairs` times.

    `shard` is `(k, n)`: case *batches* are dealt round-robin to n shards and
    this process runs the k-th. Both arms of a case stay in the same shard,
    so a shard can run on its own machine and the pieces merge (`merge`).
    """
    reg = cases()
    ids = sorted(c for c in reg if c.split("/")[0] in suites and (not pattern or re.search(pattern, c)))
    batches = _batches(ids)
    if shard:
        k, n = shard
        batches = [b for i, b in enumerate(batches) if i % n == k]
        ids = [c for b in batches for c in b]
    rows = []
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    t0 = time.perf_counter()
    for p in range(pairs):
        order = list(arms) if p % 2 == 0 else list(reversed(arms))
        rnd = random.Random(p)
        seq = batches[:]
        rnd.shuffle(seq)
        for batch in seq:
            for arm in order:
                arm_dir = os.path.join(ARMS, arm)
                cmd = [sys.executable, os.path.abspath(__file__), "_worker", arm_dir, *batch]
                res = subprocess.run(cmd, env=dict(env, PYTHONPATH=arm_dir), cwd=tempfile.gettempdir(),
                                     capture_output=True, text=True, check=False)
                got = [json.loads(line[6:]) for line in res.stdout.splitlines() if line.startswith("@@ROW ")]
                if res.returncode != 0 or len(got) != len(batch):
                    tail = (res.stderr or "")[-800:]
                    got += [{"case": c, "arm": arm, "error": f"worker exit {res.returncode}: {tail}"}
                            for c in batch[len(got):]]
                for g in got:
                    g["pair"] = p
                rows.extend(got)
        with open(out, "w") as fh:
            json.dump({"arms": arms, "suites": suites, "pattern": pattern, "pairs": pairs,
                       "threads": threads, "shard": shard, **_host(), "rows": rows}, fh, indent=0)
        print(f"pair {p + 1}/{pairs} done, {len(ids)} cases, {time.perf_counter() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------- #

PRIMARY = ("query_ms_per_epoch", "ms_per_epoch", "us_per_call", "us_per_query",
           "fit_ms", "update_ms", "promote_ms", "query_ms", "ms_per_call")
HIGHER_BETTER = ("recall",)
LOWER_BETTER_QUALITY = ("projection_discrepancy", "wrap_around_discrepancy")


def merge(paths, out):
    """Merge shard JSONs into one, and refuse a case split across machines.

    A case whose two arms were measured on different hosts is not a paired
    comparison at all, so merging is where that has to be caught.
    """
    rows, arms, suites, pairs, meta = [], None, None, 0, {}
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        if arms is None:
            arms, suites = d["arms"], d["suites"]
        elif [arms, suites] != [d["arms"], d["suites"]]:
            raise SystemExit(f"{p}: arms/suites differ from the first shard")
        pairs = max(pairs, d["pairs"])
        rows.extend(d["rows"])
        meta[p] = (d.get("host"), d.get("shard"))
    seen = {}
    for r in rows:
        seen.setdefault(r["case"], set()).add((r.get("host"), r.get("cpu")))
    split = {c: h for c, h in seen.items() if len(h) > 1}
    if split:
        raise SystemExit("cases measured on more than one machine: "
                         + ", ".join(sorted(split)[:5]))
    with open(out, "w") as fh:
        json.dump({"arms": arms, "suites": suites, "pattern": None, "pairs": pairs,
                   "threads": None, "shards": meta, "rows": rows}, fh, indent=0)
    print(f"{out}: {len(rows)} rows, {len(seen)} cases, {len(paths)} shards, "
          f"hosts {sorted({h for v in seen.values() for h, _ in v})}")


def _log_ratios(ra, rb, pairs, metric):
    return [math.log(rb[p][metric] / ra[p][metric]) for p in pairs
            if isinstance(ra[p].get(metric), (int, float)) and ra[p][metric] > 0
            and isinstance(rb[p].get(metric), (int, float)) and rb[p][metric] > 0]


def _spread(lr):
    return max(statistics.stdev(lr), 0.005) if len(lr) >= 2 else float("nan")


def _paired(path):
    with open(path) as fh:
        data = json.load(fh)
    by = {}
    for r in data["rows"]:
        by.setdefault(r["case"], {}).setdefault(r["arm"], {})[r.get("pair")] = r
    return data, by


def summarise(path, aa=None, z_crit=3.0):
    """Per case and metric: the paired median ratio, and whether it is noise.

    The verdict is a z-test on the median log ratio, `z = median / (1.2533 *
    s / sqrt(n))`, with `s` the **larger** of the A/A run's spread on that
    case and the comparison's own. Both halves of that are measured choices,
    against a 7-pair A/A run of `b06c7df` against itself (461 case-metrics):

    * a bootstrap interval of the median, the first rule tried, flagged about
      10% of the A/A metrics as significant in one direction or the other;
    * a MAD spread flagged 13 at |z| > 3, and the standard deviation none,
      because noise on this host arrives in bursts that a MAD discounts;
    * calibrating on half the A/A pairs and testing the other half flagged 24,
      because the bursts drift through a session, so the band has to come
      from a full run and may never be narrower than the comparison's own.

    With both, the A/A run flags 0 of 461 at |z| > 3, which is the threshold.
    """
    data, by = _paired(path)
    a, b = data["arms"]
    cal = {}
    if aa:
        adata, aby = _paired(aa)
        a0, b0 = adata["arms"]
        for cid, arms in aby.items():
            ra, rb = arms.get(a0, {}), arms.get(b0, {})
            okp = [p for p in sorted(set(ra) & set(rb)) if not ra[p].get("error") and not rb[p].get("error")]
            for metric in PRIMARY:
                lr = _log_ratios(ra, rb, okp, metric)
                if len(lr) >= 2:
                    cal[(cid, metric)] = _spread(lr)
    out = []
    for cid in sorted(by):
        ra, rb = by[cid].get(a, {}), by[cid].get(b, {})
        errs = sorted({x["error"] for x in list(ra.values()) + list(rb.values()) if x.get("error")})
        row = {"case": cid, "metrics": {}, "errors": errs}
        okp = [p for p in sorted(set(ra) & set(rb)) if not ra[p].get("error") and not rb[p].get("error")]
        for metric in PRIMARY + ("maxrss_mb",):
            lr = _log_ratios(ra, rb, okp, metric)
            if len(lr) < 2:
                continue
            own = _spread(lr)
            band = cal.get((cid, metric))
            s = max(own, band) if band is not None else own
            med = statistics.median(lr)
            z = med / (1.2533 * s / math.sqrt(len(lr)))
            if metric == "maxrss_mb":
                verdict = "worse" if med > math.log(1.01) and z > z_crit else (
                    "better" if med < math.log(0.99) and z < -z_crit else "same")
            else:
                verdict = "REGRESSION" if z > z_crit else ("faster" if z < -z_crit else "same")
            row["metrics"][metric] = {
                "median": math.exp(med), "z": z, "n": len(lr), "spread": s, "calibrated": band is not None,
                "a_median": statistics.median(ra[p][metric] for p in okp), "verdict": verdict}
        for metric in HIGHER_BETTER + LOWER_BETTER_QUALITY:
            va = [ra[p][metric] for p in okp if isinstance(ra[p].get(metric), (int, float))
                  and not math.isnan(ra[p][metric])]
            vb = [rb[p][metric] for p in okp if isinstance(rb[p].get(metric), (int, float))
                  and not math.isnan(rb[p][metric])]
            if va and vb:
                ma, mb = statistics.median(va), statistics.median(vb)
                worse = mb < ma - 1e-12 if metric in HIGHER_BETTER else mb > ma + 1e-12
                row["metrics"][metric] = {"a": ma, "b": mb, "verdict": "WORSE" if worse else "ok"}
        fa = {ra[p].get("fingerprint") for p in okp}
        fb = {rb[p].get("fingerprint") for p in okp}
        row["fingerprint"] = ("identical" if fa == fb and len(fa) == 1 else
                              "nondeterministic" if len(fa) > 1 or len(fb) > 1 else "DIFFERENT")
        out.append(row)
    return {"arms": [a, b], "cases": out}


def report(path, aa=None, only_flagged=False):
    s = summarise(path, aa)
    a, b = s["arms"]
    print(f"## {b} vs {a} (ratio = {b}/{a}; < 1 is faster; |z| > 3 is outside noise)\n")
    print("| case | metric | ratio | z | noise s | verdict | fingerprint | accuracy |")
    print("|---|---|---|---|---|---|---|---|")
    counts = {}
    for row in s["cases"]:
        acc = "; ".join(f"{m} {v['a']:.4g}->{v['b']:.4g} {v['verdict']}"
                        for m, v in row["metrics"].items() if "a" in v)
        if row["errors"]:
            print(f"| {row['case']} | ERROR | | | | {row['errors'][0][:120]} | | |")
            counts["ERROR"] = counts.get("ERROR", 0) + 1
            continue
        for metric, v in row["metrics"].items():
            if "median" not in v:
                continue
            counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
            flagged = (v["verdict"] not in ("same",) or row["fingerprint"] == "DIFFERENT" or "WORSE" in acc)
            if only_flagged and not flagged:
                continue
            cal = "" if v["calibrated"] else " (uncal.)"
            print(f"| {row['case']} | {metric} | {v['median']:.3f} | {v['z']:+.1f} | {v['spread']:.3f}{cal} "
                  f"| {v['verdict']} | {row['fingerprint']} | {acc} |")
    diff = sum(1 for r in s["cases"] if r["fingerprint"] == "DIFFERENT")
    worse = sum(1 for r in s["cases"] for v in r["metrics"].values() if v.get("verdict") == "WORSE")
    print(f"\nverdicts: {counts}; fingerprints different: {diff}; accuracy worse: {worse}")
    return s


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        worker(sys.argv[2], sys.argv[3:])
        sys.exit(0)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("name")
    b.add_argument("--rev")
    b.add_argument("--features")
    b.add_argument("--rustflags")
    b.add_argument("--profile")
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs=2, required=True)
    r.add_argument("--suites", nargs="+", default=["W1", "W2", "W3", "W4", "W5"])
    r.add_argument("--cases", help="regex over case ids")
    r.add_argument("--pairs", type=int, default=7)
    r.add_argument("--threads", type=int, default=os.cpu_count())
    r.add_argument("--out", required=True)
    r.add_argument("--shard", help="K/N: run the K-th of N case shards (0-based)")
    mg = sub.add_parser("merge")
    mg.add_argument("paths", nargs="+")
    mg.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("path")
    p.add_argument("--aa")
    p.add_argument("--flagged", action="store_true")
    ls = sub.add_parser("list")
    ls.add_argument("--suites", nargs="+")
    args = ap.parse_args()
    if args.cmd == "build":
        build_arm(args.name, args.rev, args.rustflags, args.profile, args.features)
    elif args.cmd == "run":
        shard = None
        if args.shard:
            k, n = (int(x) for x in args.shard.split("/"))
            if not 0 <= k < n:
                raise SystemExit("--shard K/N needs 0 <= K < N")
            shard = (k, n)
        run(args.arms, args.suites, args.cases, args.pairs, args.out, args.threads, shard)
    elif args.cmd == "merge":
        merge(args.paths, args.out)
    elif args.cmd == "report":
        report(args.path, args.aa, args.flagged)
    else:
        for cid in cases():
            if not args.suites or cid.split("/")[0] in args.suites:
                print(cid)
