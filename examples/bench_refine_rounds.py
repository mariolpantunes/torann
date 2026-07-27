"""ESS as a refinement method: repeated calls over an accumulating set.

Everything else in `examples/` measures a *single* `esa` call. That is not
how ESS is used. It is called again and again, each time handed the points
it produced before, to refine the exploration of a space that already has
points in it. The static tier is therefore the normal case and it grows
without bound, while the batch stays the same size.

That workload stresses parts of the index the single-call benchmarks never
touch:

* Every `esa` call re-fits the index from scratch and re-tunes ``(B, K, L)``
  on the new ``n``, so a round pays a full build over all accumulated
  points to add a fixed-size batch. Whether that build is a rounding error
  or the dominant cost is what the ``setup`` column answers.
* The static:candidate ratio climbs every round, which is exactly the axis
  along which recall was measured to improve (0.689 empty vs 0.861 with
  anchors at d=32).
* Quality has to be judged on the *accumulated* set, not on the batch, so
  the question "does round 6 still add regularity, or is it filling a set
  that is already as regular as it will get?" has an answer.

Against that, the one-shot alternative: a single call generating the same
total number of points. Same budget, no accumulation. If iterative refinement
costs much more for the same quality, that is worth knowing before optimising
the path further; if it costs more but *buys* quality, that is worth knowing
too.

Run from the repository root (needs `ess` on the path)::

    python examples/bench_refine_rounds.py
    python examples/bench_refine_rounds.py --cases 3 --rounds 4
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torann import ToroidalNN  # noqa: E402

try:
    import ess
    import ess.utils
except ImportError:  # pragma: no cover - benchmark-only dependency
    sys.exit("ess not importable: add ~/git/ess/src to PYTHONPATH")

OUT = os.path.join(os.path.dirname(__file__), "out")

# (dim, points per round). Rounds come from --rounds.
CASES = ((2, 256), (8, 512), (32, 2000))


def metrics(points):
    return (float(ess.utils.toroidal_clark_evans(points)),
            float(ess.utils.toroidal_separation(points)))


def one_round(acc, dim, n, seed):
    """One `esa` call with `acc` as the static tier. Returns points + timings."""
    bounds = np.array([[0.0, 1.0]] * dim)
    index = ToroidalNN(seed=seed)
    stats: dict = {}
    t0 = time.perf_counter()
    pts = ess.esa(acc, bounds, n=n, index=index, seed=seed, stats=stats)
    wall = time.perf_counter() - t0
    return pts, {
        "wall_s": wall,
        "query_s": stats.get("query_s", 0.0),
        "setup_s": stats.get("setup_s", 0.0),
        "update_s": stats.get("update_s", 0.0),
        "force_s": stats.get("force_s", 0.0),
        "epochs": stats.get("epochs_total"),
        "tables": index.n_tables,
        "mode": "lsh" if index.is_approximate else "brute",
    }


def run_iterative(dim, n, rounds, seed):
    """`rounds` successive calls, each handed everything produced so far."""
    acc = np.empty((0, dim))
    out = []
    for r in range(rounds):
        pts, info = one_round(acc, dim, n, seed + r)
        acc = np.vstack([acc, pts])
        ce, sep = metrics(acc)
        out.append({"round": r + 1, "static_in": acc.shape[0] - n,
                    "total": acc.shape[0], "clark_evans": ce,
                    "separation": sep, **info})
        print(f"  [d={dim} round {r + 1}/{rounds}: {acc.shape[0]} pts, "
              f"{info['wall_s']:.2f}s, CE {ce:.4f}]", flush=True)
    return out, acc


def run_one_shot(dim, total, seed):
    """A single call for the same final point count."""
    pts, info = one_round(np.empty((0, dim)), dim, total, seed)
    ce, sep = metrics(pts)
    return {"round": 0, "static_in": 0, "total": total, "clark_evans": ce,
            "separation": sep, **info}


def report(rows):
    for dim in sorted({r["dim"] for r in rows}, reverse=True):
        sub = [r for r in rows if r["dim"] == dim]
        it = [r for r in sub if r["variant"] == "iterative"]
        os_ = [r for r in sub if r["variant"] == "one-shot"]
        print(f"\nd={dim}")
        head = ("round", "static in", "total", "mode", "L", "wall",
                "setup", "query", "epochs", "CE", "separation")
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for r in it:
            print(f"| {r['round']} | {r['static_in']} | {r['total']} "
                  f"| {r['mode']} | {r['tables']} | {r['wall_s']:.2f}s "
                  f"| {100 * r['setup_s'] / max(r['wall_s'], 1e-9):.0f}% "
                  f"| {100 * r['query_s'] / max(r['wall_s'], 1e-9):.0f}% "
                  f"| {r['epochs']} | {r['clark_evans']:.4f} "
                  f"| {r['separation']:.4f} |")
        for r in os_:
            print(f"| one-shot | 0 | {r['total']} | {r['mode']} "
                  f"| {r['tables']} | {r['wall_s']:.2f}s "
                  f"| {100 * r['setup_s'] / max(r['wall_s'], 1e-9):.0f}% "
                  f"| {100 * r['query_s'] / max(r['wall_s'], 1e-9):.0f}% "
                  f"| {r['epochs']} | {r['clark_evans']:.4f} "
                  f"| {r['separation']:.4f} |")
        if it and os_:
            tot = sum(r["wall_s"] for r in it)
            print(f"\n  iterative total {tot:.2f}s vs one-shot "
                  f"{os_[0]['wall_s']:.2f}s "
                  f"({tot / max(os_[0]['wall_s'], 1e-9):.2f}x) — "
                  f"final CE {it[-1]['clark_evans']:.4f} vs "
                  f"{os_[0]['clark_evans']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="+")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=os.path.join(OUT, "refine_rounds.json"))
    args = ap.parse_args()

    import torann
    print(f"[torann: {torann.__file__}]")
    picked = ([CASES[i - 1] for i in args.cases] if args.cases else list(CASES))

    rows = []
    for dim, n in picked:
        it, _ = run_iterative(dim, n, args.rounds, args.seed)
        rows += [{"dim": dim, "per_round": n, "variant": "iterative", **r}
                 for r in it]
        shot = run_one_shot(dim, n * args.rounds, args.seed)
        rows.append({"dim": dim, "per_round": n, "variant": "one-shot", **shot})
        print(f"  [d={dim} one-shot {n * args.rounds} pts: "
              f"{shot['wall_s']:.2f}s, CE {shot['clark_evans']:.4f}]",
              flush=True)

    report(rows)
    os.makedirs(OUT, exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n[saved {args.json}]")
