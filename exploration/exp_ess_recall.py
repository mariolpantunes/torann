"""ESS dispersion against index effort: one run, one JSON line.

ESS is ranked on `wrap_disc` alone (wrap-around discrepancy, 1.0 = random,
lower is better). Time and neighbour quality are reported beside it, so a
lost neighbour can be priced in design quality rather than in recall.

    python exploration/exp_ess_recall.py --label s3fix --d 32 --anchors 8000 \
        --cands 8000 --mode knn --seed 0 --setting auto --out run.json
"""

import argparse
import json
import time

import ess
import numpy as np
from ess.utils import expected_discrepancy, wrap_around_discrepancy

from torann import ToroidalNN
from torann.brute import exact_knn

K = 5
TAKE = 500


def make_index(setting, seed):
    """`default` is what ESS gets today; a number forces LSH at that target."""
    if setting == "default":
        return ToroidalNN(seed=seed)
    if setting == "exact":
        return ToroidalNN(seed=seed, brute_threshold=10**12)
    target = "auto" if setting == "auto" else float(setting)
    return ToroidalNN(seed=seed, brute_threshold=0, recall=target)


def neighbour_quality(index, seed):
    """Hit rate per rank, and how often each position holds its true rank."""
    if not index.is_approximate:
        return {"rank_hit": [1.0] * K, "position_ok": [1.0] * K}
    arena = index._arena
    rows = np.random.default_rng(seed).choice(arena.shape[0], min(TAKE, arena.shape[0]), replace=False)
    ids = rows.astype(np.int64)
    queries = np.ascontiguousarray(arena[rows])
    got, _ = index.query(k=K, queries=queries, exclude_ids=ids)
    want, _ = exact_knn(arena, queries, K, exclude_ids=ids)
    hit = [float(np.mean([want[i, r] in got[i] for i in range(len(rows))])) for r in range(K)]
    return {"rank_hit": hit, "position_ok": (got == want).mean(axis=0).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--anchors", type=int, required=True)
    ap.add_argument("--cands", type=int, required=True)
    ap.add_argument("--mode", choices=("knn", "radius"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = args.d
    rng = np.random.default_rng(args.seed)
    anchors = rng.random((args.anchors, d)) if args.anchors else np.empty((0, d))
    index = make_index(args.setting, args.seed)
    stats = {}
    t0 = time.perf_counter()
    pts = ess.esa(
        anchors,
        np.array([[0.0, 1.0]] * d),
        n=args.cands,
        index=index,
        seed=args.seed,
        stats=stats,
        search_mode="k_nn" if args.mode == "knn" else "radius",
    )
    total = time.perf_counter() - t0
    design = np.vstack([anchors, pts]) if args.anchors else pts
    row = {
        **vars(args),
        "total_s": total,
        "query_s": stats.get("query_s"),
        "epochs": stats.get("epochs_total"),
        "approximate": bool(index.is_approximate),
        "tables": getattr(index, "n_tables", getattr(index, "_L", None)),
        "wrap_disc": float(wrap_around_discrepancy(design) / expected_discrepancy(*design.shape)),
        **neighbour_quality(index, args.seed),
    }
    with open(args.out, "w") as f:
        json.dump(row, f)


if __name__ == "__main__":
    main()
