"""Check `ToroidalNN._tune` against the real hash and probe rule.

For the true k-th neighbour pairs of a workload, draw many tables with the
normative hash and probe rule (`lsh.py`: base key + the min(probes, K) closest
walls) and measure the per-table hit rate. Compare it with:

  code   the p_table formula `_tune` used before
  model  `wrapper._wall_hit_rate`, the model `_tune` uses now

Run: python exploration/exp_tune_model.py [--quick] [--json out.json]
"""

import argparse
import json
from itertools import pairwise
from math import comb, exp

import numpy as np

from torann import ToroidalNN
from torann.brute import exact_knn
from torann.wrapper import _wall_hit_rate


def code_p_table(mu, K, m):
    """The p_table `_tune` used before the wall model, kept as the reference."""
    q0 = min(0.95, mu)
    p0 = 1.0 - q0
    p1 = p0**K
    if m == 0:
        return p1
    q1 = mu * (1.0 - exp(-1.0 / mu)) * (1.0 - exp(-1.0 / (2.0 * mu)))
    X = min(1.0, 2.0 * q0)
    return p1 + (p0 ** (K - 1)) * (q1 / X) * sum(
        min(r, m) * comb(K, r) * (X**r) * ((1.0 - X) ** (K - r)) for r in range(1, K + 1)
    )


def empirical(Q, X, B, K, m, tables, rng):
    """Per-pair hit rate over `tables` independent draws of the normative hash."""
    pairs, d = Q.shape
    hits = np.zeros(pairs)
    rows = np.arange(pairs)
    m1, m2 = min(m, K), min(max(0, m - K), comb(K, 2))
    pairs2 = [(a, b) for b in range(1, K) for a in range(b)][:m2]
    for _ in range(tables):
        S = rng.integers(0, d, size=K)
        U = rng.random(K)
        fq = np.mod(Q[:, S] + U, 1.0) * B
        fx = np.mod(X[:, S] + U, 1.0) * B
        cq = np.minimum(fq.astype(np.int64), B - 1)
        cx = np.minimum(fx.astype(np.int64), B - 1)
        diff = cq != cx
        ndiff = diff.sum(axis=1)
        hit = ndiff == 0
        if m1 > 0:
            f = fq - cq
            alt = (cq + np.where(f >= 0.5, 1, -1)) % B
            close = np.minimum(f, 1.0 - f)
            order = np.argsort(close, axis=1, kind="stable")
            top = order[:, :m1]
            j = diff.argmax(axis=1)
            in_top = (top == j[:, None]).any(axis=1)
            hit |= (ndiff == 1) & in_top & (alt[rows, j] == cx[rows, j])
            if m2 > 0:
                for r in np.flatnonzero(ndiff == 2):
                    a, b = np.flatnonzero(diff[r])
                    if alt[r, a] == cx[r, a] and alt[r, b] == cx[r, b]:
                        ord_r = order[r]
                        if any(
                            (ord_r[pa] == a and ord_r[pb] == b) or (ord_r[pa] == b and ord_r[pb] == a)
                            for pa, pb in pairs2
                        ):
                            hit[r] = True
        hits += hit
    return hits / tables


def workload(d, n, data, rng):
    if data == "uniform":
        return rng.random((n, d))
    centres = rng.random((16, d))
    return np.mod(centres[rng.integers(0, 16, n)] + rng.normal(0.0, 0.02, (n, d)), 1.0)


def one(d, n, p, data, probes, pairs, tables, seed):
    rng = np.random.default_rng(seed)
    pts = workload(d, n, data, rng)
    k = min(2 * d, 64)
    q_ids = rng.choice(n, pairs, replace=False)
    Q = pts[q_ids]
    idx, dst = exact_knn(pts, Q, k + 1, p=p)  # rank 0 is the query itself
    X = pts[idx[:, k]]
    r_lp = float(np.median(dst[:, k]))
    gap = np.abs(Q - X)
    gap = np.minimum(gap, 1.0 - gap)
    nn = ToroidalNN(seed=seed, brute_threshold=0, probes=probes, p=p, backend="python").fit(pts, k=k)
    B, K, L = nn._B, nn._K, nn._L
    delta = float(np.median(gap.sum(axis=1))) / d
    mu = B * delta * (1.0 if p == 1.0 else p**-0.15)
    m = min(probes, K + comb(K, 2))
    emp = empirical(Q, X, B, K, m, tables, rng)
    code = code_p_table(mu, K, m)
    model = _wall_hit_rate(gap[: min(pairs, 256)], B, K, m)
    # Radius path of _tune: delta guessed from an L^p radius.
    guess = r_lp / d if p == 1.0 else r_lp / d ** (1.0 / p) * p**-0.32

    def rec(P):
        return float(np.mean(1.0 - (1.0 - P) ** L))

    idx_real = ToroidalNN(seed=seed + 1, brute_threshold=0, probes=probes, p=p, backend="python")
    idx_real.fit(pts, k=k)
    got, _ = idx_real.query(k=k, queries=Q[:200], exclude_ids=q_ids[:200])
    full = np.mean([len(set(a) & set(b)) / k for a, b in zip(got, idx[:200, 1 : k + 1], strict=True)])
    return {
        "d": d,
        "n": n,
        "p": p,
        "data": data,
        "probes": probes,
        "B": B,
        "K": K,
        "L": L,
        "mu": round(mu, 3),
        "emp": float(emp.mean()),
        "code": code,
        "model": float(model.mean()),
        "rec_emp": rec(emp),
        "rec_code": 1.0 - (1.0 - code) ** L,
        "rec_model": rec(model),
        "rec_real": float(full),
        "delta": delta,
        "delta_radius": guess,
        "L_for_90_emp": L_for(emp, 0.90),
    }


def L_for(P, target):
    """Smallest L with mean recall >= target (capped at 256)."""
    for L in range(1, 257):
        if np.mean(1.0 - (1.0 - P) ** L) >= target:
            return L
    return 257


def stability():
    """Bounds and monotonicity of the code's p_table on a dense grid."""
    bad_bound, bad_probe, bad_mu, top = 0, 0, 0, 0.0
    first = None
    mus = np.geomspace(1e-9, 2.0, 200)
    for K in range(1, 63):
        for mu in mus:
            prev = -1.0
            for m in range(K + 1):
                v = code_p_table(float(mu), K, m)
                top = max(top, v)
                if not 0.0 <= v <= 1.0 + 1e-12:
                    bad_bound += 1
                    first = first or ("bound", K, float(mu), m, v)
                if v < prev - 1e-12:
                    bad_probe += 1
                    first = first or ("probe", K, float(mu), m, v, prev)
                prev = v
        for m in sorted({0, min(1, K), min(4, K)}):
            v = [code_p_table(float(mu), K, m) for mu in mus]
            if any(b > a + 1e-9 for a, b in pairwise(v)):
                bad_mu += 1
    return {"bound_viol": bad_bound, "probe_nonmono": bad_probe, "mu_nonmono_KM": bad_mu, "max": top, "first": first}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--dims", default=None)
    ap.add_argument("--data", default="uniform,clustered")
    ap.add_argument("--probes", default="0,1,4,8")
    ap.add_argument("--ps", default="1,0.75,0.5,0.25")
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--tables", type=int, default=1000)
    a = ap.parse_args()
    print("stability", stability())
    pairs, tables = (300, 300) if a.quick else (a.pairs, a.tables)
    dims = (8, 32) if a.quick else (2, 4, 8, 16, 32, 64, 100)
    dims = tuple(int(x) for x in a.dims.split(",")) if a.dims else dims
    rows = []
    hdr = "data    d      n    p prb B  K  L    mu     emp    code   model | rec_emp rec_code rec_model real | L90 | dlt/rad"
    print(hdr)
    for data in a.data.split(","):
        for d in dims:
            for n in (4096, 32768):
                for p in (float(x) for x in a.ps.split(",")):
                    for probes in (int(x) for x in a.probes.split(",")):
                        r = one(d, n, p, data, probes, pairs, tables, seed=d * 7 + n)
                        rows.append(r)
                        print(
                            f"{data[:7]:7s} {d:3d} {n:6d} {p:4.2f} {probes:3d} {r['B']} {r['K']:2d} {r['L']:2d}"
                            f" {r['mu']:5.3f} {r['emp']:7.4f} {r['code']:7.4f} {r['model']:7.4f}"
                            f" | {r['rec_emp']:.3f} {r['rec_code']:.3f} {r['rec_model']:.3f} {r['rec_real']:.3f}"
                            f" | {r['L_for_90_emp']:3d} | {r['delta'] / r['delta_radius']:.2f}",
                            flush=True,
                        )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f)


if __name__ == "__main__":
    main()
