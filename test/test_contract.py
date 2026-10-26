"""Strict contract suite: torus geometry, ordering and corner cases.

Written before any optimisation (``.agents``-side spec ``test-spec.md``):
an idea that fails here is rejected before it is benchmarked. Every test
runs on every installed backend and on two paths with *exact* expected
results -- the brute scan, and an LSH configuration whose single one-digit
``B=2`` table plus one probe covers both cells, so every point is a
candidate while the whole LSH code path (hash, probe, gather, dedupe,
join, refine) runs.

The oracle is float64 NumPy over the torus, ordered by ``(distance, id)``.
Distances from the float32 LSH refine are compared within ``tol32(d)``;
dyadic data (multiples of ``2**-16``) is exact in float32 as well, so there
the comparison is bit for bit.
"""

import os
import unittest

import numpy as np

from torann import ToroidalNN, available_backends

BACKENDS = available_backends()
SLOW = os.environ.get("TORANN_SLOW") == "1"

BRUTE = {"brute_threshold": 10**9}
EXHAUSTIVE = {"brute_threshold": 1, "resolution": 2, "dims_per_table": 1, "probes": 1, "num_tables": 1}
PATHS = {"brute": BRUTE, "lsh-exhaustive": EXHAUSTIVE}

GRID = 2.0**-16  # dyadic step: every f32 operation on it is exact for d <= 128


def tol32(d):
    """Bound on the float32 refine error at dimension ``d``."""
    return d * 2.0**-23


def torus(Q, P):
    """Toroidal L1, float64: the oracle's metric."""
    diff = np.abs(np.asarray(Q, float)[:, None, :] - np.asarray(P, float)[None, :, :])
    return np.minimum(diff, 1.0 - diff).sum(-1)


def oracle_knn(P, Q, k, exclude=None):
    """Exact k-NN ordered by (distance, id), padded with (-1, inf)."""
    P, Q = np.mod(P, 1.0), np.mod(np.atleast_2d(Q), 1.0)
    D = torus(Q, P)
    if exclude is not None:
        for row, e in enumerate(exclude):
            if e >= 0:
                D[row, e] = np.inf
    m, n = D.shape
    ids = np.full((m, k), -1, dtype=np.int64)
    dst = np.full((m, k), np.inf)
    for row in range(m):
        order = np.lexsort((np.arange(n), D[row]))
        order = order[np.isfinite(D[row, order])][:k]
        ids[row, : order.size] = order
        dst[row, : order.size] = D[row, order]
    return ids, dst


def oracle_radius(P, Q, r, exclude=None):
    """Exact closed-ball rows ordered by (distance, id)."""
    P, Q = np.mod(P, 1.0), np.mod(np.atleast_2d(Q), 1.0)
    D = torus(Q, P)
    rows = []
    for row in range(D.shape[0]):
        hit = np.flatnonzero(D[row] <= r)
        if exclude is not None and exclude[row] >= 0:
            hit = hit[hit != exclude[row]]
        hit = hit[np.lexsort((hit, D[row, hit]))]
        rows.append((hit, D[row, hit]))
    return rows


def dyadic(rng, shape):
    return rng.integers(0, int(1 / GRID), shape) * GRID


def make(backend, path, **kw):
    return ToroidalNN(seed=0, backend=backend, **PATHS[path], **kw)


class ContractCase(unittest.TestCase):
    """Assertions shared by every class below."""

    def each(self):
        """Every (backend, path) pair; callers open a subTest per pair."""
        return [(backend, path) for backend in BACKENDS for path in PATHS]

    def assert_sorted(self, ids, dists):
        """Rows ascending by distance, ties by ascending id, padding last."""
        ids, dists = np.atleast_2d(ids), np.atleast_2d(dists)
        if ids.size == 0:
            return
        real = ids >= 0
        self.assertTrue(np.all(real[:, :-1] >= real[:, 1:]), "padding before a real id")
        pair = real[:, :-1] & real[:, 1:]  # both slots hold real neighbours
        step = np.diff(dists, axis=1)
        bad = pair & ((step < 0) | ((step == 0) & (np.diff(ids, axis=1) <= 0)))
        rows = np.flatnonzero(bad.any(1))
        self.assertEqual(rows.size, 0, f"order broken in rows {rows[:5]}: {ids[rows[:1]]} {dists[rows[:1]]}")

    def assert_identical(self, got, want):
        """Bit-exact ids and distances (dyadic data, or the brute path)."""
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])

    def assert_equivalent(self, got, P, Q, k, exclude=None, D=None):
        """k-NN exact up to float32 rounding: as many real ids as reachable
        points (then -1 / inf padding), distinct ids, the excluded id never
        returned, every distance the true one within tol32, and no point
        closer than the k-th returned (minus tol32) left out. Within tol32
        of each other two points may swap -- nothing else may differ."""
        P, Q = np.mod(P, 1.0), np.mod(np.atleast_2d(Q), 1.0)
        tol = tol32(P.shape[1])
        if D is None:
            D = torus(Q, P)
        D = D.copy()
        if exclude is not None:
            rows = np.flatnonzero(np.asarray(exclude) >= 0)
            D[rows, np.asarray(exclude)[rows]] = np.inf
        gi, gd = np.atleast_2d(got[0]), np.atleast_2d(got[1])
        m, n = D.shape
        self.assertEqual(gi.shape, (m, k))
        real = gi >= 0
        want_real = np.minimum(k, np.isfinite(D).sum(1))
        np.testing.assert_array_equal(real.sum(1), want_real, err_msg="padding differs")
        self.assertTrue(np.all(real[:, :-1] >= real[:, 1:]), "padding before a real id")
        self.assertTrue(np.all(np.isinf(gd[~real])), "padding distance is not inf")
        rows, cols = np.nonzero(real)
        ids = gi[rows, cols]
        np.testing.assert_allclose(gd[rows, cols], D[rows, ids], rtol=0, atol=tol)
        member = np.zeros((m, n), bool)
        member[rows, ids] = True
        np.testing.assert_array_equal(member.sum(1), want_real, err_msg="duplicate id in a row")
        last = np.where(want_real > 0, gd[np.arange(m), np.maximum(want_real - 1, 0)], -np.inf)
        missing = (D < (last - tol)[:, None]) & ~member
        self.assertFalse(missing.any(), f"closer point missing in rows {np.flatnonzero(missing.any(1))[:5]}")
        self.assert_sorted(gi, gd)


class TestOrder(ContractCase):
    """T0a: rows sorted by distance (ESS relies on it), ties by ascending id."""

    def test_knn_ties_follow_id_order(self):
        rng = np.random.default_rng(1)
        G = rng.integers(0, 4, (300, 3)) / 4  # heavy ties
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                got = make(backend, path).fit(G).query(25, G[:30])
                self.assert_identical(got, oracle_knn(G, G[:30], 25))

    def test_radius_ties_follow_id_order(self):
        rng = np.random.default_rng(2)
        G = rng.integers(0, 4, (300, 3)) / 4
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                rows = make(backend, path).fit(G).query_radius(0.5, G[:30])
                for got, want in zip(rows, oracle_radius(G, G[:30], 0.5)):
                    self.assert_identical(got, want)

    def test_continuous_rows_are_sorted(self):
        rng = np.random.default_rng(3)
        P = rng.random((800, 8))
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                nn = make(backend, path).fit(P[:600], P[600:])
                self.assert_sorted(*nn.query(16))
                padded = nn.query_radius(1.5, pad=True)
                self.assert_sorted(*padded)


class TestSeam(ContractCase):
    """T1: values at and around the seam reduce to their torus image."""

    FIXTURES = (0.0, -0.0, 2.0**-24, 0.5, 1 - 2.0**-24, 1 - 2.0**-53, 1.0, -1e-20, 1 + 2.0**-20, -3.0)

    def test_fixtures_match_the_oracle(self):
        vals = np.array(self.FIXTURES)
        P = np.stack(np.meshgrid(vals, vals), -1).reshape(-1, 2)
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                nn = make(backend, path).fit(P)
                self.assert_equivalent(nn.query(5, P), P, P, 5)

    def test_reduced_coordinates_stay_below_one(self):
        """C4: a value whose mod is exactly 1.0 is stored as 0.0."""
        P = np.array([[-1e-20, 0.5], [0.25, -1e-300], [0.3, 0.7]])
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                nn = make(backend, path).fit(P)
                self.assertTrue(np.all((nn._arena >= 0.0) & (nn._arena < 1.0)))
                self.assertEqual(nn._arena[0, 0], 0.0)

    def test_short_way_round(self):
        P = np.array([[0.01, 0.5], [0.99, 0.5], [0.5, 0.5]])
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                ids, dist = make(backend, path).fit(P).query(1, P[:1], exclude_ids=[0])
                self.assertEqual(ids[0, 0], 1)
                self.assertAlmostEqual(dist[0, 0], 0.02, delta=tol32(2))


class TestInvariance(ContractCase):
    """T2: the torus has no origin, no orientation and no preferred axis.
    On dyadic data every transform must give bit-identical answers."""

    def setUp(self):
        rng = np.random.default_rng(4)
        self.P = dyadic(rng, (400, 6))
        self.shifts = [dyadic(rng, (6,)) for _ in range(3)] + [np.full(6, 1 - GRID)]

    def answers(self, backend, path, P):
        nn = make(backend, path).fit(P)
        return nn.query(12, P[:40]), nn.query_radius(1.0, P[:40], pad=True)

    def check(self, transform):
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                knn, rad = self.answers(backend, path, self.P)
                tknn, trad = self.answers(backend, path, transform(self.P))
                self.assert_identical(tknn, knn)
                self.assert_identical(trad, rad)
                self.assert_identical(knn, oracle_knn(self.P, self.P[:40], 12))

    def test_translation(self):
        for t in self.shifts:
            self.check(lambda P, t=t: np.mod(P + t, 1.0))

    def test_reflection(self):
        for mask in (np.ones(6, bool), np.arange(6) % 2 == 0):
            self.check(lambda P, mask=mask: np.where(mask, np.mod(1.0 - P, 1.0), P))

    def test_permutation(self):
        perm = np.random.default_rng(5).permutation(6)
        self.check(lambda P: P[:, perm])

    def test_integer_shift(self):
        self.check(lambda P: P + 3.0)
        self.check(lambda P: P - 2.0)


class TestAntipodal(ContractCase):
    """T3: half a turn in every dimension is the maximum distance d/2."""

    def test_maximum_distance(self):
        for d in (1, 2, 5):
            q = np.full((1, d), 0.25)
            P = np.vstack([np.full(d, 0.75), np.full(d, 0.75) - 1.0, np.full(d, -0.25)])
            for backend, path in self.each():
                with self.subTest(backend=backend, path=path):
                    ids, dist = make(backend, path).fit(P).query(3, q)
                    np.testing.assert_array_equal(dist[0], np.full(3, d / 2))
                    np.testing.assert_array_equal(ids[0], [0, 1, 2])


class TestOracleAgreement(ContractCase):
    """T4: continuous data, every width the kernels specialise on, k from 1
    to past n, self-join and external queries, block sizes either side of
    the batched join's minimum."""

    DIMS = (1, 2, 3, 8, 16, 32, 33, 64)

    def test_knn(self):
        rng = np.random.default_rng(6)
        for d in self.DIMS:
            P = rng.random((260, d))
            Q = rng.random((70, d))
            D_ext, D_self = torus(Q, P), torus(P[200:], P)
            self_ids = np.arange(200, 260)
            for backend, path in self.each():
                blocks = (1, 63, 64, 1024) if path != "brute" else (None,)  # brute ignores blocks
                for block in blocks:
                    nn = make(backend, path, query_block_size=block).fit(P[:200], P[200:])
                    for k in (1, 5, 2 * d, 259, 263):
                        with self.subTest(backend=backend, path=path, d=d, k=k, block=block):
                            self.assert_equivalent(nn.query(k, Q), P, Q, k, D=D_ext)
                            self.assert_equivalent(nn.query(k), P, P[200:], k, self_ids, D=D_self)


class TestRadius(ContractCase):
    """T5: closed ball, exact on dyadic data, join and scan both pinned."""

    def test_closed_ball_on_dyadic_data(self):
        rng = np.random.default_rng(7)
        for d in (2, 8, 32, 33):
            P = dyadic(rng, (300, d))
            r = float(np.sort(torus(P[:1], P)[0])[20])  # a distance some point has exactly
            for block in (8, 256):  # below BATCH_MIN: scan; above: join (d <= 32)
                for backend, path in self.each():
                    with self.subTest(backend=backend, path=path):
                        rows = make(backend, path, query_block_size=block).fit(P).query_radius(r, P[:150])
                        with self.subTest(d=d, block=block):
                            for got, want in zip(rows, oracle_radius(P, P[:150], r)):
                                self.assert_identical(got, want)

    def test_join_equals_scan(self):
        rng = np.random.default_rng(8)
        for d in (8, 32):
            P = rng.random((2000, d))
            r = 0.3 * d / 4
            for backend in BACKENDS:
                kw = {"seed": 3, "backend": backend, "brute_threshold": 1}
                scan = ToroidalNN(query_block_size=8, **kw).fit(P[:1500], P[1500:]).query_radius(r, pad=True)
                join = ToroidalNN(query_block_size=512, **kw).fit(P[:1500], P[1500:]).query_radius(r, pad=True)
                with self.subTest(d=d, backend=backend):
                    self.assert_identical(join, scan)

    def test_degenerate_radii(self):
        P = np.vstack([np.random.default_rng(9).random((40, 3)), [[0.1, 0.2, 0.3]]])
        q = np.array([[0.1, 0.2, 0.3]])
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                nn = make(backend, path).fit(P)
                self.assertEqual(nn.query_radius(0.0, q)[0].ids.tolist(), [40])
                self.assertEqual(nn.query_radius(np.inf, q)[0].ids.size, 41)
                for bad in (-1e-9, -1.0, np.nan):
                    with self.assertRaises(ValueError):
                        nn.query_radius(bad, q)


class TestProbeWrap(ContractCase):
    """T7: the neighbour cell of cell 0 is cell B-1 (keys wrap mod B)."""

    def test_probe_reaches_across_the_key_seam(self):
        for B in (2, 3, 8):
            for backend in BACKENDS:
                kw = {"seed": 1, "backend": backend, "brute_threshold": 1, "resolution": B}
                kw.update(dims_per_table=1, num_tables=1)
                base = ToroidalNN(probes=1, **kw).fit(np.full((2, 1), 0.5))  # same n: same draws
                u = float(base._U[0, 0])
                eps = 1 / (16 * B)
                q = np.mod(-u + eps, 1.0)  # hashes to cell 0, near its lower wall
                near = np.mod(q - 2 * eps, 1.0)  # same distance, cell B-1
                far = np.mod(q + 0.9 / B, 1.0)  # still cell 0, but far
                P = np.array([[near], [far]])
                with self.subTest(B=B, backend=backend):
                    ids, _ = ToroidalNN(probes=1, **kw).fit(P).query(1, [[q]])
                    self.assertEqual(ids[0, 0], 0)
                    ids, _ = ToroidalNN(probes=0, **kw).fit(P).query(1, [[q]])
                    self.assertEqual(ids[0, 0], 1, "without the probe only the own cell is read")


class TestLifecycleAcrossTheSeam(ContractCase):
    """T9: moves that cross 0/1 keep every answer exact."""

    def test_updates_crossing_the_seam(self):
        rng = np.random.default_rng(10)
        P = rng.random((300, 4))
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                nn = make(backend, path).fit(P[:200], P[200:])
                cand = P[200:].copy()
                for step in (0.6, -0.9, 0.31):
                    cand = cand + step  # deliberately not reduced: the index must wrap
                    nn.update(cand)
                    allp = np.vstack([P[:200], cand])
                    self.assert_equivalent(nn.query(7), allp, allp[200:], 7, np.arange(200, 300))
                extra = rng.random((50, 4)) - 1.0
                nn.promote(extra)
                allp = np.vstack([P[:200], cand, extra])
                self.assert_equivalent(nn.query(7), allp, allp[300:], 7, np.arange(300, 350))


class TestCornerContract(ContractCase):
    """C1-C23 of the corner-case contract."""

    def setUp(self):
        rng = np.random.default_rng(11)
        self.P = rng.random((60, 3))
        self.Q = rng.random((5, 3))

    def fitted(self, backend, path):
        return make(backend, path).fit(self.P[:50], self.P[50:])

    def assert_unchanged(self, nn, call):
        """C23: a rejected call leaves the index answering as before."""
        before = nn.query(4), nn.query(4, self.Q)
        with self.assertRaises(ValueError):
            call()
        after = nn.query(4), nn.query(4, self.Q)
        for a, b in zip(before, after):
            self.assert_identical(a, b)

    def test_non_finite_input_is_rejected(self):
        """C1-C3, C23."""
        for bad in (np.nan, np.inf, -np.inf):
            for backend, path in self.each():
                with self.subTest(backend=backend, path=path):
                    broken = self.P.copy()
                    broken[3, 1] = bad
                    with self.assertRaises(ValueError):
                        make(backend, path).fit(broken)
                    with self.assertRaises(ValueError):
                        make(backend, path).fit(self.P[:50], broken[:10])
                    nn = self.fitted(backend, path)
                    moved = self.P[50:].copy()
                    moved[2, 0] = bad
                    self.assert_unchanged(nn, lambda nn=nn, moved=moved: nn.update(moved))
                    self.assert_unchanged(nn, lambda nn=nn, moved=moved: nn.promote(moved))
                    q = self.Q.copy()
                    q[0, 2] = bad
                    self.assert_unchanged(nn, lambda nn=nn, q=q: nn.query(3, q))
                    self.assert_unchanged(nn, lambda nn=nn, q=q: nn.query_radius(0.5, q))

    def test_huge_finite_coordinates_are_reduced(self):
        """C5: 1e17 has no fractional bits left; it reduces to 0.0."""
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                nn = make(backend, path).fit(np.vstack([self.P, [[1e17, 0.5, 0.5]]]))
                self.assertTrue(np.all(nn._arena < 1.0))
                np.testing.assert_array_equal(nn._arena[-1], [0.0, 0.5, 0.5])

    def test_k_must_be_an_integer(self):
        """C6-C8."""
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                nn = self.fitted(backend, path)
                for bad in (2.7, 2.0, True, "3"):
                    with self.assertRaises(TypeError):
                        nn.query(bad, self.Q)
                for bad in (0, -1):
                    with self.assertRaises(ValueError):
                        nn.query(bad, self.Q)
                self.assertEqual(nn.query(np.int32(3), self.Q).ids.shape, (5, 3))
                ids, dist = nn.query(70)
                self.assertTrue(np.all(ids[:, 59:] == -1) and np.all(np.isinf(dist[:, 59:])))
                self.assertTrue(np.all(ids[:, :59] >= 0))

    def test_exclude_ids_are_checked(self):
        """C9-C10; -1 means exclude nothing."""
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                nn = self.fitted(backend, path)
                self.assert_equivalent(nn.query(3, self.Q, exclude_ids=[-1] * 5), self.P, self.Q, 3)
                last = self.P[-1:]  # -1 must not wrap round to the last point
                self.assertEqual(nn.query(1, last, exclude_ids=[-1]).ids[0, 0], 59)
                full = nn.query(70, self.Q, exclude_ids=[-1] * 5).ids
                self.assertTrue(np.all(full[:, :60] >= 0) and np.all(full[:, 60:] == -1))
                for bad in ([60] * 5, [-2] * 5):
                    with self.assertRaises(ValueError):
                        nn.query(3, self.Q, exclude_ids=bad)
                with self.assertRaises(ValueError):
                    nn.query(3, self.Q, exclude_ids=[0, 1])

    def test_query_shapes(self):
        """C11-C13."""
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                nn = self.fitted(backend, path)
                with self.assertRaises(ValueError):
                    nn.query(2, self.Q[None])
                self.assertEqual(nn.query(2, self.Q[0]).ids.shape, (1, 2))
                self.assertEqual(nn.query(2, np.empty((0, 3))).ids.shape, (0, 2))
                self.assertEqual(nn.query_radius(0.5, np.empty((0, 3))), [])
                self.assertEqual(nn.query_radius(0.5, np.empty((0, 3)), pad=True).ids.shape[0], 0)

    def test_duplicates_and_degenerate_sets(self):
        """C15-C17, C20."""
        Z = np.full((30, 3), 0.25)
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                ids, dist = make(backend, path).fit(Z).query(4, Z[:1])
                np.testing.assert_array_equal(ids[0], [0, 1, 2, 3])
                np.testing.assert_array_equal(dist[0], np.zeros(4))
                ids, dist = make(backend, path).fit(Z[:10], Z[10:]).query(3)
                self.assertTrue(np.all(ids != np.arange(10, 30)[:, None]), "self returned")
                np.testing.assert_array_equal(dist, np.zeros((20, 3)))
                ids, _ = make(backend, path).fit(self.P[:1]).query(2, self.P[:1])
                np.testing.assert_array_equal(ids[0], [0, -1])
                line = np.random.default_rng(12).random((40, 1))
                self.assert_equivalent(make(backend, path).fit(line).query(40, line[:5]), line, line[:5], 40)

    def test_input_layouts_agree(self):
        """C18-C19: dtype, memory order and strides do not change answers."""
        rng = np.random.default_rng(13)
        wide = rng.random((80, 6))
        base = np.ascontiguousarray(wide[:, ::2])
        layouts = {
            "fortran": np.asfortranarray(base),
            "strided": wide[:, ::2],
        }
        ints = rng.integers(-5, 5, (80, 3))
        tiny = np.where(base < 0.2, 5e-324, base)
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                want = make(backend, path).fit(base).query(5, base[:10])
                for name, P in layouts.items():
                    with self.subTest(layout=name):
                        self.assert_identical(make(backend, path).fit(P).query(5, P[:10]), want)
                self.assert_equivalent(make(backend, path).fit(ints).query(5, ints[:10]), ints, ints[:10], 5)
                self.assert_equivalent(make(backend, path).fit(tiny).query(5, tiny[:10]), tiny, tiny[:10], 5)
                f32 = base.astype(np.float32)
                f64 = f32.astype(np.float64)
                self.assert_identical(
                    make(backend, path).fit(f32).query(5, f32[:10]), make(backend, path).fit(f64).query(5, f64[:10])
                )

    def test_dimension_mismatch_and_unfitted(self):
        """C21-C22."""
        for backend, path in self.each():
            with self.subTest(backend=backend, path=path):
                nn = self.fitted(backend, path)
                for call in (
                    lambda: nn.update(np.zeros((10, 4))),
                    lambda: nn.promote(np.zeros((2, 4))),
                    lambda: nn.query(2, np.zeros((2, 4))),
                    lambda: nn.query_radius(0.1, np.zeros((2, 4))),
                ):
                    self.assert_unchanged(nn, call)
                fresh = ToroidalNN(backend=backend, **PATHS[path])
                for call in (
                    lambda: fresh.query(2, self.Q),
                    lambda: fresh.update(self.Q),
                    lambda: fresh.promote(self.Q),
                ):
                    with self.assertRaises(RuntimeError):
                        call()


class TestPrefilterExactness(ContractCase):
    """N1 guard: a quantised lower-bound prefilter may drop work, never answers.

    Native results on the tuned LSH path are compared bit for bit with the
    pure-Python reference. The data is dyadic (multiples of ``2**-16``, so
    float32 is exact and summation order cannot matter) and deliberately
    awkward for a ``u8`` grid: a third of the coordinates sit on cell walls
    (``j/256`` and its two grid neighbours), on the 0/1 seam, or at 1.0.
    The point set is large enough that every heap fills, so a bound is
    actually consulted; a bound that is even one cell too large loses a
    neighbour here.
    """

    N = 1500

    @staticmethod
    def awkward(rng, n, d):
        X = dyadic(rng, (n, d))
        walls = rng.integers(0, 256, (n, d)) / 256.0 + rng.integers(-1, 2, (n, d)) * GRID
        special = rng.choice([0.0, 1.0, GRID, 1.0 - GRID, 0.5], (n, d))
        pick = rng.random((n, d))
        X = np.where(pick < 0.2, walls, np.where(pick < 0.35, special, X))
        return np.clip(X, 0.0, 1.0)

    def pair(self, d, k, **kw):
        rng = np.random.default_rng(d * 7 + k)
        P = self.awkward(rng, self.N, d)
        C = self.awkward(rng, 600, d)
        Q = self.awkward(rng, 200, d)
        runs = {}
        for backend in BACKENDS:
            nn = ToroidalNN(seed=3, backend=backend, brute_threshold=1, **kw).fit(P, C, k=k)
            runs[backend] = (nn.query(k), nn.query(k, Q), nn.query_radius(0.05 * d, Q, pad=True))
        return runs

    def test_knn_matches_reference_bit_for_bit(self):
        for d in (4, 16, 17, 24, 33, 64):
            for k in (1, 40):
                runs = self.pair(d, k)
                ref = runs["python"]
                for backend, got in runs.items():
                    for which in (0, 1):
                        with self.subTest(backend=backend, d=d, k=k, which=which):
                            self.assert_identical(got[which], ref[which])

    def test_radius_matches_reference_bit_for_bit(self):
        for d in (16, 17, 32, 33):
            runs = self.pair(d, 8)
            ref = runs["python"][2]
            for backend, got in runs.items():
                with self.subTest(backend=backend, d=d):
                    self.assert_identical(got[2], ref)

    def test_after_moves_across_the_seam(self):
        d, k = 32, 8
        rng = np.random.default_rng(21)
        P, C = self.awkward(rng, 2000, d), self.awkward(rng, 600, d)
        fresh = self.awkward(rng, 200, d)
        out = {}
        for backend in BACKENDS:
            nn = ToroidalNN(seed=5, backend=backend, brute_threshold=1).fit(P, C, k=k)
            res = [nn.query(k)]
            for step in (GRID * 300, -GRID * 700, 0.5):
                nn.update(np.mod(nn.candidates + step, 1.0))
                res.append(nn.query(k))
            nn.promote(fresh)
            res.append(nn.query(k))
            out[backend] = res
        for backend, res in out.items():
            for i, (got, want) in enumerate(zip(res, out["python"], strict=True)):
                with self.subTest(backend=backend, step=i):
                    self.assert_identical(got, want)


class TestPrefilterEdges(ContractCase):
    """N1 edge cases on continuous data against the float64 oracle.

    Every point is a candidate on the exhaustive path, so the batched join
    scores (and the prefilter prunes) real pairs; 80 external queries are past
    the join's 64-query minimum. The cases are the ones a quantised bound can
    get wrong: a tiny k-th distance (so a tiny slack), a zero one (duplicates),
    f64 queries that round up to 1.0 in float32, ties on a coarse lattice,
    mixed magnitudes across the seam, the largest width the filter serves and
    the first one past it.
    """

    def check(self, P, Q, ks, dims_note, self_join=False):
        P, Q = np.mod(P, 1.0), np.mod(Q, 1.0)
        D = torus(Q, P)
        for backend, path in self.each():
            nn = make(backend, path).fit(P)
            for k in ks:
                with self.subTest(case=dims_note, backend=backend, path=path, k=k):
                    self.assert_equivalent(nn.query(k, Q), P, Q, k, D=D)

    def test_tight_clusters_on_the_seam(self):
        """k-th distances ~1e-3: the smallest slack, coordinates 1 - 1e-4 against 1e-4."""
        rng = np.random.default_rng(31)
        for d in (17, 33):
            c = np.zeros((4, d))
            c[1] = 0.5
            c[2, : d // 2] = 0.9999
            lab = rng.integers(0, 4, 300)
            P = c[lab] + rng.normal(0, 1e-4, (300, d))
            Q = c[rng.integers(0, 4, 80)] + rng.normal(0, 1e-4, (80, d))
            self.check(P, Q, (1, 5, 40), f"seam-cluster d={d}")

    def test_duplicates_and_zero_distances(self):
        rng = np.random.default_rng(32)
        for d in (17, 32):
            base = rng.random((20, d))
            P = np.repeat(base, 15, axis=0)  # every point has 14 exact copies
            Q = np.vstack([base, base + 1e-7, rng.random((40, d))])
            self.check(P, Q, (1, 14, 15, 16, 60), f"duplicates d={d}")

    def test_queries_that_round_up_to_one(self):
        """f64 queries within 2**-30 of 1.0 become 1.0f32 (cell 0): points near 0 are close."""
        rng = np.random.default_rng(33)
        for d in (17, 33):
            P = np.concatenate([rng.random((150, d)) * 1e-3, rng.random((150, d))])
            Q = 1.0 - rng.random((80, d)) * 2.0**-30
            Q[40:] = rng.random((40, d)) * 1e-4
            self.check(P, Q, (1, 8, 40), f"rounds-to-1 d={d}")

    def test_mixed_magnitudes_across_the_seam(self):
        """a = 1 - j 2**-24 against tiny b: the inexact-subtraction terms."""
        rng = np.random.default_rng(34)
        for d in (17, 32, 33):
            tiny = 10.0 ** rng.uniform(-9, -3, (200, d))
            top = 1.0 - rng.integers(1, 40, (200, d)) * 2.0**-24
            P = np.where(rng.random((200, d)) < 0.5, tiny, top)
            Q = np.where(
                rng.random((80, d)) < 0.5,
                10.0 ** rng.uniform(-9, -3, (80, d)),
                1.0 - rng.integers(1, 40, (80, d)) * 2.0**-24,
            )
            self.check(P, Q, (1, 8, 40), f"mixed-magnitude d={d}")

    def test_lattice_ties_are_exact(self):
        """A 1/8 lattice has hundreds of exact ties: ids must follow (distance, id)."""
        rng = np.random.default_rng(35)
        for d in (17, 32, 33):
            P = rng.integers(0, 8, (300, d)) / 8.0
            Q = rng.integers(0, 8, (80, d)) / 8.0
            want = {k: oracle_knn(P, Q, k) for k in (1, 8, 40)}
            for backend, path in self.each():
                nn = make(backend, path).fit(P)
                for k, ref in want.items():
                    with self.subTest(case=f"lattice d={d}", backend=backend, path=path, k=k):
                        self.assert_identical(nn.query(k, Q), ref)

    def test_self_join_with_exclusion_in_the_batched_join(self):
        rng = np.random.default_rng(36)
        d = 24
        P, C = rng.random((100, d)), rng.random((140, d))  # 140 candidates: batched
        ids = np.arange(100, 240)
        allp = np.vstack([P, C])
        D = torus(C, allp)
        for backend, path in self.each():
            nn = make(backend, path).fit(P, C)
            for k in (1, 8, 40):
                with self.subTest(backend=backend, path=path, k=k):
                    self.assert_equivalent(nn.query(k), allp, C, k, ids, D=D)

    def test_widest_filtered_dimension_and_the_first_unfiltered_one(self):
        rng = np.random.default_rng(37)
        for d in (4096, 4097):
            P, Q = rng.random((90, d)), rng.random((70, d))
            self.check(P, Q, (1, 8), f"wide d={d}")


@unittest.skipUnless(SLOW, "slow tier: set TORANN_SLOW=1")
class TestSlowGeometry(ContractCase):
    """T6 hash law and T8 wrapped clusters (statistical; fixed seeds)."""

    def test_collision_law_across_the_seam(self):
        """Per digit P[collide] = max(0, 1 - B*delta) for pairs straddling 0/1."""
        trials = 20000
        for B in (2, 3, 7, 8):
            for delta in (0.0, 1 / (2 * B), 1 / B, 0.5):
                rng = np.random.default_rng(B * 100 + int(delta * 64))
                x = np.mod(rng.uniform(-delta, delta, trials), 1.0)  # around the seam
                y = np.mod(x + delta, 1.0)
                u = rng.random(trials)
                cx = np.minimum(np.floor(B * np.mod(x + u, 1.0)), B - 1)
                cy = np.minimum(np.floor(B * np.mod(y + u, 1.0)), B - 1)
                p = max(0.0, 1 - B * delta)
                half = 3.3 * np.sqrt(max(p * (1 - p), 1e-12) / trials)  # 99.9% band
                with self.subTest(B=B, delta=delta):
                    self.assertLessEqual(abs((cx == cy).mean() - p), half + 1e-12)

    def test_tables_follow_the_law(self):
        """The index's own codes, not a re-derivation: codes in [0, B-1]
        for every stored point, including ones reduced from 1.0."""
        P = np.vstack([np.random.default_rng(14).random((5000, 4)), [[-1e-20, 0.0, 1 - 2.0**-53, 0.5]]])
        for backend in BACKENDS:
            nn = ToroidalNN(seed=2, backend=backend, brute_threshold=1).fit(P)
            keys = nn._impl.tables()
            self.assertTrue(all(np.all(np.asarray(v) >= 0) for v in keys.values()))
            frac = np.mod(nn._arena[:, nn._S] + nn._U, 1.0) * nn._B
            self.assertTrue(np.all(np.minimum(frac.astype(np.int64), nn._B - 1) < nn._B))

    def test_clusters_on_the_seam_recall_like_interior_ones(self):
        rng = np.random.default_rng(15)
        d, n, k = 16, 16384, 10

        def blobs(centres):
            lab = rng.integers(0, len(centres), n)
            return np.mod(np.asarray(centres)[lab] + rng.normal(0, 0.04, (n, d)), 1.0)

        seam = blobs([np.full(d, c) for c in (0.0, 0.99, 0.01, 0.995)])
        inner = blobs([np.full(d, c) for c in (0.3, 0.45, 0.55, 0.7)])
        for backend in BACKENDS:
            got = {}
            for name, X in (("seam", seam), ("inner", inner)):
                ids, _ = ToroidalNN(seed=0, backend=backend, brute_threshold=1).fit(X).query(k, X[:300])
                want, _ = oracle_knn(X, X[:300], k)
                got[name] = np.mean([len(np.intersect1d(a, b)) / k for a, b in zip(ids, want)])
            with self.subTest(backend=backend, **got):
                self.assertGreaterEqual(got["seam"], got["inner"] - 0.03)


if __name__ == "__main__":
    unittest.main()
