"""Conformance suite.

Backend-independent behaviour (brute mode, validation) is tested once.
Everything that exercises the LSH path runs once per *installed* backend
(python always; c / rust when their wheels are present) — the same classes
are the conformance suite a native backend must pass. Cross-backend
equivalence (byte-identical tables, matching results) is tested at the end.
"""

import io
import logging
import unittest
import unittest.mock

import numpy as np

from torann import Neighbours, ToroidalNN, available_backends, rust, tune_thresholds
from torann.brute import BruteIndex, exact_knn, exact_radius, knn_scan, pairwise_l1, pairwise_lp
from torann.wrapper import _torus_gaps, _wall_hit_rate, _wall_profile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKENDS = available_backends()


def torus_l1(Q, P):
    diff = np.abs(Q[:, None, :] - P[None, :, :])
    return np.minimum(diff, 1.0 - diff).sum(-1)


def brute_knn(points, queries, k, exclude_ids=None):
    """Reference toroidal-L1 k-NN."""
    D = torus_l1(queries, points)
    if exclude_ids is not None:
        D[np.arange(len(queries)), exclude_ids] = np.inf
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(D, idx, axis=1)


def recall(approx_idx, exact_idx):
    hits = sum(len(np.intersect1d(a, e)) for a, e in zip(approx_idx, exact_idx))
    return hits / exact_idx.size


def ref_keys(nn, ids):
    """Hash keys recomputed from the facade's parameters — the normative
    L1 formula, independent of any backend's tables."""
    X = nn._arena[ids]
    keys = np.empty((nn._L, len(ids)), dtype=np.int64)
    for t in range(nn._L):
        frac = np.mod(X[:, nn._S[t]] + nn._U[t], 1.0) * nn._B
        codes = np.minimum(frac.astype(np.int64), nn._B - 1)
        keys[t] = (codes * nn._pw).sum(axis=1)
    return keys


class TestBruteMode(unittest.TestCase):
    """Below brute_threshold every answer must be exact (backend-free path)."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.static = rng.random((400, 4))
        self.cands = rng.random((100, 4))
        self.all_pts = np.vstack([self.static, self.cands])

    def test_mode_and_tiers(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        self.assertFalse(nn.is_approximate)
        self.assertEqual((nn.n_static, nn.n_candidates), (400, 100))

    def test_default_query_is_candidate_self_join(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands, k=10)
        idx, dst = nn.query()
        self.assertEqual(idx.shape, (100, 10))
        cand_ids = np.arange(400, 500)
        self.assertFalse((idx == cand_ids[:, None]).any())
        ridx, rdst = brute_knn(self.all_pts, self.cands, 10, cand_ids)
        np.testing.assert_allclose(dst, rdst, atol=1e-12)

    def test_explicit_queries(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        Q = np.random.default_rng(1).random((20, 4))
        idx, dst = nn.query(k=5, queries=Q)
        _, rdst = brute_knn(self.all_pts, Q, 5)
        np.testing.assert_allclose(dst, rdst, atol=1e-12)

    def test_wrap_around(self):
        static = np.full((10, 3), 0.5)
        static[0] = [0.99, 0.5, 0.5]
        nn = ToroidalNN(seed=0).fit(static, np.array([[0.01, 0.5, 0.5]]))
        idx, dst = nn.query(k=1)
        self.assertEqual(idx[0, 0], 0)
        self.assertAlmostEqual(dst[0, 0], 0.02, places=9)

    def test_k_larger_than_n_pads(self):
        nn = ToroidalNN(seed=0).fit(self.static[:5], self.cands[:2])
        idx, dst = nn.query(k=10)
        self.assertTrue((idx[:, :6] >= 0).all())  # 7 points minus self
        self.assertTrue((idx[:, 6:] == -1).all())
        self.assertTrue(np.isinf(dst[:, 6:]).all())

    def test_out_of_range_inputs_reduced_mod_1(self):
        """The facade owns the [0,1) domain: any input is reduced mod 1."""
        nn = ToroidalNN(seed=0).fit(self.static + 3.0, self.cands - 2.0)
        ref = ToroidalNN(seed=0).fit(self.static, self.cands)
        a = nn.query(k=5)
        b = ref.query(k=5)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1], atol=1e-12)

    def test_explicit_exclude_ids(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        Q = self.all_pts[:20]  # query the indexed points themselves
        ex = np.arange(20, dtype=np.int64)
        idx, dst = nn.query(k=3, queries=Q, exclude_ids=ex)
        self.assertFalse((idx == ex[:, None]).any())  # self never returned
        self.assertTrue((dst[:, 0] > 0).all())

    def test_promote_empty_batch(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        nn.promote()  # no new batch
        self.assertEqual((nn.n_static, nn.n_candidates), (500, 0))
        with self.assertRaises(ValueError):
            nn.query(k=2)  # no candidate tier to self-join
        idx, _ = nn.query(k=2, queries=self.cands[:4])
        self.assertEqual(idx.shape, (4, 2))

    def test_fit_from_scratch_empty_static(self):
        """The from-scratch case: no anchors, first batch repels itself."""
        rng = np.random.default_rng(3)
        cands = rng.random((40, 4))
        nn = ToroidalNN(seed=0, brute_threshold=64).fit(np.empty((0, 4)), cands, k=5)
        self.assertEqual((nn.n_static, nn.n_candidates), (0, 40))
        idx, dst = nn.query(k=5)
        self.assertEqual(idx.shape, (40, 5))
        self.assertFalse((idx == np.arange(40)[:, None]).any())  # self excluded
        nn.promote(rng.random((40, 4)))  # crosses threshold -> LSH build
        self.assertEqual((nn.n_static, nn.n_candidates), (40, 40))
        self.assertTrue(nn.is_approximate)
        idx2, _ = nn.query(k=5)
        self.assertEqual(idx2.shape, (40, 5))
        with self.assertRaises(ValueError):
            ToroidalNN(seed=0).fit(np.empty((0, 4)))  # no points at all

    def test_candidates_property_tracks_updates(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        moved = np.mod(self.cands + 0.25, 1.0)
        nn.update(moved)
        np.testing.assert_allclose(nn.candidates, moved, atol=1e-12)
        self.assertEqual(nn.dimensions, 4)


class TestLSHMode(unittest.TestCase):
    """Above brute_threshold: approximate filter, exact distances."""

    BACKEND = "python"
    D = 8

    def setUp(self):
        rng = np.random.default_rng(7)
        self.static = rng.random((7000, self.D))
        self.cands = rng.random((1000, self.D))
        self.all_pts = np.vstack([self.static, self.cands])
        self.cand_ids = np.arange(7000, 8000)

    def make(self, **kw):
        # These classes test LSH behaviour, so they select LSH rather than
        # inheriting whichever crossover the defaults currently hold.
        # They used to rely on 7000 points clearing a threshold of 512; when
        # the compiled brute kernel moved that threshold to 8192 they
        # silently became brute-mode tests and ten of them failed.
        kw.setdefault("seed", 0)
        kw.setdefault("brute_threshold", 1)
        return ToroidalNN(backend=self.BACKEND, **kw)

    def test_is_approximate(self):
        nn = self.make().fit(self.static, self.cands)
        self.assertTrue(nn.is_approximate)
        self.assertEqual(nn.backend_name, self.BACKEND)

    def test_recall_of_default_query(self):
        k = 2 * self.D
        nn = self.make().fit(self.static, self.cands, k=k)
        idx, dst = nn.query()
        ridx, _ = brute_knn(self.all_pts, self.cands, k, self.cand_ids)
        r = recall(idx, ridx)
        logger.info("[%s] LSH recall@%d: %.3f", self.BACKEND, k, r)
        self.assertGreater(r, 0.85)
        self.assertTrue((idx >= 0).all())  # exact fallback fills all rows

    def test_distances_are_true_l1_and_sorted(self):
        nn = self.make().fit(self.static, self.cands, k=16)
        idx, dst = nn.query()
        d_true = np.take_along_axis(torus_l1(self.cands, self.all_pts), idx, axis=1)
        np.testing.assert_allclose(dst, d_true, atol=1e-5)  # float32 refine
        self.assertTrue((np.diff(dst, axis=1) >= -1e-6).all())

    def test_wrap_around(self):
        static = self.static.copy()
        static[0] = 0.999
        nn = self.make().fit(static, np.full((1, self.D), 0.001))
        idx, dst = nn.query(k=1)
        self.assertEqual(idx[0, 0], 0)
        self.assertAlmostEqual(dst[0, 0], 0.002 * self.D, places=5)

    def test_determinism(self):
        a = self.make(seed=99).fit(self.static, self.cands).query(k=16)
        b = self.make(seed=99).fit(self.static, self.cands).query(k=16)
        np.testing.assert_array_equal(a[0], b[0])

    def test_tuning_respects_explicit_args(self):
        nn = self.make(resolution=2, num_tables=5, dims_per_table=7).fit(self.static, self.cands)
        self.assertEqual((nn._B, nn._L, nn._K), (2, 5, 7))

    def test_tables_match_normative_hash(self):
        """The backend's tables must be exactly the normative L1 hash of the
        current positions, key-sorted with ids aligned."""
        nn = self.make().fit(self.static, self.cands)
        tab = nn._impl.tables()
        fresh_s = ref_keys(nn, np.arange(nn.n_static))
        fresh_c = ref_keys(nn, np.arange(nn.n_static, nn.n_points))
        np.testing.assert_array_equal(tab["cand_keys"], fresh_c)
        for t in range(nn._L):
            np.testing.assert_array_equal(np.sort(fresh_s[t]), tab["static_keys"][t])
            np.testing.assert_array_equal(fresh_s[t][tab["static_ids"][t]], tab["static_keys"][t])

    def test_radius_hint_tuning(self):
        """fit(radius=...) is the alternative tuning hint; the index must
        come up approximate, deterministic and usable."""
        nn = self.make().fit(self.static, self.cands, radius=0.25)
        self.assertTrue(nn.is_approximate)
        idx, dst = nn.query(k=8)
        self.assertTrue((idx >= 0).all())
        self.assertTrue((np.diff(dst, axis=1) >= -1e-6).all())

    def test_query_radius_explicit_queries(self):
        nn = self.make().fit(self.static, self.cands)
        Q = np.random.default_rng(2).random((16, self.D))
        res = nn.query_radius(0.4, queries=Q)
        self.assertEqual(len(res), 16)
        D = torus_l1(Q, self.all_pts)
        for i, (ids, dst) in enumerate(res):
            self.assertTrue((D[i, ids] <= 0.4 + 1e-6).all())  # true hits only
            np.testing.assert_allclose(dst, D[i, ids], atol=1e-5)

    def test_starved_index_fills_rows_via_relaxation(self):
        """One table, no probes, oversized keys: buckets are near-empty, so
        most queries must be completed by prefix relaxation, not brute force.
        """
        nn = self.make(num_tables=1, probes=0, dims_per_table=14).fit(self.static, self.cands, k=16)
        idx, dst = nn.query()
        self.assertTrue((idx >= 0).all())
        self.assertTrue(np.isfinite(dst).all())
        cand_ids = np.arange(7000, 8000)
        self.assertFalse((idx == cand_ids[:, None]).any())
        d_true = np.take_along_axis(torus_l1(self.cands, self.all_pts), idx, axis=1)
        np.testing.assert_allclose(dst, d_true, atol=1e-5)
        self.assertTrue((np.diff(dst, axis=1) >= -1e-6).all())


class TestLifecycle(unittest.TestCase):
    """The ESS loop: fit -> (query, update)* -> promote -> repeat."""

    BACKEND = "python"
    D = 8

    def setUp(self):
        rng = np.random.default_rng(3)
        self.rng = rng
        self.static = rng.random((6000, self.D))
        self.cands = rng.random((800, self.D))

    def make(self, **kw):
        # These classes test LSH behaviour, so they select LSH rather than
        # inheriting whichever crossover the defaults currently hold.
        # They used to rely on 7000 points clearing a threshold of 512; when
        # the compiled brute kernel moved that threshold to 8192 they
        # silently became brute-mode tests and ten of them failed.
        kw.setdefault("seed", 0)
        kw.setdefault("brute_threshold", 1)
        return ToroidalNN(backend=self.BACKEND, **kw)

    def _reference(self, nn, k):
        all_pts = np.vstack([nn._arena[: nn.n_static], nn.candidates])
        cand_ids = np.arange(nn.n_static, nn.n_points)
        return brute_knn(all_pts, nn.candidates.copy(), k, cand_ids)

    def test_recall_holds_through_epochs(self):
        k = 2 * self.D
        nn = self.make().fit(self.static, self.cands, k=k)
        for _ in range(5):
            step = self.rng.normal(0, 0.01, (800, self.D))
            nn.update(np.mod(nn.candidates + step, 1.0))
        idx, _ = nn.query()
        ridx, _ = self._reference(nn, k)
        r = recall(idx, ridx)
        logger.info("[%s] recall after 5 update epochs: %.3f", self.BACKEND, r)
        self.assertGreater(r, 0.85)

    def test_update_touches_only_changed_keys(self):
        nn = self.make().fit(self.static, self.cands)
        keys_before = nn._impl.tables()["cand_keys"]
        step = self.rng.normal(0, 0.002, (800, self.D))
        nn.update(np.mod(nn.candidates + step, 1.0))
        changed = (nn._impl.tables()["cand_keys"] != keys_before).mean()
        logger.info("[%s] keys changed by tiny step: %.3f", self.BACKEND, changed)
        self.assertLess(changed, 0.30)  # most keys survive a small step

    def test_selective_update_equals_full_rebuild(self):
        nn = self.make().fit(self.static, self.cands)
        for sigma in (0.001, 0.02, 0.3):  # few, some, most keys change
            step = self.rng.normal(0, sigma, (800, self.D))
            nn.update(np.mod(nn.candidates + step, 1.0))
            tab = nn._impl.tables()
            fresh = ref_keys(nn, np.arange(nn.n_static, nn.n_points))
            np.testing.assert_array_equal(tab["cand_keys"], fresh)
            for t in range(nn._L):
                np.testing.assert_array_equal(np.sort(fresh[t]), tab["cand_sorted_keys"][t])
                np.testing.assert_array_equal(
                    fresh[t][tab["cand_sorted_ids"][t] - nn.n_static], tab["cand_sorted_keys"][t]
                )

    def test_promote_merge_equals_full_rebuild(self):
        nn = self.make().fit(self.static, self.cands)
        new_batch = self.rng.random((500, self.D))
        nn.promote(new_batch)
        self.assertEqual((nn.n_static, nn.n_candidates), (6800, 500))
        tab = nn._impl.tables()
        keys = ref_keys(nn, np.arange(nn.n_static))
        for t in range(nn._L):  # merged arrays must be perfectly sorted
            self.assertTrue((np.diff(tab["static_keys"][t]) >= 0).all())
            np.testing.assert_array_equal(np.sort(keys[t]), tab["static_keys"][t])
            np.testing.assert_array_equal(keys[t][tab["static_ids"][t]], tab["static_keys"][t])

    def test_promote_freezes_current_positions(self):
        nn = self.make().fit(self.static, self.cands)
        nn.update(self.rng.random((800, self.D)))  # teleport before freezing
        nn.promote(self.rng.random((100, self.D)))
        tab = nn._impl.tables()
        keys = ref_keys(nn, np.arange(nn.n_static))
        for t in range(nn._L):  # frozen keys must reflect current positions
            np.testing.assert_array_equal(np.sort(keys[t]), tab["static_keys"][t])

    def test_full_loop_correctness(self):
        k = 2 * self.D
        nn = self.make().fit(self.static, self.cands, k=k)
        for batch in range(3):
            for _ in range(3):
                idx, dst = nn.query()
                self.assertTrue((idx >= 0).all())
                step = self.rng.normal(0, 0.01, (nn.n_candidates, self.D))
                nn.update(np.mod(nn.candidates + step, 1.0))
            nn.promote(self.rng.random((300, self.D)))
        self.assertEqual(nn.n_points, 6800 + 3 * 300)
        idx, _ = nn.query()
        ridx, _ = self._reference(nn, k)
        self.assertGreater(recall(idx, ridx), 0.80)

    def test_promote_crosses_brute_threshold(self):
        nn = self.make(brute_threshold=1500).fit(self.static[:1000], self.cands[:400])
        self.assertFalse(nn.is_approximate)
        nn.promote(self.rng.random((300, self.D)))  # 1700 > 1500
        self.assertTrue(nn.is_approximate)
        self.assertEqual((nn.n_static, nn.n_candidates), (1400, 300))
        idx, dst = nn.query(k=8)
        ridx, rdst = self._reference(nn, 8)
        self.assertGreater(recall(idx, ridx), 0.85)


class TestRangeQueries(unittest.TestCase):
    BACKEND = "python"
    D = 6

    def setUp(self):
        rng = np.random.default_rng(5)
        self.static = rng.random((4500, self.D))
        self.cands = rng.random((600, self.D))
        self.all_pts = np.vstack([self.static, self.cands])
        self.radius = 0.3

    def make(self, **kw):
        # These classes test LSH behaviour, so they select LSH rather than
        # inheriting whichever crossover the defaults currently hold.
        # They used to rely on 7000 points clearing a threshold of 512; when
        # the compiled brute kernel moved that threshold to 8192 they
        # silently became brute-mode tests and ten of them failed.
        kw.setdefault("seed", 0)
        kw.setdefault("brute_threshold", 1)
        return ToroidalNN(backend=self.BACKEND, **kw)

    def _exact_sets(self):
        D = torus_l1(self.cands, self.all_pts)
        D[np.arange(600), np.arange(4500, 5100)] = np.inf
        return [np.flatnonzero(row <= self.radius) for row in D]

    def test_exact_flag(self):
        nn = self.make().fit(self.static, self.cands)
        self.assertTrue(nn.is_approximate)
        res = nn.query_radius(self.radius, exact=True)
        for (ids, dst), ref in zip(res, self._exact_sets()):
            np.testing.assert_array_equal(np.sort(ids), ref)
            self.assertTrue((np.diff(dst) >= -1e-12).all())

    def test_post_filter_returns_only_true_hits(self):
        nn = self.make().fit(self.static, self.cands)
        res = nn.query_radius(self.radius)
        refs = self._exact_sets()
        found = total = 0
        for (ids, dst), ref in zip(res, refs):
            self.assertTrue(np.isin(ids, ref).all())  # no false positives
            self.assertTrue((np.diff(dst) >= -1e-6).all())
            found += len(ids)
            total += len(ref)
        r = found / max(1, total)
        logger.info("[%s] range recall: %.3f", self.BACKEND, r)
        self.assertGreater(r, 0.5)

    def test_brute_mode(self):
        nn = self.make(brute_threshold=10_000).fit(self.static, self.cands)
        res = nn.query_radius(self.radius)
        for (ids, _), ref in zip(res, self._exact_sets()):
            np.testing.assert_array_equal(np.sort(ids), ref)


class TestCpuFloor(unittest.TestCase):
    """The AVX2 gate in front of the compiled backend.

    Published x86-64 wheels are built with ``+avx2,+fma`` because the floor
    is worth 25%. The gate is what makes that safe on a CPU
    without it: the module falls back to the pure-Python backend instead of
    taking SIGILL, which Python cannot catch and which would surface as a
    bare crash with no traceback.
    """

    @staticmethod
    def _platform(machine, **system):
        """Patch what the detector sees: machine, and how `system` behaves."""
        return (
            unittest.mock.patch.object(rust.platform, "machine", return_value=machine),
            unittest.mock.patch.object(rust.platform, "system", **system),
        )

    def test_detection_answers_and_never_raises(self):
        self.assertIsInstance(rust._cpu_supports_avx2(), bool)

    def test_non_x86_needs_no_floor(self):
        """aarch64 has NEON as baseline, so there is nothing to gate on."""
        with unittest.mock.patch.object(rust.platform, "machine", return_value="aarch64"):
            self.assertTrue(rust._cpu_supports_avx2())

    def test_unknown_platform_assumes_supported(self):
        """
        The fallback costs 25-75x, so an undetectable CPU must not be assumed
        old: guessing "no" on a machine that was fine is the more damaging
        error of the two.
        """
        machine, system = self._platform("x86_64", return_value="Plan9")
        with machine, system:
            self.assertTrue(rust._cpu_supports_avx2())

    def test_detection_failure_assumes_supported(self):
        """Detection must never be able to break the import."""
        machine, system = self._platform("x86_64", side_effect=OSError("boom"))
        with machine, system:
            self.assertTrue(rust._cpu_supports_avx2())

    def test_linux_reads_the_cpuinfo_flags(self):
        for flags, expected in ((" fpu vme avx2 bmi1 ", True), (" fpu vme avx bmi1 ", False), (" sse2 ssse3 ", False)):
            with (
                unittest.mock.patch.object(rust.platform, "machine", return_value="x86_64"),
                unittest.mock.patch.object(rust.platform, "system", return_value="Linux"),
                unittest.mock.patch("builtins.open", return_value=io.StringIO(f"flags\t:{flags}\n")),
            ):
                self.assertIs(rust._cpu_supports_avx2(), expected, flags)


class TestValidation(unittest.TestCase):
    def test_query_before_fit(self):
        with self.assertRaises(RuntimeError):
            ToroidalNN().query(k=1)

    def test_bad_params(self):
        for kwargs in ({"resolution": 1}, {"num_tables": 0}, {"probes": -1}, {"dims_per_table": 0}):
            with self.assertRaises(ValueError):
                ToroidalNN(**kwargs)

    def test_unknown_backend(self):
        rng = np.random.default_rng(0)
        nn = ToroidalNN(seed=0, backend="fortran", brute_threshold=10)
        with self.assertRaises(ValueError):
            nn.fit(rng.random((50, 3)))

    def test_no_candidates_needs_explicit_queries(self):
        nn = ToroidalNN(seed=0).fit(np.random.default_rng(0).random((50, 3)))
        with self.assertRaises(ValueError):
            nn.query(k=2)
        idx, _ = nn.query(k=2, queries=np.zeros((1, 3)))
        self.assertEqual(idx.shape, (1, 2))

    def test_update_shape_checked(self):
        rng = np.random.default_rng(0)
        nn = ToroidalNN(seed=0).fit(rng.random((50, 3)), rng.random((10, 3)))
        with self.assertRaises(ValueError):
            nn.update(rng.random((5, 3)))

    def test_dimension_mismatch(self):
        nn = ToroidalNN(seed=0).fit(np.random.default_rng(0).random((10, 4)))
        with self.assertRaises(ValueError):
            nn.query(k=1, queries=np.zeros((1, 3)))

    def test_bad_k(self):
        nn = ToroidalNN(seed=0).fit(np.random.default_rng(0).random((10, 3)), np.random.default_rng(1).random((4, 3)))
        with self.assertRaises(ValueError):
            nn.query(k=0)

    def test_exclude_ids_shape_checked(self):
        nn = ToroidalNN(seed=0).fit(np.random.default_rng(0).random((10, 3)))
        with self.assertRaises(ValueError):
            nn.query(k=1, queries=np.zeros((2, 3)), exclude_ids=np.array([0], dtype=np.int64))

    def test_available_backends(self):
        names = available_backends()
        self.assertIn("python", names)  # the reference is always present
        self.assertTrue(set(names) <= {"rust", "python"})


@unittest.skipIf(len(BACKENDS) < 2, "only one backend installed")
class TestBackendEquivalence(unittest.TestCase):
    """Native backends must reproduce the reference backend exactly:
    byte-identical tables through the whole lifecycle, identical ids (ties
    are ordered by id on every backend), and distances within two float32
    refine bounds (``2 * d * 2**-23``: the backends sum in different orders)."""

    D = 8

    def _run_lifecycle(self, backend):
        rng = np.random.default_rng(11)
        # LSH on both sides: this compares the tables, and the two backends
        # no longer share a crossover -- python switches at 4096, rust at
        # 8192 now that its brute path is compiled -- so 6800 points would
        # put one on LSH and the other on brute.
        nn = ToroidalNN(seed=4, backend=backend, brute_threshold=1).fit(
            rng.random((6000, self.D)), rng.random((800, self.D)), k=16
        )
        results = [nn.query()]
        for _ in range(3):
            step = rng.normal(0, 0.01, (nn.n_candidates, self.D))
            nn.update(np.mod(nn.candidates + step, 1.0))
            results.append(nn.query())
        nn.promote(rng.random((400, self.D)))
        results.append(nn.query())
        return nn, results

    def test_tables_and_results_match_reference(self):
        ref_nn, ref_res = self._run_lifecycle("python")
        ref_tab = ref_nn._impl.tables()
        for backend in BACKENDS:
            if backend == "python":
                continue
            with self.subTest(backend=backend):
                nn, res = self._run_lifecycle(backend)
                tab = nn._impl.tables()
                for name in ref_tab:  # byte-identical tables
                    np.testing.assert_array_equal(tab[name], ref_tab[name], err_msg=name)
                for (ai, ad), (bi, bd) in zip(ref_res, res):
                    np.testing.assert_array_equal(bi, ai)
                    np.testing.assert_allclose(bd, ad, rtol=0, atol=2 * self.D * 2.0**-23)


class TestMetrics(unittest.TestCase):
    """`toroidal_separation` — the shared design-quality metric.

    ESS scores its designs with this, so the definition is pinned here
    rather than in whichever benchmark happens to call it.
    """

    def test_wrap_is_the_short_way_round(self):
        """The whole point: a pair across the seam is near, not far."""
        from torann.metrics import toroidal_separation

        self.assertAlmostEqual(toroidal_separation(np.array([[0.99], [0.01]])), 0.02)

    def test_matches_a_brute_pairwise_minimum(self):
        """Agrees with an independent O(n^2) scan, wrap included."""
        from torann.metrics import toroidal_separation

        rng = np.random.default_rng(0)
        for d in (1, 2, 8):
            pts = rng.random((60, d))
            D = torus_l1(pts, pts)
            np.fill_diagonal(D, np.inf)
            self.assertAlmostEqual(toroidal_separation(pts), float(D.min()), places=12, msg=f"d={d}")

    def test_inputs_are_reduced_mod_1(self):
        """Same contract as the index: any real input is accepted."""
        from torann.metrics import toroidal_separation

        rng = np.random.default_rng(1)
        pts = rng.random((40, 4))
        for shift in (-3.0, 2.0):
            self.assertAlmostEqual(toroidal_separation(pts), toroidal_separation(pts + shift), places=12)

    def test_degenerate_inputs(self):
        """Fewer than two points has no pair; duplicates are 0 apart."""
        from torann.metrics import toroidal_separation

        self.assertEqual(toroidal_separation(np.zeros((0, 3))), 0.0)
        self.assertEqual(toroidal_separation(np.zeros((1, 3))), 0.0)
        self.assertEqual(toroidal_separation(np.zeros((5, 3))), 0.0)
        with self.assertRaises(ValueError):
            toroidal_separation(np.zeros(5))

    def test_anchors_score_only_pairs_touching_the_batch(self):
        """The refinement contract: anchor-anchor pairs must not floor it.

        Two anchors 0.01 apart and a batch placed far from both. Scoring the
        union would report 0.01 and call a good batch bad.
        """
        from torann.metrics import toroidal_separation

        anchors = np.array([[0.00], [0.01]])
        batch = np.array([[0.40], [0.70]])
        # Each batch point's nearest neighbour is the other one, 0.30 away.
        self.assertAlmostEqual(toroidal_separation(batch, anchors), 0.30, places=12)
        # Scoring the union instead would return the 0.01 anchor gap.
        self.assertAlmostEqual(toroidal_separation(np.vstack([anchors, batch])), 0.01, places=12)

    def test_anchors_none_and_empty_agree(self):
        from torann.metrics import toroidal_separation

        rng = np.random.default_rng(3)
        pts = rng.random((30, 4))
        self.assertEqual(toroidal_separation(pts), toroidal_separation(pts, np.empty((0, 4))))

    def test_a_batch_on_top_of_an_anchor_scores_zero(self):
        """The failure the metric exists to catch."""
        from torann.metrics import toroidal_separation

        anchors = np.array([[0.25, 0.25], [0.75, 0.75]])
        batch = np.array([[0.25, 0.25], [0.10, 0.90]])
        self.assertAlmostEqual(toroidal_separation(batch, anchors), 0.0)

    def test_empty_batch_against_anchors(self):
        """No new points means no pair to score."""
        from torann.metrics import toroidal_separation

        self.assertEqual(toroidal_separation(np.zeros((0, 2)), np.ones((3, 2)) * 0.5), 0.0)

    def test_anchor_shape_is_checked(self):
        from torann.metrics import toroidal_separation

        with self.assertRaises(ValueError):
            toroidal_separation(np.zeros((4, 3)), np.zeros((4, 2)))

    def test_keeps_its_margin_up_to_d32(self):
        """The property the module docstring rests on, asserted.

        A design confined to a tenth of each axis is unambiguously worse
        spread than a uniform one, and separation has to say so *at d=32* —
        which is where Clark-Evans, coverage radius and the discrepancies
        stop resolving. The bar is a 5x margin, far outside seed noise;
        coverage radius on the ESS-vs-LHS comparison manages 0.02% there.
        """
        from torann.metrics import toroidal_separation

        rng = np.random.default_rng(2)
        for d in (2, 8, 32):
            n = 512
            spread = toroidal_separation(rng.random((n, d)))
            clumped = toroidal_separation(rng.random((n, d)) * 0.1)
            self.assertGreater(spread, 5.0 * clumped, msg=f"d={d}")


def _parametrize(base):
    """One subclass per installed native backend (python runs in the base)."""
    for name in BACKENDS:
        if name == "python":
            continue
        cls = type(f"{base.__name__}_{name}", (base,), {"BACKEND": name})
        globals()[cls.__name__] = cls


_parametrize(TestLSHMode)
_parametrize(TestLifecycle)
_parametrize(TestRangeQueries)


class TestRustBruteKernel(unittest.TestCase):
    """The compiled scan against the NumPy one it serves in place of.

    Skipped on a pure-Python install, where `BruteIndex` still runs the NumPy
    path and there is nothing to compare.
    """

    @classmethod
    def setUpClass(cls):
        if rust.brute_knn is None:
            raise unittest.SkipTest("compiled backend not available")

    def _case(self, n, m, d, k, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((n, d)), rng.random((m, d)), k

    def test_it_agrees_with_the_reference_scan(self):
        """Ids exactly, distances to f64 ulp -- not to f32, which is why the
        kernel is f64 despite the LSH path being f32."""
        for n, m, d, k in ((60, 30, 100, 8), (400, 200, 100, 8), (300, 300, 32, 16), (200, 100, 8, 8)):
            with self.subTest(n=n, m=m, d=d, k=k):
                pts, q, k = self._case(n, m, d, k)
                i_np, d_np = exact_knn(pts, q, k)
                i_rs, d_rs = rust.brute_knn(pts, q, k, None)
                np.testing.assert_array_equal(i_np, i_rs)
                np.testing.assert_allclose(d_np, d_rs, atol=1e-12)

    def test_fewer_points_than_k_pads_rather_than_fails(self):
        pts, q, _ = self._case(3, 5, 4, 8)
        idx, dst = rust.brute_knn(pts, q, 8, None)
        self.assertEqual(idx.shape, (5, 8))
        np.testing.assert_array_equal(idx[:, 3:], -1)
        self.assertTrue(np.isinf(dst[:, 3:]).all())

    def test_an_excluded_id_is_never_returned(self):
        """The self-join contract: a query may not find its own point."""
        pts, _, _ = self._case(40, 1, 6, 5)
        ex = np.arange(40, dtype=np.int64)
        idx, _ = rust.brute_knn(pts, pts, 5, ex)
        self.assertFalse((idx == ex[:, None]).any())

    def test_mismatched_widths_are_refused(self):
        pts, _, _ = self._case(10, 1, 4, 3)
        with self.assertRaises(ValueError):
            rust.brute_knn(pts, np.zeros((2, 5)), 3, None)

    def test_it_wraps_like_the_reference(self):
        pts = np.array([[0.99], [0.50]])
        idx, dst = rust.brute_knn(pts, np.array([[0.01]]), 1, None)
        self.assertEqual(int(idx[0, 0]), 0)
        self.assertAlmostEqual(float(dst[0, 0]), 0.02, places=9)


class TestRustRadiusKernel(unittest.TestCase):
    """The compiled range scan against `exact_radius`, which it serves in
    place of below the crossover.

    A range query is checked more strictly than the k-NN one in a way the
    tolerance cannot express: `<=` is a threshold, so the two paths must
    agree on *membership*, and the sets are compared exactly. Distances are
    then compared on those members to f64 ulp.
    """

    @classmethod
    def setUpClass(cls):
        if rust.brute_radius is None:
            raise unittest.SkipTest("compiled backend not available")

    @staticmethod
    def _case(n, m, d, seed=0):
        rng = np.random.default_rng(seed)
        pts = np.ascontiguousarray(rng.random((n, d)))
        q = np.ascontiguousarray(rng.random((m, d)))
        # A radius that admits roughly eight neighbours, which is the regime
        # ESS queries in -- not a round number that would admit none at d=100
        # or everything at d=2.
        kk = min(9, n)
        r = float(np.median(rust.brute_knn(pts, q, kk, None)[1][:, kk - 1]))
        return pts, q, r

    def test_it_agrees_with_the_reference_scan(self):
        for n, m, d in ((60, 30, 100), (400, 200, 100), (200, 100, 8), (50, 50, 2), (300, 64, 1000)):
            with self.subTest(n=n, m=m, d=d):
                pts, q, r = self._case(n, m, d)
                a = exact_radius(pts, q, r, None)
                b = rust.brute_radius(pts, q, r, None)
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])
                np.testing.assert_allclose(a[2], b[2], atol=1e-12)

    def test_an_excluded_id_is_never_returned(self):
        pts, _, r = self._case(80, 1, 6)
        ex = np.arange(80, dtype=np.int64)
        indptr, ids, _ = rust.brute_radius(pts, pts, r, ex)
        for i in range(80):
            self.assertNotIn(i, ids[indptr[i] : indptr[i + 1]])

    def test_rows_are_sorted_by_distance(self):
        pts, q, r = self._case(300, 50, 10)
        indptr, _, dst = rust.brute_radius(pts, q, r, None)
        for i in range(50):
            row = dst[indptr[i] : indptr[i + 1]]
            self.assertTrue((np.diff(row) >= 0).all())

    def test_degenerate_radii_match_the_reference(self):
        """Nothing, and everything. Both are legal and neither may differ."""
        pts, q, _ = self._case(40, 8, 5)
        for r in (0.0, -1.0, 1e9):
            with self.subTest(radius=r):
                a = exact_radius(pts, q, r, None)
                b = rust.brute_radius(pts, q, float(r), None)
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])

    def test_mismatched_widths_are_refused(self):
        pts, _, _ = self._case(10, 1, 4)
        with self.assertRaises(ValueError):
            rust.brute_radius(pts, np.zeros((2, 5)), 1.0, None)

    def test_a_short_exclude_list_is_refused(self):
        """Checked in Rust rather than left to index a slice: the scan runs
        with the GIL released, where the overrun would surface as a panic
        unwinding through rayon instead of as a Python error."""
        pts, q, r = self._case(20, 6, 4)
        with self.assertRaises(ValueError):
            rust.brute_radius(pts, q, r, np.zeros(3, dtype=np.int64))

    def test_it_wraps_like_the_reference(self):
        pts = np.array([[0.99], [0.50]])
        indptr, ids, dst = rust.brute_radius(pts, np.array([[0.01]]), 0.05, None)
        np.testing.assert_array_equal(ids, [0])
        self.assertAlmostEqual(float(dst[0]), 0.02, places=9)


class TestPaddedRangeResults(unittest.TestCase):
    """`pad=True` against the list form it replaces.

    The list is what ESS consumed until it had to loop over it a second time
    to pad it; this asserts the vectorised path produces exactly that, so the
    two passes could be deleted rather than merely bypassed.
    """

    def _index(self):
        rng = np.random.default_rng(11)
        pts = rng.random((400, 20))
        return (ToroidalNN(seed=0).fit(pts[:200], pts[200:]), 200)

    def test_padded_matches_the_list_form(self):
        nn, m = self._index()
        r = float(np.median(nn.query(k=9)[1][:, 8]))
        rows = nn.query_radius(r)
        ids, dst = nn.query_radius(r, pad=True)
        width = max(1, max(len(i) for i, _ in rows))
        self.assertEqual(ids.shape, (m, width))
        for i, (row_ids, row_dst) in enumerate(rows):
            n = len(row_ids)
            np.testing.assert_array_equal(ids[i, :n], row_ids)
            np.testing.assert_array_equal(dst[i, :n], row_dst)
            np.testing.assert_array_equal(ids[i, n:], -1)
            self.assertTrue(np.isinf(dst[i, n:]).all())

    def test_an_all_empty_result_still_has_a_usable_shape(self):
        """Width floors at 1: a downstream kernel indexing `[:, 0]` must not
        meet a zero-width array on the epoch where the radius found nothing."""
        nn, m = self._index()
        ids, dst = nn.query_radius(0.0, pad=True)
        self.assertEqual(ids.shape, (m, 1))
        np.testing.assert_array_equal(ids, -1)
        self.assertTrue(np.isinf(dst).all())


class TestWeightedDirections(unittest.TestCase):
    """The weighted toroidal direction sum, against the NumPy expression it
    replaces -- which is lifted verbatim from ESS's force kernel, so this is
    a conformance test against the caller, not a restatement of the kernel."""

    @classmethod
    def setUpClass(cls):
        if rust.weighted_directions is None:
            raise unittest.SkipTest("compiled backend not available")

    @staticmethod
    def _ref(pts, q, ids, w):
        valid = ids >= 0
        safe = np.where(valid, ids, 0)
        disp = q[:, None, :] - pts[safe]
        disp -= np.round(disp)
        grad = np.sign(disp)
        norms = np.abs(grad).sum(axis=2, keepdims=True)
        dirs = grad / np.maximum(norms, 1e-9)
        out = []
        for t in range(w.shape[0]):
            ww = np.where(valid, w[t], 0.0)
            out.append(np.sum(dirs * ww[..., None], axis=1))
        return np.stack(out)

    @staticmethod
    def _case(n, m, d, k, nw=1, seed=0):
        rng = np.random.default_rng(seed)
        return (
            np.ascontiguousarray(rng.random((n, d))),
            np.ascontiguousarray(rng.random((m, d))),
            rng.integers(-1, n, (m, k)).astype(np.int64),
            rng.standard_normal((nw, m, k)),
        )

    def test_it_agrees_with_the_reference_expression(self):
        for n, m, d, k, nw in (
            (400, 200, 100, 8, 1),
            (400, 200, 100, 64, 2),
            (120, 60, 10, 5, 1),
            (200, 100, 3, 4, 2),
            (300, 64, 200, 32, 2),
        ):
            with self.subTest(n=n, m=m, d=d, k=k, nw=nw):
                pts, q, ids, w = self._case(n, m, d, k, nw)
                np.testing.assert_allclose(
                    self._ref(pts, q, ids, w), rust.weighted_directions(pts, q, ids, w), atol=1e-12
                )

    def test_the_wrap_rounds_half_to_even(self):
        """NumPy rounds half to even; Rust's `f64::round` rounds half away
        from zero. An exact half is where they part, and every axis of every
        pair goes through that rounding."""
        pts = np.array([[0.25], [0.75]])
        q = np.array([[0.75]])
        ids = np.array([[0, 1]], dtype=np.int64)
        w = np.ones((1, 1, 2))
        np.testing.assert_array_equal(self._ref(pts, q, ids, w), rust.weighted_directions(pts, q, ids, w))

    def test_a_shared_axis_contributes_no_direction(self):
        """`np.sign(0) == 0`, where Rust's `signum` is +/-1 -- which would
        invent a unit step on every axis the two points agree on."""
        pts = np.array([[0.3, 0.9]])
        q = np.array([[0.3, 0.1]])
        ids = np.array([[0]], dtype=np.int64)
        out = rust.weighted_directions(pts, q, ids, np.ones((1, 1, 1)))
        self.assertEqual(float(out[0, 0, 0]), 0.0)

    def test_coincident_points_give_a_zero_vector(self):
        pts = np.array([[0.3, 0.4]])
        out = rust.weighted_directions(pts, pts.copy(), np.array([[0]], dtype=np.int64), np.full((1, 1, 1), 2.0))
        np.testing.assert_array_equal(out, 0.0)

    def test_missing_neighbours_contribute_nothing(self):
        pts, q, _, _ = self._case(20, 5, 4, 3)
        ids = np.full((5, 3), -1, dtype=np.int64)
        out = rust.weighted_directions(pts, q, ids, np.ones((1, 5, 3)))
        np.testing.assert_array_equal(out, 0.0)

    def test_weightings_share_one_geometry(self):
        """Two weightings in one call must equal two separate calls -- that
        equivalence is the whole reason the shared form is safe to use."""
        pts, q, ids, w = self._case(300, 100, 40, 16, 2)
        both = rust.weighted_directions(pts, q, ids, w)
        for t in range(2):
            single = rust.weighted_directions(pts, q, ids, w[t : t + 1])
            np.testing.assert_array_equal(both[t], single[0])

    def test_malformed_shapes_are_refused(self):
        pts, q, ids, w = self._case(40, 10, 6, 4)
        with self.assertRaises(ValueError):
            rust.weighted_directions(pts, np.zeros((10, 7)), ids, w)
        with self.assertRaises(ValueError):
            rust.weighted_directions(pts, q, ids, np.ones((1, 10, 5)))
        with self.assertRaises(ValueError):
            rust.weighted_directions(pts, np.zeros((9, 6)), ids, w)


class TestQueryResultType(unittest.TestCase):
    """Both queries answer with the same named pair, and it is still a
    tuple -- the name is added, nothing is taken away."""

    D = 8

    def make(self, backend="auto"):
        rng = np.random.default_rng(4)
        return ToroidalNN(seed=1, backend=backend).fit(rng.random((300, self.D)), rng.random((40, self.D)))

    def test_knn_and_padded_radius_agree_on_the_type(self):
        nn = self.make()
        for res in (nn.query(k=4), nn.query_radius(0.6, pad=True)):
            with self.subTest(res=type(res)):
                self.assertIsInstance(res, Neighbours)
                self.assertIsInstance(res, tuple)
                self.assertEqual(res.ids.dtype, np.int64)
                self.assertEqual(res.distances.dtype, np.float64)
                self.assertEqual(res.ids.shape, res.distances.shape)

    def test_the_names_are_the_positions(self):
        """Nothing reordered: a caller unpacking the old pair is unaffected."""
        res = self.make().query(k=4)
        ids, dists = res
        self.assertIs(ids, res.ids)
        self.assertIs(dists, res.distances)
        self.assertIs(res[0], res.ids)
        self.assertIs(res[1], res.distances)

    def test_the_ragged_form_is_one_per_query(self):
        rows = self.make().query_radius(0.6)
        self.assertEqual(len(rows), 40)
        for row in rows:
            self.assertIsInstance(row, Neighbours)
            self.assertEqual(row.ids.shape, row.distances.shape)

    def test_every_backend_answers_with_it(self):
        for backend in available_backends():
            with self.subTest(backend=backend):
                nn = self.make(backend)
                self.assertIsInstance(nn.query(k=4), Neighbours)
                self.assertIsInstance(nn.query_radius(0.6, pad=True), Neighbours)


class TestInstrumentationIsBuildTime(unittest.TestCase):
    """Release builds carry no query instrumentation at all: no counters in
    the hot path and no method exposing them. Only debug builds and builds
    with the `stats` feature have `stats()`."""

    def test_stats_exist_exactly_when_the_build_has_them(self):
        if rust.RustLshIndex is None:
            self.skipTest("compiled backend not available")
        for name in ("stats", "reset_stats"):
            self.assertEqual(hasattr(rust.RustLshIndex, name), rust.STATS, name)


class TestNativeStats(unittest.TestCase):
    """`RustLshIndex.stats()` is the instrument every performance change is
    measured with, so its fields are pinned here: one definition per field
    on every path. The batched join used to write neither `gather_ns` nor
    `cands`, which is how two fields read zero on the production path for as
    long as nothing tested them."""

    KEYS = frozenset(
        {"gather_ns", "refine_ns", "relax_ns", "queries", "cands", "pairs", "loads", "relaxed", "block", "path"}
    )

    @classmethod
    def setUpClass(cls):
        if rust.RustLshIndex is None:
            raise unittest.SkipTest("compiled backend not available")
        if not rust.STATS:
            raise unittest.SkipTest("release build: no instrumentation (build with --features stats)")

    def _index(self, block=0, d=8, n=4096, seed=0):
        rng = np.random.default_rng(seed)
        pts = rng.random((n, d))
        nn = ToroidalNN(seed=seed, brute_threshold=0).fit(pts[: n // 2], pts[n // 2 :], k=2 * d)
        native = rust.RustLshIndex(nn._B, nn._K, nn._L, nn.probes, nn._S, nn._U, block)
        native.build(np.ascontiguousarray(nn._arena), nn.n_static)
        return native, nn._arena, rng

    def test_the_key_set_is_exact(self):
        native, _, _ = self._index()
        self.assertEqual(set(native.stats()), self.KEYS)
        self.assertEqual(native.stats()["path"], "none")

    def test_every_path_fills_every_counter(self):
        cases = (
            ("per_query", 1, "knn"),
            ("batched", 0, "knn"),
            ("radius", 1, "radius"),
            ("radius_batched", 0, "radius"),
        )
        for path, block, op in cases:
            with self.subTest(path=path):
                native, arena, rng = self._index(block)
                q = np.ascontiguousarray(arena[rng.choice(len(arena), 256, replace=False)])
                if op == "knn":
                    native.query_knn(q, 10)
                elif block == 0:
                    # Enough queries per block for the radius join to engage.
                    native.query_radius(np.ascontiguousarray(arena), 0.5)
                else:
                    native.query_radius(q, 0.5)
                st = native.stats()
                self.assertEqual(st["path"], path)
                self.assertEqual(st["queries"], len(arena) if path == "radius_batched" else 256)
                for key in ("gather_ns", "refine_ns", "cands", "pairs", "loads"):
                    self.assertGreater(st[key], 0, key)
                self.assertLessEqual(st["cands"], st["pairs"])
                self.assertLessEqual(st["loads"], st["pairs"])
                if path in ("batched", "radius_batched"):
                    self.assertEqual(st["cands"], st["pairs"])
                else:
                    self.assertEqual(st["loads"], st["pairs"])

    def test_pairs_count_the_same_thing_on_both_knn_paths(self):
        a, arena, rng = self._index(block=1)
        b, _, _ = self._index(block=0)
        q = np.ascontiguousarray(arena[rng.choice(len(arena), 128, replace=False)])
        ra, rb = a.query_knn(q, 10), b.query_knn(q, 10)
        self.assertEqual(a.stats()["pairs"], b.stats()["pairs"])
        np.testing.assert_array_equal(ra[0], rb[0])

    def test_reset_zeroes_every_counter(self):
        native, arena, _ = self._index()
        native.query_knn(np.ascontiguousarray(arena[:100]), 5)
        native.reset_stats()
        st = native.stats()
        for key in self.KEYS - {"block", "path"}:
            self.assertEqual(st[key], 0, key)
        self.assertEqual(st["path"], "none")

    def test_small_batches_never_take_the_join(self):
        """`BATCH_MIN`: below 64 queries the join has nothing to amortise,
        whatever `block` says."""
        native, arena, _ = self._index(block=0)
        native.query_knn(np.ascontiguousarray(arena[:63]), 5)
        self.assertEqual(native.stats()["path"], "per_query")
        native.query_knn(np.ascontiguousarray(arena[:64]), 5)
        self.assertEqual(native.stats()["path"], "batched")
        # Radius gates on queries *per block*: 64 queries split over the
        # workers is too little reuse, the whole arena is enough on any
        # machine with up to 64 workers.
        native.query_radius(np.ascontiguousarray(arena[:64]), 0.5)
        self.assertEqual(native.stats()["path"], "radius")
        native.query_radius(np.ascontiguousarray(arena), 0.5)
        self.assertEqual(native.stats()["path"], "radius_batched")

    def test_block_one_selects_the_per_query_path(self):
        native, arena, _ = self._index(block=1)
        native.query_knn(np.ascontiguousarray(arena[:1000]), 5)
        self.assertEqual(native.stats()["path"], "per_query")
        self.assertEqual(native.stats()["block"], 1)

    def test_relaxation_is_booked_only_when_it_runs(self):
        """One table of 512 cells over 512 points and no probes: a bucket
        holds about one point, so k = 50 relaxes every query and k = 1
        almost none."""
        rng = np.random.default_rng(3)
        pts = np.ascontiguousarray(rng.random((512, 3)))
        S = np.array([[0, 1, 2]], dtype=np.int64)
        U = rng.random((1, 3))
        for block in (0, 1):
            with self.subTest(block=block):
                native = rust.RustLshIndex(8, 3, 1, 0, S, U, block)
                native.build(pts, 256)
                native.query_knn(np.ascontiguousarray(pts[:100]), 50)
                st = native.stats()
                self.assertEqual(st["relaxed"], 100)
                self.assertGreater(st["relax_ns"], 0)
                native.reset_stats()
                q = np.ascontiguousarray(pts[:100])
                native.query_knn(q, 1)
                st = native.stats()
                self.assertEqual(st["relaxed"] == 0, st["relax_ns"] == 0)


class TestRadiusJoin(unittest.TestCase):
    """The batched radius join returns exactly what the per-query scan does.

    Same ids, same order, same f64 distances, on every row -- including the
    points that sit on the cutoff, which is where two kernels that disagree
    in the last f32 bit would part. Runs on a release build: no `stats()`
    needed, `block=1` selects the per-query scan and `block=0` the join.
    """

    @classmethod
    def setUpClass(cls):
        if rust.RustLshIndex is None:
            raise unittest.SkipTest("compiled backend not available")

    def _pair(self, d, n, seed):
        rng = np.random.default_rng(seed)
        pts = rng.random((n, d))
        nn = ToroidalNN(seed=seed, brute_threshold=0).fit(pts[: n // 2], pts[n // 2 :], k=2 * d)
        arena = np.ascontiguousarray(nn._arena)
        out = []
        for block in (1, 0):
            native = rust.RustLshIndex(nn._B, nn._K, nn._L, nn.probes, nn._S, nn._U, block)
            native.build(arena, nn.n_static)
            out.append(native)
        return out, arena, rng

    def _assert_same(self, per_query, batched, q, r, ex=None):
        a = per_query.query_radius(q, r, ex)
        b = batched.query_radius(q, r, ex)
        for x, y, name in zip(a, b, ("indptr", "ids", "dists"), strict=True):
            np.testing.assert_array_equal(x, y, err_msg=name)
        return a

    def test_identical_to_the_per_query_scan(self):
        # d = 100 is above RADIUS_JOIN_MAX_D, so it pins that the gate keeps
        # the scan there, not the join's arithmetic.
        for d in (2, 8, 16, 24, 32, 100):
            (per_query, batched), arena, rng = self._pair(d, 4096, seed=d)
            # Every point as a query: the join engages at 64 queries per block,
            # which this reaches on machines with up to 64 workers.
            q = np.ascontiguousarray(arena[rng.permutation(len(arena))])
            ex = rng.integers(0, len(arena), len(arena)).astype(np.int64)
            for frac in (0.02, 0.06, 0.15):
                with self.subTest(d=d, frac=frac):
                    indptr, _, dists = self._assert_same(per_query, batched, q, frac * d, ex)
                    self.assertGreater(indptr[-1], 0)
                    # A cutoff exactly on returned distances: the boundary case.
                    for r in np.unique(dists)[:: max(1, len(dists) // 5)][:5]:
                        self._assert_same(per_query, batched, q, float(r))

    def test_self_join_excludes_the_query_point(self):
        (per_query, batched), arena, _ = self._pair(8, 2048, seed=1)
        q = np.ascontiguousarray(arena)
        ids = np.arange(len(arena), dtype=np.int64)
        indptr, got, _ = self._assert_same(per_query, batched, q, 0.6, ids)
        for i in range(len(arena)):
            self.assertNotIn(i, got[indptr[i] : indptr[i + 1]])


class TestRecallParameter(unittest.TestCase):
    """Construction-time `recall` knob ('auto' or float in (0, 1))."""

    def test_auto_is_bit_identical_to_default(self):
        rng = np.random.default_rng(7)
        pts = rng.random((1500, 12))
        q = rng.random((64, 12))
        a = ToroidalNN(seed=3, brute_threshold=0).fit(pts[:1000], pts[1000:], k=10)
        b = ToroidalNN(seed=3, brute_threshold=0, recall="auto").fit(pts[:1000], pts[1000:], k=10)
        self.assertEqual((a._B, a._K, a._L), (b._B, b._K, b._L))
        ia, da = a.query(k=10, queries=q)
        ib, db = b.query(k=10, queries=q)
        np.testing.assert_array_equal(ia, ib)
        np.testing.assert_array_equal(da, db)

    def test_recall_scales_tables_and_empirical_recall(self):
        rng = np.random.default_rng(11)
        pts = rng.random((2000, 16))
        q = rng.random((100, 16))
        k = 10
        exact_idx, _ = exact_knn(pts, q, k)
        prev_l = 0
        recalls = []
        for target in (0.30, 0.50, 0.75, 0.90, 0.98):
            nn = ToroidalNN(seed=5, brute_threshold=0, recall=target).fit(pts, k=k)
            self.assertGreaterEqual(nn.n_tables, prev_l)
            prev_l = nn.n_tables
            got_idx, _ = nn.query(k=k, queries=q)
            hits = sum(len(np.intersect1d(got_idx[i], exact_idx[i])) for i in range(len(q)))
            recalls.append(hits / exact_idx.size)
        self.assertLess(
            ToroidalNN(seed=5, brute_threshold=0, recall=0.50).fit(pts, k=k).n_tables,
            ToroidalNN(seed=5, brute_threshold=0, recall="auto").fit(pts, k=k).n_tables,
        )
        self.assertLess(
            ToroidalNN(seed=5, brute_threshold=0, recall=0.90, probes=2, dims_per_table=6).fit(pts, k=k).n_tables,
            ToroidalNN(seed=5, brute_threshold=0, recall=0.90, probes=0, dims_per_table=6).fit(pts, k=k).n_tables,
        )
        self.assertGreater(recalls[-1], recalls[0])

    def test_invalid_recall_raises(self):
        for bad in (0.0, 1.0, -0.2, 1.5, float("nan"), float("inf"), True, False, "fast"):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                ToroidalNN(recall=bad)  # type: ignore[arg-type]


class TestTuneContract(unittest.TestCase):
    """What `_tune` promises whatever the model behind it."""

    def test_native_and_python_tune_alike(self):
        if rust.brute_knn is None:
            self.skipTest("no compiled kernel")
        for d, n, p in ((4, 2048, 1.0), (8, 4096, 1.0), (16, 4096, 0.5), (32, 2048, 0.25), (64, 9000, 1.0)):
            for seed in range(4):
                pts = np.random.default_rng(seed).random((n, d))
                with self.subTest(d=d, n=n, p=p, seed=seed):
                    a = ToroidalNN(seed=seed, brute_threshold=0, p=p, backend="python").fit(pts, k=2 * d)
                    b = ToroidalNN(seed=seed, brute_threshold=0, p=p, backend="rust").fit(pts, k=2 * d)
                    self.assertEqual((a._B, a._K, a._L), (b._B, b._K, b._L))

    def test_more_probes_never_need_more_tables(self):
        pts = np.random.default_rng(2).random((4096, 16))
        tables = [
            ToroidalNN(seed=1, brute_threshold=0, probes=m, recall=0.8, dims_per_table=7).fit(pts, k=16)._L
            for m in (0, 1, 2, 4, 8)
        ]
        self.assertEqual(tables, sorted(tables, reverse=True))

    def test_extreme_hints_tune_to_a_usable_index(self):
        rng = np.random.default_rng(3)
        pts = rng.random((512, 8))
        cases = [{"radius": r} for r in (0.0, 1e-300, 1e3, 1e308, float("inf"))]
        cases += [{"k": 1}, {"k": 511}, {"k": 4096}]
        for p in (1.0, 0.5, 0.25):
            for hint in cases:
                with self.subTest(p=p, hint=hint):
                    nn = ToroidalNN(seed=4, brute_threshold=0, p=p, backend="python").fit(pts, **hint)
                    self.assertTrue(2 <= nn._B <= 8)
                    self.assertGreaterEqual(nn._K, 1)
                    self.assertGreaterEqual(nn._L, 1)
                    self.assertEqual(nn.query(k=4, queries=pts[:8])[0].shape, (8, 4))

    def test_degenerate_data_tunes_to_a_usable_index(self):
        rng = np.random.default_rng(5)
        sets = {
            "duplicates": np.tile(rng.random((1, 8)), (512, 1)),
            "two sites": rng.random((2, 8))[rng.integers(0, 2, 512)],
            "one dimension": rng.random((512, 1)),
            "more hash digits than dimensions": rng.random((8192, 2)),
        }
        for p in (1.0, 0.5):
            for name, pts in sets.items():
                with self.subTest(name=name, p=p):
                    nn = ToroidalNN(seed=6, brute_threshold=0, p=p, backend="python").fit(pts, k=4)
                    self.assertGreaterEqual(nn._L, 1)
                    self.assertEqual(nn.query(k=4, queries=pts[:8])[0].shape, (8, 4))

    def measured_hit_rate(self, Q, X, B, K, m, tables, rng):
        """Per-pair hit rate of the normative hash and probe rule (`lsh.py`)."""
        d, rows = Q.shape[1], np.arange(Q.shape[0])
        hits = np.zeros(Q.shape[0])
        for _ in range(tables):
            S, U = rng.integers(0, d, K), rng.random(K)
            fq, fx = np.mod(Q[:, S] + U, 1.0) * B, np.mod(X[:, S] + U, 1.0) * B
            cq, cx = np.minimum(fq.astype(np.int64), B - 1), np.minimum(fx.astype(np.int64), B - 1)
            differ = cq != cx
            hit = ~differ.any(axis=1)
            if m:
                f = fq - cq
                walls = np.argpartition(np.minimum(f, 1.0 - f), m - 1, axis=1)[:, :m]
                j = differ.argmax(axis=1)
                across = (cq + np.where(f >= 0.5, 1, -1)) % B
                hit |= (differ.sum(axis=1) == 1) & (walls == j[:, None]).any(axis=1) & (across[rows, j] == cx[rows, j])
            hits += hit
        return hits / tables

    def test_predicted_hit_rate_matches_the_real_hash(self):
        rng = np.random.default_rng(8)
        for d, n, p, B, K, m in (
            (8, 4096, 1.0, 3, 4, 4),
            (8, 4096, 0.25, 2, 7, 1),
            (16, 4096, 0.5, 2, 7, 4),
            (32, 4096, 1.0, 2, 6, 0),
            (32, 4096, 0.25, 2, 6, 4),
            (2, 4096, 1.0, 8, 4, 4),  # more hash digits than dimensions
        ):
            with self.subTest(d=d, p=p, B=B, K=K, m=m):
                pts = rng.random((n, d))
                Q = pts[:256]
                ids, _ = exact_knn(pts, Q, 2 * d + 1, p=p)
                X = pts[ids[:, 2 * d]]
                gaps = _torus_gaps(Q, X)
                self.assertAlmostEqual(
                    _wall_hit_rate(gaps, B, K, m).mean() / self.measured_hit_rate(Q, X, B, K, m, 2000, rng).mean(),
                    1.0,
                    delta=0.08,
                )

    def test_a_requested_recall_is_met_where_the_cap_allows(self):
        rng = np.random.default_rng(9)
        d, k = 16, 16
        pts = rng.random((4096, d))
        ids = rng.choice(4096, 256, replace=False)
        want, _ = exact_knn(pts, pts[ids], k, exclude_ids=ids)
        for goal in (0.5, 0.7, 0.85):
            with self.subTest(goal=goal):
                nn = ToroidalNN(seed=2, brute_threshold=0, recall=goal, backend="python").fit(pts, k=k)
                got, _ = nn.query(k=k, queries=pts[ids], exclude_ids=ids)
                real = sum(len(set(a) & set(b)) for a, b in zip(got.tolist(), want.tolist(), strict=True)) / want.size
                self.assertLess(nn._L, 24)
                self.assertGreaterEqual(nn.expected_recall, goal)
                self.assertGreaterEqual(real, goal)
                self.assertLess(real - goal, 0.25)  # met, not over-built

    def test_an_index_larger_than_the_tuning_sample_is_not_over_built(self):
        pts = np.random.default_rng(12).random((32768, 16))
        k = 16
        nn = ToroidalNN(seed=3, brute_threshold=0, recall=0.7, backend="python").fit(pts, k=k)
        Q = pts[:256]
        ids, _ = exact_knn(pts, Q, k + 1)
        true_gaps = _torus_gaps(Q, pts[ids[:, k]])
        L, _ = nn._reach(_wall_profile(true_gaps, nn._B), nn._K, nn.probes, 0.7)
        self.assertLessEqual(abs(nn._L - L), 1)

    def test_expected_recall_reports_a_blocked_target(self):
        pts = np.random.default_rng(10).random((8192, 64))
        nn = ToroidalNN(seed=1, brute_threshold=0, recall=0.98, probes=0, dims_per_table=8, backend="python")
        nn.fit(pts, k=32)
        self.assertEqual(nn._L, 24)
        self.assertLess(nn.expected_recall, 0.98)

    def test_past_the_cap_a_shallower_hash_keeps_the_target(self):
        pts = np.random.default_rng(13).random((32768, 32))
        pinned = ToroidalNN(seed=4, brute_threshold=0, probes=4, dims_per_table=9, backend="python").fit(pts, k=32)
        nn = ToroidalNN(seed=4, brute_threshold=0, backend="python").fit(pts, k=32)
        self.assertEqual(pinned._L, 24)
        self.assertLess(pinned.expected_recall, 0.96)
        self.assertLess(nn._K, pinned._K)
        self.assertLessEqual(nn._L, 24)
        self.assertGreaterEqual(nn.expected_recall, 0.96)

    def test_probes_the_target_does_not_need_are_dropped(self):
        rng = np.random.default_rng(14)
        pts = np.mod(rng.random((16, 32))[rng.integers(0, 16, 8192)] + rng.normal(0.0, 0.02, (8192, 32)), 1.0)
        kept = ToroidalNN(seed=4, brute_threshold=0, probes=4, backend="python").fit(pts, k=32)
        nn = ToroidalNN(seed=4, brute_threshold=0, backend="python").fit(pts, k=32)
        self.assertLess(nn.probes, 4)
        self.assertEqual(nn._L, kept._L)
        self.assertGreaterEqual(nn.expected_recall, kept.expected_recall - 0.01)

    def test_explicit_settings_are_never_tuned_away(self):
        pts = np.random.default_rng(15).random((8192, 32))
        self.assertEqual(ToroidalNN(seed=1, brute_threshold=0, probes=2).fit(pts, k=8).probes, 2)
        self.assertEqual(ToroidalNN(seed=1, brute_threshold=0, dims_per_table=9).fit(pts, k=8)._K, 9)
        self.assertEqual(ToroidalNN(seed=1, brute_threshold=0, num_tables=3).fit(pts, k=8).probes, 4)
        for bad in (-1, "fast", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ToroidalNN(probes=bad)  # type: ignore[arg-type]

    def test_expected_recall_is_none_without_a_tuned_table_count(self):
        pts = np.random.default_rng(11).random((512, 8))
        self.assertIsNone(ToroidalNN(seed=1, brute_threshold=10**6).fit(pts, k=8).expected_recall)
        self.assertIsNone(ToroidalNN(seed=1, brute_threshold=0, num_tables=3).fit(pts, k=8).expected_recall)

    def test_default_recall_does_not_fall(self):
        """Floors are the recall the defaults gave before the wall model, less 0.02."""
        floors = {
            ("knn", 8, 1.0): 0.93,
            ("knn", 8, 0.5): 0.94,
            ("knn", 16, 1.0): 0.92,
            ("knn", 16, 0.5): 0.97,
            ("knn", 32, 1.0): 0.94,
            ("knn", 32, 0.5): 0.97,
            ("radius", 8, 1.0): 0.94,
            ("radius", 8, 0.5): 0.95,
            ("radius", 16, 1.0): 0.91,
            ("radius", 16, 0.5): 0.96,
            ("radius", 32, 1.0): 0.95,
            ("radius", 32, 0.5): 0.98,
        }
        for (mode, d, p), floor in floors.items():
            rng = np.random.default_rng(d)
            pts = rng.random((4096, d))
            ids = rng.choice(4096, 300, replace=False)
            k = 2 * d
            want, dst = exact_knn(pts, pts[ids], k, exclude_ids=ids, p=p)
            radius = float(np.median(dst[:, -1]))
            with self.subTest(mode=mode, d=d, p=p):
                nn = ToroidalNN(seed=d, brute_threshold=0, p=p, backend="python")
                if mode == "knn":
                    got, _ = nn.fit(pts, k=k).query(k=k, queries=pts[ids], exclude_ids=ids)
                    hits = sum(len(set(a) & set(b)) for a, b in zip(got.tolist(), want.tolist(), strict=True))
                    total = want.size
                else:
                    nn.fit(pts, k=k, radius=radius)
                    got = nn.query_radius(radius, queries=pts[ids])
                    ref = nn.query_radius(radius, queries=pts[ids], exact=True)
                    hits = sum(len(set(a.ids.tolist()) & set(b.ids.tolist())) for a, b in zip(got, ref, strict=True))
                    total = sum(b.ids.size for b in ref)
                self.assertGreaterEqual(hits / total, floor)


class TestExactLp(unittest.TestCase):
    """Exact toroidal L^p search (`pairwise_lp`, `exact_knn`, `exact_radius`, `BruteIndex`)."""

    PS = (0.25, 0.5, 0.75, 1.0, 2.0)

    def naive(self, Q, P, p):
        diff = np.abs(Q[:, None, :] - P[None, :, :])
        delta = np.minimum(diff, 1.0 - diff)
        return (delta**p).sum(-1) ** (1.0 / p)

    def sample(self, m, n, d, seed=3):
        rng = np.random.default_rng(seed)
        return np.ascontiguousarray(rng.random((m, d))), np.ascontiguousarray(rng.random((n, d)))

    def test_p_one_is_bit_identical_to_l1(self):
        Q, P = self.sample(20, 60, 5)
        np.testing.assert_array_equal(pairwise_lp(Q, P, 1.0), pairwise_l1(Q, P))

    def test_pairwise_matches_naive_across_p_and_all_three_paths(self):
        for d in (2, 5, 8, 16, 129):
            Q, P = self.sample(12, 35, d)
            for p in self.PS:
                got = pairwise_lp(Q, P, p) ** (1.0 / p)
                np.testing.assert_allclose(got, self.naive(Q, P, p), rtol=1e-12, atol=1e-12)

    def test_knn_and_radius_match_naive(self):
        Q, P = self.sample(20, 100, 9)
        for p in self.PS:
            ref = self.naive(Q, P, p)
            order = np.argsort(ref, axis=1, kind="stable")[:, :7]
            idx, dst = exact_knn(P, Q, 7, p=p)
            np.testing.assert_array_equal(idx, order)
            np.testing.assert_allclose(dst, np.take_along_axis(ref, order, 1), rtol=1e-11)
            thr = float(np.median(ref[:, :5]))
            indptr, rids, rdst = exact_radius(P, Q, thr, p=p)
            for i in range(len(Q)):
                row = rids[indptr[i] : indptr[i + 1]]
                exp = np.flatnonzero(ref[i] <= thr)
                self.assertEqual(set(row.tolist()), set(exp.tolist()))
                np.testing.assert_allclose(rdst[indptr[i] : indptr[i + 1]], np.sort(ref[i, row]), rtol=1e-11)
            bi = BruteIndex(p=p)
            bi.build(P, len(P))
            b_idx, b_dst = bi.query_knn(Q, 7)
            np.testing.assert_array_equal(b_idx, idx)
            np.testing.assert_allclose(b_dst, dst, rtol=1e-11, atol=1e-12)


class TestLshLp(unittest.TestCase):
    """Toroidal L^p (0 < p <= 1) LSH index across Python and Rust backends."""

    def test_p_one_is_bit_identical_to_default(self):
        rng = np.random.default_rng(19)
        pts = rng.random((1200, 10))
        q = rng.random((50, 10))
        for b in BACKENDS:
            a = ToroidalNN(seed=2, brute_threshold=0, backend=b).fit(pts[:800], pts[800:], k=8)
            c = ToroidalNN(seed=2, brute_threshold=0, backend=b, p=1.0).fit(pts[:800], pts[800:], k=8)
            ia, da = a.query(k=8, queries=q)
            ic, dc = c.query(k=8, queries=q)
            np.testing.assert_array_equal(ia, ic)
            np.testing.assert_array_equal(da, dc)

    def test_seam_invariance_and_backend_equivalence_for_fractional_p(self):
        rng = np.random.default_rng(23)
        pts = rng.random((1500, 12))
        q = rng.random((80, 12))
        for p in (0.25, 0.5, 0.6, 0.75):
            ref_idx, ref_dst = exact_knn(pts, q, 8, p=p)
            # Brute mode via ToroidalNN
            b_nn = ToroidalNN(seed=4, brute_threshold=5000, p=p).fit(pts[:1000], pts[1000:], k=8)
            bi, bd = b_nn.query(k=8, queries=q)
            np.testing.assert_array_equal(bi, ref_idx)
            np.testing.assert_allclose(bd, ref_dst, rtol=1e-11, atol=1e-12)
            results = {}
            for b in BACKENDS:
                nn = ToroidalNN(seed=4, brute_threshold=0, backend=b, p=p).fit(pts[:1000], pts[1000:], k=8)
                idx, dst = nn.query(k=8, queries=q)
                results[b] = (nn, idx, dst)
                hits = sum(len(np.intersect1d(idx[i], ref_idx[i])) for i in range(len(q)))
                self.assertGreaterEqual(hits / ref_idx.size, 0.90, f"backend={b} p={p}")
                # Check exact L^p distances of returned ids (f32 refine amplified by 1/p)
                for i in range(10):
                    diff = np.abs(pts[idx[i]] - q[i])
                    d_true = ((np.minimum(diff, 1.0 - diff) ** p).sum(-1)) ** (1.0 / p)
                    np.testing.assert_allclose(dst[i], d_true, rtol=1e-4 / p, atol=1e-5)
                # Radius query + lifecycle (update + promote)
                r_cut = float(np.median(ref_dst[:, 3]))
                r_res = nn.query_radius(r_cut, queries=q[:16], pad=True)
                self.assertEqual(r_res.ids.shape[0], 16)
                r_exact = nn.query_radius(r_cut, queries=q[:16], exact=True, pad=True)
                self.assertGreaterEqual(r_exact.ids.shape[1], r_res.ids.shape[1])
                nn.update((pts[1000:] + 0.01) % 1.0)
                nn.promote(np.random.default_rng(7).random((100, 12)))
                idx2, _ = nn.query(k=5, queries=q[:10])
                self.assertEqual(idx2.shape, (10, 5))
                r_tuned = ToroidalNN(seed=4, brute_threshold=0, backend=b, p=p).fit(
                    pts[:1000], pts[1000:], radius=r_cut
                )
                self.assertGreater(r_tuned.n_tables, 0)
            if "python" in results and "rust" in results:
                py_nn, py_i, py_d = results["python"]
                rs_nn, rs_i, rs_d = results["rust"]
                for k_name, v in py_nn._impl.tables().items():
                    np.testing.assert_array_equal(v, rs_nn._impl.tables()[k_name], err_msg=k_name)
                np.testing.assert_allclose(py_d, rs_d, rtol=1e-5, atol=1e-5)
                agree = (py_i == rs_i).mean()
                self.assertGreaterEqual(agree, 0.99)

    def test_invalid_p_raises(self):
        for bad in (0.0, -0.5, 1.5, 2.0, float("nan"), float("inf"), True):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                ToroidalNN(p=bad)  # type: ignore[arg-type]


class TestCrossoverThresholds(unittest.TestCase):
    """Mode-specific crossover thresholds, crossover-band routing, and `tune_thresholds` (S3)."""

    def test_defaults_and_override_precedence(self):
        table = {  # (d, p): (k-NN threshold, radius threshold)
            (2, 1.0): (512, 1024),
            (8, 1.0): (2048, 2048),
            (32, 1.0): (2048, 512),
            (64, 1.0): (8192, 512),
            (100, 1.0): (65536, 65536),
            (16, 0.5): (512, 512),
            (32, 0.5): (1024, 512),
            (100, 0.25): (8192, 512),
        }
        pts = np.random.default_rng(1)
        for (d, p), (knn, radius) in table.items():
            with self.subTest(d=d, p=p):
                nn = ToroidalNN(p=p, backend="python").fit(pts.random((8, d)), k=2)
                self.assertEqual((nn._threshold("knn"), nn._threshold("radius")), (12288, 12288))
                if "rust" in available_backends():
                    nn = ToroidalNN(p=p, backend="rust").fit(pts.random((8, d)), k=2)
                    self.assertEqual((nn._threshold("knn"), nn._threshold("radius")), (knn, radius))
                    self.assertEqual(nn._threshold(), knn)
        g = ToroidalNN(brute_threshold=500)
        self.assertEqual((g._threshold("knn"), g._threshold("radius"), g._threshold()), (500, 500, 500))
        m = ToroidalNN(brute_threshold=500, brute_threshold_knn=800, brute_threshold_radius=200)
        self.assertEqual((m._threshold("knn"), m._threshold("radius"), m._threshold()), (800, 200, 500))

    def test_invalid_thresholds_raise(self):
        for kw in ("brute_threshold", "brute_threshold_knn", "brute_threshold_radius"):
            for bad in (-1, -100, True, False, 1.5, "auto"):
                with self.subTest(kw=kw, bad=bad), self.assertRaises((ValueError, TypeError)):
                    ToroidalNN(**{kw: bad})  # type: ignore[arg-type]

    def test_crossover_band_routing_and_lifecycle(self):
        rng = np.random.default_rng(31)
        pts = rng.random((400, 8))
        q = rng.random((24, 8))
        for b in BACKENDS:
            for p in (1.0, 0.5):
                # Band 1: thr_radius=100 < n=250 <= thr_knn=500
                nn = ToroidalNN(
                    seed=7,
                    backend=b,
                    p=p,
                    brute_threshold_knn=500,
                    brute_threshold_radius=100,
                ).fit(pts[:150], pts[150:250], k=6)
                self.assertFalse(nn.is_approximate)
                self.assertFalse(nn.is_approximate_knn)
                self.assertTrue(nn.is_approximate_radius)
                # query() stays exact before LSH is built
                ref_i, ref_d = knn_scan(pts[:250], q, 6, p=p)
                got_i, got_d = nn.query(k=6, queries=q)
                np.testing.assert_array_equal(got_i, ref_i)
                np.testing.assert_allclose(got_d, ref_d, rtol=1e-12)
                self.assertFalse(nn.is_approximate)
                # query_radius() builds LSH on demand and routes through LSH
                r_cut = float(np.median(ref_d[:, 3]))
                r_approx = nn.query_radius(r_cut, queries=q, pad=True)
                self.assertTrue(nn.is_approximate)
                self.assertEqual(r_approx.ids.shape[0], len(q))
                # query() still routes through exact knn_scan even after LSH is built
                got_i2, got_d2 = nn.query(k=6, queries=q)
                np.testing.assert_array_equal(got_i2, ref_i)
                np.testing.assert_allclose(got_d2, ref_d, rtol=1e-12)
                # update() keeps arena and LSH in sync; promote() crosses thr_knn
                moved = (pts[150:250] + 0.02) % 1.0
                nn.update(moved)
                ref_u_i, _ = knn_scan(np.vstack([pts[:150], moved]), q, 6, p=p)
                got_u_i, _ = nn.query(k=6, queries=q)
                np.testing.assert_array_equal(got_u_i, ref_u_i)
                nn.promote(pts[250:400])
                nn.promote(rng.random((200, 8)))
                self.assertTrue(nn.is_approximate_knn)
                self.assertEqual(nn.query(k=6, queries=q).ids.shape, (len(q), 6))

                # Band 2: thr_knn=100 < n=250 <= thr_radius=500 (fit with radius hint)
                nn2 = ToroidalNN(
                    seed=7,
                    backend=b,
                    p=p,
                    brute_threshold_knn=100,
                    brute_threshold_radius=500,
                ).fit(pts[:150], pts[150:250], k=6, radius=r_cut)
                self.assertFalse(nn2.is_approximate)
                self.assertTrue(nn2.is_approximate_knn)
                self.assertFalse(nn2.is_approximate_radius)
                # query() lazily builds LSH; query_radius() still uses exact radius_scan
                nn2.query(k=6, queries=q)
                self.assertTrue(nn2.is_approximate)
                r_got = nn2.query_radius(r_cut, queries=q, pad=True)
                r_ex = nn2.query_radius(r_cut, queries=q, exact=True, pad=True)
                np.testing.assert_array_equal(r_got.ids, r_ex.ids)
                np.testing.assert_allclose(r_got.distances, r_ex.distances, rtol=1e-12)
                # promote() across radius threshold when fitted with radius hint
                nn3 = ToroidalNN(
                    seed=7,
                    backend=b,
                    p=p,
                    brute_threshold_knn=900,
                    brute_threshold_radius=300,
                ).fit(pts[:150], pts[150:250], radius=r_cut)
                self.assertFalse(nn3.is_approximate)
                nn3.promote(pts[250:400])
                self.assertTrue(nn3.is_approximate)

    def test_tune_thresholds_self_and_external(self):
        rng = np.random.default_rng(41)
        pts = rng.random((300, 8))
        q = rng.random((32, 8))
        for p in (1.0, 0.5):
            cfg_self = tune_thresholds(pts, k=6, p=p, repeats=1, seed=3)
            self.assertEqual(set(cfg_self.keys()), {"brute_threshold_knn", "brute_threshold_radius"})
            for v in cfg_self.values():
                self.assertIn(v, (0, len(pts)))  # a single size was measured
            cfg_ext = tune_thresholds(pts, q, k=6, radius=0.8, p=p, repeats=1, seed=3)
            nn = ToroidalNN(seed=3, p=p, **cfg_ext).fit(pts, k=6)
            self.assertEqual(nn.query(k=6, queries=q).ids.shape, (len(q), 6))
        for bad_kw in (
            {"points": rng.random((1, 4))},
            {"points": pts, "repeats": 0},
            {"points": pts, "repeats": True},
            {"points": pts, "epochs": 0},
            {"points": pts, "epochs": True},
            {"points": pts, "k": 0},
            {"points": pts, "queries": rng.random((10, 5))},
        ):
            with self.subTest(bad_kw=list(bad_kw.keys())), self.assertRaises((ValueError, TypeError)):
                tune_thresholds(**bad_kw)  # type: ignore[arg-type]

    def test_tune_thresholds_finds_a_crossover_inside_the_data(self):
        """A size is judged on real timings: pin the search, not the clock."""
        pts = np.random.default_rng(42).random((1024, 8))
        # Stub a clock that makes LSH lose below 600 points.
        real_query = ToroidalNN.query

        def query(nn, *a, **kw):
            out = real_query(nn, *a, **kw)
            clock.now += 1.0 if nn._use_lsh and nn.n_points > 600 else 3.0 if nn._use_lsh else 2.0
            return out

        clock = unittest.mock.Mock(now=0.0)
        with (
            unittest.mock.patch.object(ToroidalNN, "query", query),
            unittest.mock.patch("torann.wrapper.perf_counter", lambda: clock.now),
        ):
            cfg = tune_thresholds(pts, k=4, repeats=1, seed=3, min_recall=0.0)
        self.assertEqual(cfg["brute_threshold_knn"], 512)

    def test_tune_thresholds_honours_the_recall_floor(self):
        pts = np.random.default_rng(43).random((2048, 16))
        cfg = tune_thresholds(pts, k=8, recall=0.2, repeats=1, seed=3, min_recall=1.0)
        self.assertEqual(cfg["brute_threshold_knn"], 2048)


class TestLpBackendAgreement(unittest.TestCase):
    """At p < 1 the backends may split ties within float32 rounding, and nothing else."""

    def setUp(self):
        if rust.brute_knn is None:
            self.skipTest("no compiled kernel")

    def test_results_differ_only_within_rounding(self):
        rng = np.random.default_rng(5)
        sets = {
            "uniform": rng.random((2048, 16)),
            "dyadic": rng.integers(0, 16, (2048, 16)) / 16.0,
            "tiny gaps": 0.5 + rng.normal(0, 1e-7, (2048, 16)),
        }
        for name, pts in sets.items():
            for p in (0.75, 0.5, 0.3, 0.25):
                with self.subTest(name=name, p=p):
                    a = ToroidalNN(seed=9, brute_threshold=0, backend="python", p=p).fit(pts, k=8)
                    b = ToroidalNN(seed=9, brute_threshold=0, backend="rust", p=p).fit(pts, k=8)
                    ai, ad = a.query(k=8, queries=pts[:128])
                    bi, bd = b.query(k=8, queries=pts[:128])
                    np.testing.assert_allclose(ad, bd, rtol=1e-5)
                    r = float(np.median(ad[:, -1])) * 1.001
                    for x, y in zip(
                        a.query_radius(r, queries=pts[:128]), b.query_radius(r, queries=pts[:128]), strict=True
                    ):
                        np.testing.assert_array_equal(np.sort(x.ids), np.sort(y.ids))
                    self.assertLess(np.mean(ai != bi), 0.02)


if __name__ == "__main__":
    unittest.main()
