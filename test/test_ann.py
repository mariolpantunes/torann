import logging
import unittest

import numpy as np

from src.ann import ToroidalNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class TestBruteMode(unittest.TestCase):
    """Below brute_threshold every answer must be exact."""

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
        self.assertTrue((idx[:, :6] >= 0).all())   # 7 points minus self
        self.assertTrue((idx[:, 6:] == -1).all())
        self.assertTrue(np.isinf(dst[:, 6:]).all())


class TestLSHMode(unittest.TestCase):
    """Above brute_threshold: approximate filter, exact distances."""

    D = 8

    def setUp(self):
        rng = np.random.default_rng(7)
        self.static = rng.random((7000, self.D))
        self.cands = rng.random((1000, self.D))
        self.all_pts = np.vstack([self.static, self.cands])
        self.cand_ids = np.arange(7000, 8000)

    def test_is_approximate(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        self.assertTrue(nn.is_approximate)

    def test_recall_of_default_query(self):
        k = 2 * self.D
        nn = ToroidalNN(seed=0).fit(self.static, self.cands, k=k)
        idx, dst = nn.query()
        ridx, _ = brute_knn(self.all_pts, self.cands, k, self.cand_ids)
        r = recall(idx, ridx)
        logger.info("LSH recall@%d: %.3f", k, r)
        self.assertGreater(r, 0.85)
        self.assertTrue((idx >= 0).all())  # exact fallback fills all rows

    def test_distances_are_true_l1_and_sorted(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands, k=16)
        idx, dst = nn.query()
        d_true = np.take_along_axis(
            torus_l1(self.cands, self.all_pts), idx, axis=1)
        np.testing.assert_allclose(dst, d_true, atol=1e-5)  # float32 refine
        self.assertTrue((np.diff(dst, axis=1) >= -1e-6).all())

    def test_wrap_around(self):
        static = self.static.copy()
        static[0] = 0.999
        nn = ToroidalNN(seed=0).fit(static, np.full((1, self.D), 0.001))
        idx, dst = nn.query(k=1)
        self.assertEqual(idx[0, 0], 0)
        self.assertAlmostEqual(dst[0, 0], 0.002 * self.D, places=5)

    def test_determinism(self):
        a = ToroidalNN(seed=99).fit(self.static, self.cands).query(k=16)
        b = ToroidalNN(seed=99).fit(self.static, self.cands).query(k=16)
        np.testing.assert_array_equal(a[0], b[0])

    def test_tuning_respects_explicit_args(self):
        nn = ToroidalNN(seed=0, resolution=2, num_tables=5,
                        dims_per_table=7).fit(self.static, self.cands)
        self.assertEqual((nn._B, nn._L, nn._K), (2, 5, 7))

    def test_starved_index_fills_rows_via_relaxation(self):
        """One table, no probes, oversized keys: buckets are near-empty, so
        most queries must be completed by prefix relaxation, not brute force.
        """
        nn = ToroidalNN(seed=0, num_tables=1, probes=0,
                        dims_per_table=14).fit(self.static, self.cands, k=16)
        idx, dst = nn.query()
        self.assertTrue((idx >= 0).all())
        self.assertTrue(np.isfinite(dst).all())
        cand_ids = np.arange(7000, 8000)
        self.assertFalse((idx == cand_ids[:, None]).any())
        d_true = np.take_along_axis(
            torus_l1(self.cands, self.all_pts), idx, axis=1)
        np.testing.assert_allclose(dst, d_true, atol=1e-5)
        self.assertTrue((np.diff(dst, axis=1) >= -1e-6).all())


class TestLifecycle(unittest.TestCase):
    """The ESS loop: fit -> (query, update)* -> promote -> repeat."""

    D = 8

    def setUp(self):
        rng = np.random.default_rng(3)
        self.rng = rng
        self.static = rng.random((6000, self.D))
        self.cands = rng.random((800, self.D))

    def _reference(self, nn, k):
        all_pts = np.vstack([nn._arena[:nn.n_static], nn.candidates])
        cand_ids = np.arange(nn.n_static, nn.n_points)
        return brute_knn(all_pts, nn.candidates.copy(), k, cand_ids)

    def test_recall_holds_through_epochs(self):
        k = 2 * self.D
        nn = ToroidalNN(seed=0).fit(self.static, self.cands, k=k)
        for _ in range(5):
            step = self.rng.normal(0, 0.01, (800, self.D))
            nn.update(np.mod(nn.candidates + step, 1.0))
        idx, _ = nn.query()
        ridx, _ = self._reference(nn, k)
        r = recall(idx, ridx)
        logger.info("recall after 5 update epochs: %.3f", r)
        self.assertGreater(r, 0.85)

    def test_update_touches_only_changed_keys(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        keys_before = nn._keys_c.copy()
        step = self.rng.normal(0, 0.002, (800, self.D))
        nn.update(np.mod(nn.candidates + step, 1.0))
        changed = (nn._keys_c != keys_before).mean()
        logger.info("keys changed by tiny step: %.3f", changed)
        self.assertLess(changed, 0.30)  # most keys survive a small step

    def test_selective_update_equals_full_rebuild(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        for sigma in (0.001, 0.02, 0.3):  # few, some, most keys change
            step = self.rng.normal(0, sigma, (800, self.D))
            nn.update(np.mod(nn.candidates + step, 1.0))
            fresh = nn._tier_keys(nn._cand_sids)
            for t in range(nn._L):
                np.testing.assert_array_equal(
                    np.sort(fresh[t]), nn._skeys_c[t])
                np.testing.assert_array_equal(
                    fresh[t][nn._sids_c[t] - nn.n_static], nn._skeys_c[t])

    def test_promote_merge_equals_full_rebuild(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        new_batch = self.rng.random((500, self.D))
        nn.promote(new_batch)
        self.assertEqual((nn.n_static, nn.n_candidates), (6800, 500))
        for t in range(nn._L):  # merged arrays must be perfectly sorted
            self.assertTrue((np.diff(nn._skeys_s[t]) >= 0).all())
            keys = nn._tier_keys(np.arange(nn.n_static))[t]
            np.testing.assert_array_equal(np.sort(keys), nn._skeys_s[t])
            np.testing.assert_array_equal(keys[nn._sids_s[t]], nn._skeys_s[t])

    def test_promote_freezes_current_positions(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        nn.update(self.rng.random((800, self.D)))  # teleport before freezing
        nn.promote(self.rng.random((100, self.D)))
        keys = nn._tier_keys(np.arange(nn.n_static))
        for t in range(nn._L):  # frozen keys must reflect current positions
            np.testing.assert_array_equal(np.sort(keys[t]), nn._skeys_s[t])

    def test_full_loop_correctness(self):
        k = 2 * self.D
        nn = ToroidalNN(seed=0).fit(self.static, self.cands, k=k)
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
        nn = ToroidalNN(seed=0, brute_threshold=1500).fit(
            self.static[:1000], self.cands[:400])
        self.assertFalse(nn.is_approximate)
        nn.promote(self.rng.random((300, self.D)))  # 1700 > 1500
        self.assertTrue(nn.is_approximate)
        self.assertEqual((nn.n_static, nn.n_candidates), (1400, 300))
        idx, dst = nn.query(k=8)
        ridx, rdst = self._reference(nn, 8)
        self.assertGreater(recall(idx, ridx), 0.85)


class TestRangeQueries(unittest.TestCase):
    D = 6

    def setUp(self):
        rng = np.random.default_rng(5)
        self.static = rng.random((4500, self.D))
        self.cands = rng.random((600, self.D))
        self.all_pts = np.vstack([self.static, self.cands])
        self.radius = 0.3

    def _exact_sets(self):
        D = torus_l1(self.cands, self.all_pts)
        D[np.arange(600), np.arange(4500, 5100)] = np.inf
        return [np.flatnonzero(row <= self.radius) for row in D]

    def test_exact_flag(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        self.assertTrue(nn.is_approximate)
        res = nn.query_radius(self.radius, exact=True)
        for (ids, dst), ref in zip(res, self._exact_sets()):
            np.testing.assert_array_equal(np.sort(ids), ref)
            self.assertTrue((np.diff(dst) >= -1e-12).all())

    def test_post_filter_returns_only_true_hits(self):
        nn = ToroidalNN(seed=0).fit(self.static, self.cands)
        res = nn.query_radius(self.radius)
        refs = self._exact_sets()
        found = total = 0
        for (ids, _), ref in zip(res, refs):
            self.assertTrue(np.isin(ids, ref).all())  # no false positives
            found += len(ids)
            total += len(ref)
        r = found / max(1, total)
        logger.info("range recall: %.3f", r)
        self.assertGreater(r, 0.5)

    def test_brute_mode(self):
        nn = ToroidalNN(seed=0, brute_threshold=10_000).fit(self.static, self.cands)
        res = nn.query_radius(self.radius)
        for (ids, _), ref in zip(res, self._exact_sets()):
            np.testing.assert_array_equal(np.sort(ids), ref)


class TestValidation(unittest.TestCase):
    def test_query_before_fit(self):
        with self.assertRaises(RuntimeError):
            ToroidalNN().query(k=1)

    def test_bad_params(self):
        for kwargs in ({"resolution": 1}, {"num_tables": 0}, {"probes": -1},
                       {"dims_per_table": 0}):
            with self.assertRaises(ValueError):
                ToroidalNN(**kwargs)

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


if __name__ == "__main__":
    unittest.main()
