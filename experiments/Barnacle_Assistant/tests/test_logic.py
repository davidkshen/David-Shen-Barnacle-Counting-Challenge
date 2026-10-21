"""Checks that protect scientific conclusions, rather than mirroring filters."""
import itertools
import unittest
import numpy as np
from barnacles.core import match_objects, metrics, ring_kernel
from barnacles.audit import corrected_count, tile_counts, tile_ids


class ScientificLogicTests(unittest.TestCase):
    def test_matching_maximizes_cardinality_not_greedy_distance(self):
        # A nearest-first greedy match would consume G0 with P0 and lose P1.
        gt = np.array([[0., 0., 4.], [4., 0., 4.]])
        pred = np.array([[1.5, 0.], [-2., 0.]])
        m = match_objects(pred, gt)
        self.assertEqual(len(m), 2)
        self.assertEqual(len(set(m[:, 0])), 2)
        self.assertEqual(len(set(m[:, 1])), 2)

    def test_duplicate_prediction_is_false_positive(self):
        gt = np.array([[10., 10., 5.]])
        m = metrics(np.array([[10., 10.], [10.5, 10.]]), gt)
        self.assertEqual((m['tp'], m['fp'], m['fn']), (1, 1, 0))

    def test_empty_predictions_are_all_misses(self):
        m = metrics(np.empty((0, 2)), np.array([[10., 10., 5.]]))
        self.assertEqual(m['fn'], 1)
        self.assertEqual(m['relative_error'], -1.)

    def test_difference_estimator_is_unbiased_over_all_samples(self):
        truth = np.array([2, 8, 1, 13]); model = np.array([1, 3, 7, 6])
        estimates = [corrected_count(model, ids, truth[list(ids)])['estimate']
                     for ids in itertools.combinations(range(4), 2)]
        self.assertAlmostEqual(np.mean(estimates), truth.sum())
        residual = truth-model
        exact_variance = 4**2*(1-2/4)*residual.var(ddof=1)/2
        self.assertAlmostEqual(np.var(estimates), exact_variance)

    def test_full_census_is_exact_even_with_bad_model(self):
        model = [200, 0, 100, 1]; truth = [1, 2, 3, 4]
        a = corrected_count(model, [0, 1, 2, 3], truth)
        self.assertEqual(a['estimate'], 10)
        self.assertEqual(a['interval95'], [10., 10.])

    def test_half_open_tile_boundaries_do_not_double_count(self):
        xy = np.array([[0., 0.], [127.99, 127.99], [128., 128.], [1023., 1023.]])
        np.testing.assert_array_equal(tile_ids(xy), [0, 0, 9, 63])
        self.assertEqual(tile_counts(xy).sum(), 4)

    def test_audit_rejects_duplicate_tiles(self):
        with self.assertRaises(ValueError):
            corrected_count(np.zeros(64), [1, 1], [2, 3])

    def test_ring_filter_has_zero_dc_response(self):
        for r in (3., 7.5, 17.):
            self.assertAlmostEqual(float(ring_kernel(r).sum()), 0., places=6)


if __name__ == '__main__':
    unittest.main()
