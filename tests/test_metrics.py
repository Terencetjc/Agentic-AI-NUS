"""Verify metric formulas using synthetic values with known outcomes."""

import unittest

from evaluation.metrics import (accuracy, accuracy_retention, complete_sum,
                                cost_per_successful_task, latency_summary,
                                token_reduction, total_api_cost, total_tokens)


class MetricsTests(unittest.TestCase):
    def test_accuracy_and_ungraded(self):
        self.assertEqual(accuracy([True, False, True, False]), 0.5)
        self.assertEqual(accuracy([False]), 0)
        self.assertIsNone(accuracy([True, None]))
        self.assertIsNone(accuracy([]))
        with self.assertRaises(ValueError):
            accuracy([1])

    def test_retention_and_zero_denominators(self):
        self.assertEqual(accuracy_retention(0.5, 1), 0.5)
        self.assertEqual(accuracy_retention(1, 0.5), 2)
        for method, baseline in ((0, 0), (None, 1), (1, None)):
            self.assertIsNone(accuracy_retention(method, baseline))

    def test_tokens_costs_and_missing_values(self):
        self.assertEqual(total_tokens(100, 20), 120)
        self.assertEqual(total_tokens(0, 0), 0)
        self.assertIsNone(total_tokens(None, 20))
        self.assertIsNone(total_tokens(100, None))
        self.assertAlmostEqual(total_api_cost([0.1, 0.2]), 0.3)
        self.assertIsNone(total_api_cost([0.1, None]))
        self.assertEqual(total_api_cost([]), 0)
        self.assertEqual(complete_sum([]), 0)
        self.assertIsNone(complete_sum([None]))

    def test_token_reduction_unclamped(self):
        self.assertEqual(token_reduction(100, 200), 0.5)
        self.assertEqual(token_reduction(300, 200), -0.5)
        for method, baseline in ((0, 0), (None, 200), (100, None)):
            self.assertIsNone(token_reduction(method, baseline))

    def test_cost_per_success(self):
        self.assertEqual(cost_per_successful_task(6, 2), 3)
        for cost, correct in ((6, 0), (None, 2), (6, None)):
            self.assertIsNone(cost_per_successful_task(cost, correct))

    def test_latency_summary(self):
        summary = latency_summary([0, 2, 4, 10])
        self.assertEqual((summary.count, summary.measured_count), (4, 4))
        self.assertEqual((summary.mean, summary.median, summary.minimum, summary.maximum),
                         (4, 3, 0, 10))

    def test_latency_does_not_silently_drop_missing_measurements(self):
        summary = latency_summary([1, None, 3])
        self.assertEqual((summary.count, summary.measured_count), (3, 2))
        for name in ("mean", "median", "minimum", "maximum"):
            self.assertIsNone(getattr(summary, name))
        empty = latency_summary([])
        self.assertEqual(empty.count, 0)
        self.assertIsNone(empty.mean)


if __name__ == "__main__":
    unittest.main()
