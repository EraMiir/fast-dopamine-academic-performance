"""Unit tests for benchmark helpers. Run: python -m unittest test_benchmark -v"""
import unittest
import numpy as np
from src.benchmark import load_config, sample_train_test_indices, build_models


class BenchmarkHelperTests(unittest.TestCase):
    def test_sample_is_deterministic_and_disjoint(self):
        train_a, test_a = sample_train_test_indices(100, 64, 0.2, 42)
        train_b, test_b = sample_train_test_indices(100, 64, 0.2, 42)
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
        self.assertEqual((len(train_a), len(test_a)), (51, 13))
        self.assertTrue(set(train_a).isdisjoint(set(test_a)))
        self.assertEqual(len(set(np.concatenate([train_a, test_a]))), 64)

    def test_invalid_sample_size_raises(self):
        with self.assertRaises(ValueError):
            sample_train_test_indices(10, 11, 0.2, 42)

    def test_invalid_test_fraction_raises(self):
        with self.assertRaises(ValueError):
            sample_train_test_indices(10, 5, 1.0, 42)

    def test_config_contains_three_models(self):
        models = build_models(load_config(), seed=42)
        self.assertEqual(set(models), {"Ridge", "Random Forest", "Gradient Boosting"})


if __name__ == "__main__":
    unittest.main()
