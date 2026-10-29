
import unittest

import numpy as np

from src.process_data import bootstrap_ci


class TestBootstrapCI(unittest.TestCase):
    def test_returns_bounds_in_correct_order(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        low, high = bootstrap_ci(
            data,
            n_boot=500,
            ci=0.95,
            seed=42,
        )

        self.assertTrue(np.isfinite(low))
        self.assertTrue(np.isfinite(high))
        self.assertLessEqual(low, high)

    def test_is_reproducible_with_same_seed(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        first = bootstrap_ci(data, n_boot=500, seed=42)
        second = bootstrap_ci(data, n_boot=500, seed=42)

        self.assertEqual(first, second)

    def test_all_nan_returns_nan_bounds(self):
        data = np.array([np.nan, np.nan])

        low, high = bootstrap_ci(data)

        self.assertTrue(np.isnan(low))
        self.assertTrue(np.isnan(high))


if __name__ == "__main__":
    unittest.main()
