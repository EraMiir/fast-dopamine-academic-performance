import unittest
import numpy as np

from src.benchmark import sample_train_test_indices


class TestSampleTrainTestIndices(unittest.TestCase):

    def test_train_and_test_do_not_overlap(self):
        cases = [
            (256, 64),
            (256, 128),
            (256, 192),
            (256, 256),
        ]

        for n_rows, n_samples in cases:
            with self.subTest(n_samples=n_samples):
                train_idx, test_idx = sample_train_test_indices(
                    n_rows=n_rows,
                    n_samples=n_samples,
                    test_size=0.2,
                    seed=42,
                )

                self.assertEqual(
                    set(train_idx.tolist())
                    .intersection(test_idx.tolist()),
                    set(),
                )

                self.assertEqual(
                    len(np.unique(np.concatenate(
                        [train_idx, test_idx]
                    ))),
                    n_samples,
                )

                self.assertGreater(len(train_idx), 0)
                self.assertGreater(len(test_idx), 0)

    def test_same_seed_gives_same_split(self):
        first = sample_train_test_indices(256, 128, 0.2, 42)
        second = sample_train_test_indices(256, 128, 0.2, 42)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_rejects_sample_larger_than_dataset(self):
        with self.assertRaises(ValueError):
            sample_train_test_indices(10, 11, 0.2, 42)

    def test_rejects_invalid_test_size(self):
        with self.assertRaises(ValueError):
            sample_train_test_indices(256, 64, 1.0, 42)


if __name__ == "__main__":
    unittest.main()
