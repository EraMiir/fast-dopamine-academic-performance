
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.benchmark import ensure_dataset, build_models


class TestEnsureDataset(unittest.TestCase):
    def test_generates_dataset_with_expected_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generated.csv"

            df = ensure_dataset(path, seed=42)

            self.assertEqual(len(df), 256)
            self.assertEqual(
                set(df.columns),
                {
                    "short_video_hours_per_day",
                    "notification_checks_per_day",
                    "sleep_hours",
                    "deep_study_hours",
                    "attention_span_minutes",
                    "gpa",
                },
            )
            self.assertTrue(path.exists())

    def test_same_seed_produces_same_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path1 = Path(tmp) / "first.csv"
            path2 = Path(tmp) / "second.csv"

            df1 = ensure_dataset(path1, seed=42)
            df2 = ensure_dataset(path2, seed=42)

            pd.testing.assert_frame_equal(df1, df2)


class TestBuildModels(unittest.TestCase):
    def test_builds_three_models(self):
        models = build_models({}, seed=42)

        self.assertEqual(
            set(models.keys()),
            {"Ridge", "Random Forest", "Gradient Boosting"},
        )

        for model in models.values():
            self.assertTrue(hasattr(model, "fit"))
            self.assertTrue(hasattr(model, "predict"))


if __name__ == "__main__":
    unittest.main()
