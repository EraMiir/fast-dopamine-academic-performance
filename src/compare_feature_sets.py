"""Compare academic-only and full-feature GPA prediction on identical splits."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/cleaned_survey.csv")
OUT = Path("data/processed/feature_set_comparison.csv")
TARGET = "Last Academic Result (GPA/CGPA)"
ACADEMIC_FEATURES = [
    "Age",
    "Study time (In Hours)",
    "Attendance rate (In Percentile)",
]
FULL_FEATURES = ACADEMIC_FEATURES + [
    "Time spent in social media (hours)",
    "Sleep Disturbance on Sleep Quality",
    "Mood Modification Scale",
    "Anxiety Scale",
    "Depression Scale",
    "Self-esteem Scale",
    "Social Media Distraction During Academic Activities",
]
N_SPLITS = 30
TEST_SIZE = 0.2
SEED = 42


def make_models(seed):
    return {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=seed
        ),
    }


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    df.columns = df.columns.astype(str).str.strip()
    all_features = list(dict.fromkeys(ACADEMIC_FEATURES + FULL_FEATURES))
    missing = [c for c in all_features + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    data = df[all_features + [TARGET]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Comparison data contains missing or non-finite values.")
    rows = []
    X_sets = {
        "Academic-only": ACADEMIC_FEATURES,
        "Full Model": FULL_FEATURES,
    }
    y = data[TARGET].to_numpy(dtype=float)
    for split_id in range(N_SPLITS):
        indices = np.arange(len(data))
        train_idx, test_idx = train_test_split(
            indices, test_size=TEST_SIZE, random_state=SEED + split_id
        )
        for set_name, features in X_sets.items():
            X = data[features].to_numpy(dtype=float)
            for model_name, model in make_models(SEED + split_id).items():
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                rows.append({
                    "split": split_id,
                    "feature_set": set_name,
                    "model": model_name,
                    "n_features": len(features),
                    "RMSE": float(np.sqrt(mean_squared_error(y[test_idx], pred))),
                    "MAE": float(mean_absolute_error(y[test_idx], pred)),
                    "R2": float(r2_score(y[test_idx], pred)),
                })
    result = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    summary = result.groupby(["model", "feature_set"]).agg(
        n_runs=("RMSE", "size"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_sd=("MAE", "std"),
        R2_mean=("R2", "mean"),
        R2_sd=("R2", "std"),
    ).reset_index()
    summary.to_csv(OUT.with_name("feature_set_comparison_summary.csv"), index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Saved run-level results to {OUT}")


if __name__ == "__main__":
    main()
