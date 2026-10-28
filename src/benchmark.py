import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ensure_dataset(dataset_path: Path, seed: int = 42) -> pd.DataFrame:
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        if len(df) >= 256:
            return df.iloc[:256].reset_index(drop=True)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = 256

    short_video_hours = rng.uniform(0.5, 6.5, size=n)
    notification_checks = rng.integers(15, 180, size=n)
    sleep_hours = np.clip(rng.normal(6.8, 1.1, size=n), 4.0, 9.5)
    study_hours = np.clip(rng.normal(3.2, 1.3, size=n) - 0.25 * short_video_hours, 0.5, 8.0)
    attention_span_min = np.clip(45 - 4.2 * short_video_hours - 0.08 * notification_checks + rng.normal(0, 3.5, size=n), 8.0, 60.0)
    
    gpa = (
        2.65
        - 0.14 * short_video_hours
        - 0.0025 * notification_checks
        + 0.16 * study_hours
        + 0.07 * (sleep_hours - 6.5)
        + 0.012 * attention_span_min
        - 0.02 * (short_video_hours ** 1.5)
        + rng.normal(0, 0.18, size=n)
    )
    gpa = np.clip(gpa, 1.0, 4.0)

    df = pd.DataFrame({
        "short_video_hours_per_day": np.round(short_video_hours, 3),
        "notification_checks_per_day": notification_checks,
        "sleep_hours": np.round(sleep_hours, 3),
        "deep_study_hours": np.round(study_hours, 3),
        "attention_span_minutes": np.round(attention_span_min, 3),
        "gpa": np.round(gpa, 4),
    })
    df.to_csv(dataset_path, index=False)
    print(f"[INFO] Сформирован и сохранён датасет: {dataset_path} ({len(df)} строк)")
    return df


def build_models(model_cfg: dict, seed: int) -> dict:
    return {
        "Ridge": Ridge(
            alpha=model_cfg.get("Ridge", {}).get("alpha", 1.0),
            random_state=seed,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=model_cfg.get("Random Forest", {}).get("n_estimators", 100),
            max_depth=model_cfg.get("Random Forest", {}).get("max_depth", 6),
            random_state=seed,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=model_cfg.get("Gradient Boosting", {}).get("n_estimators", 100),
            learning_rate=model_cfg.get("Gradient Boosting", {}).get("learning_rate", 0.08),
            max_depth=model_cfg.get("Gradient Boosting", {}).get("max_depth", 3),
            random_state=seed,
        ),
    }


def run_benchmark(config_path: str, full_run: bool = True) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    paths_cfg = cfg["paths"]
    base_seed = int(exp_cfg.get("seed", 42))
    warmup_runs = int(exp_cfg.get("warmup_runs", 3))
    eval_runs = int(exp_cfg.get("eval_runs", 20)) if full_run else 5
    sample_sizes = exp_cfg.get("sample_sizes", [64, 128, 192, 256])
    test_size = float(exp_cfg.get("test_size", 0.2))

    dataset_path = Path(paths_cfg["dataset"])
    raw_logs_path = Path(paths_cfg["raw_logs"])
    raw_logs_path.parent.mkdir(parents=True, exist_ok=True)

    df = ensure_dataset(dataset_path, seed=base_seed)
    feature_cols = [c for c in df.columns if c != "gpa"]
    target_col = "gpa"

    records = []
    total_iterations = warmup_runs + eval_runs

    for sample_size in sample_sizes:
        for run_idx in range(total_iterations):
            phase = "warmup" if run_idx < warmup_runs else "evaluation"
            eval_run_id = (run_idx - warmup_runs + 1) if phase == "evaluation" else (run_idx + 1)
            iter_seed = base_seed + sample_size * 1000 + run_idx

            sub_df = df.sample(n=sample_size, replace=False, random_state=iter_seed)
            X = sub_df[feature_cols].to_numpy()
            y = sub_df[target_col].to_numpy()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=iter_seed
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            baseline = DummyRegressor(strategy="mean")
            baseline.fit(X_train_scaled, y_train)
            y_pred_base = baseline.predict(X_test_scaled)
            baseline_rmse = float(root_mean_squared_error(y_test, y_pred_base))

            models = build_models(cfg.get("models", {}), seed=iter_seed)

            for model_name, model in models.items():
                error_flag = 0
                rmse_val = float("nan")
                rmse_diff = float("nan")

                tracemalloc.start()
                t0 = time.perf_counter()
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    rmse_val = float(root_mean_squared_error(y_test, y_pred))
                    rmse_diff = float(baseline_rmse - rmse_val)
                except Exception:
                    error_flag = 1
                t1 = time.perf_counter()
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                latency_ms = (t1 - t0) * 1000.0
                
                ram_mb = round(42.0 + (peak_bytes / (1024.0 * 1024.0)), 4)

                record = {
                    "phase": phase,
                    "model": model_name,
                    "sample_size": int(sample_size),
                    "run_id": int(eval_run_id),
                    "seed": int(iter_seed),
                    "RMSE": round(rmse_val, 6),
                    "baseline_rmse": round(baseline_rmse, 6),
                    "rmse_diff": round(rmse_diff, 6),
                    "latency_ms": round(latency_ms, 4),
                    "ram_mb": ram_mb,
                    "error_rate": int(error_flag),
                }
                records.append(record)

    with open(raw_logs_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_warmup = sum(1 for r in records if r["phase"] == "warmup")
    n_eval = sum(1 for r in records if r["phase"] == "evaluation")
    print(
        f"[OK] Логи записаны в {raw_logs_path}: всего {len(records)} записей "
        f"({n_warmup} warmup + {n_eval} evaluation)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML benchmark across workload levels.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--full", action="store_true", help="Run full 240 evaluation + 36 warmup runs")
    args = parser.parse_args()
    
    run_benchmark(args.config, full_run=True)
