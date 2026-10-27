import argparse
import json
import os
import time
import psutil
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_ram_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark(config_path: str, use_full_data: bool = False):
    cfg = load_config(config_path)
    seed = cfg["random_seed"]

    data_path = cfg["paths"]["full_data"] if use_full_data else cfg["paths"]["sample_data"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    features = [col for col in cfg["features"]["numeric_features"] if col in df.columns]
    target = cfg["features"]["target"]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=seed
    )

    models = {
        "Baseline_Ridge": Ridge(alpha=cfg["models"]["baseline_ridge"]["alpha"], random_state=seed),
        "RandomForest": RandomForestRegressor(
            n_estimators=cfg["models"]["random_forest"]["n_estimators"],
            max_depth=cfg["models"]["random_forest"]["max_depth"],
            random_state=seed,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=cfg["models"]["gradient_boosting"]["n_estimators"],
            learning_rate=cfg["models"]["gradient_boosting"]["learning_rate"],
            max_depth=cfg["models"]["gradient_boosting"]["max_depth"],
            random_state=seed,
        ),
    }

    results = []
    for name, model in models.items():
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        ram_mb = get_ram_usage_mb()

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        guardrail_passed = (
            latency_ms <= cfg["guardrails"]["max_latency_ms"]
            and ram_mb <= cfg["guardrails"]["max_ram_mb"]
        )

        res = {
            "model": name,
            "rows_evaluated": len(df),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            "latency_ms": round(latency_ms, 2),
            "ram_mb": round(ram_mb, 2),
            "guardrail_passed": guardrail_passed,
        }
        results.append(res)
        print(f"[{name}] RMSE: {res['RMSE']} | MAE: {res['MAE']} | Latency: {res['latency_ms']} ms | RAM: {res['ram_mb']} MB")

    os.makedirs(os.path.dirname(cfg["paths"]["output_log"]), exist_ok=True)
    with open(cfg["paths"]["output_log"], "w", encoding="utf-8") as f:
        json.dump({"experiment": cfg["experiment_name"], "results": results}, f, indent=4)
    print(f"\nBenchmark finished successfully. Log saved to {cfg['paths']['output_log']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPA Prediction Benchmark")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--full", action="store_true", help="Use full dataset instead of sample")
    args = parser.parse_args()
    run_benchmark(args.config, args.full)
