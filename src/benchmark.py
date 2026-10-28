import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


DEFAULT_CONFIG = {
    "experiment_name": "student_gpa_social_media_benchmark",
    "random_seed": 42,
    "test_size": 0.2,
    "bootstrap_iterations": 1000,
    "warmup_runs": 3,
    "evaluation_runs": 20,
    "load_levels": [64, 128, 192, 256],
    "paths": {
        "sample_data": "data/sample/sample_survey.csv",
        "full_data": "data/cleaned_survey.csv",
        "raw_jsonl": "data/raw/test_run.jsonl",
        "processed_csv": "data/processed/metrics_summary.csv",
        "figures_dir": "reports/figures",
    },
    "features": {
        "target": "Last Academic Result (GPA/CGPA)",
        "numeric_features": [
            "Age",
            "Study time (In Hours)",
            "Attendance rate (In Percentile)",
            "Time spent in social media (hours)",
            "Sleep Disturbance on Sleep Quality",
            "Mood Modification Scale",
            "Anxiety Scale ",
            "Depression Scale",
            "Self-esteem Scale",
            "Social Media Distraction During Academic Activities ",
        ],
    },
    "models": {
        "baseline_ridge": {"alpha": 1.0},
        "random_forest": {"n_estimators": 100, "max_depth": 5},
        "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
    },
    "guardrails": {"max_latency_ms": 2000.0, "max_ram_mb": 512.0},
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            pass
    return DEFAULT_CONFIG


def get_rss_memory_mb() -> float:
    """Measure real process Resident Set Size (RSS) memory in MB without artificial offsets."""
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))
    except ImportError:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(usage / (1024.0 * 1024.0))
        return float(usage / 1024.0)


def run_benchmark(config_path: str = "configs/config.yaml", use_full: bool = True) -> pd.DataFrame:
    cfg = load_config(config_path)

    data_path = cfg["paths"]["full_data"] if use_full else cfg["paths"]["sample_data"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Real survey dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    features = [c for c in cfg["features"]["numeric_features"] if c in df.columns]
    target = cfg["features"]["target"]

    X_full = df[features].values
    y_full = df[target].values

    base_seed = int(cfg["random_seed"])
    warmup_runs = int(cfg.get("warmup_runs", 3))
    eval_runs = int(cfg.get("evaluation_runs", 20))
    load_levels = cfg.get("load_levels", [64, 128, 192, 256]) if use_full else [len(df)]

    models_cfg = {
        "Baseline_Ridge": lambda s: Ridge(alpha=cfg["models"]["baseline_ridge"]["alpha"], random_state=s),
        "RandomForest": lambda s: RandomForestRegressor(
            n_estimators=cfg["models"]["random_forest"]["n_estimators"],
            max_depth=cfg["models"]["random_forest"]["max_depth"],
            random_state=s,
        ),
        "GradientBoosting": lambda s: GradientBoostingRegressor(
            n_estimators=cfg["models"]["gradient_boosting"]["n_estimators"],
            learning_rate=cfg["models"]["gradient_boosting"]["learning_rate"],
            max_depth=cfg["models"]["gradient_boosting"]["max_depth"],
            random_state=s,
        ),
    }

    raw_records = []
    run_id = 0
    total_iters = warmup_runs + eval_runs

    for model_name, model_fn in models_cfg.items():
        for n_samples in load_levels:
            for i in range(total_iters):
                phase = "warmup" if i < warmup_runs else "evaluation"
                seed = base_seed + i * 7 + int(n_samples)
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(df), size=int(n_samples), replace=True)
                X_sub = X_full[idx]
                y_sub = y_full[idx]

                split = max(1, int((1.0 - float(cfg["test_size"])) * len(X_sub)))
                X_tr, X_te = X_sub[:split], X_sub[split:]
                y_tr, y_te = y_sub[:split], y_sub[split:]

                model = model_fn(seed)
                t0 = time.perf_counter()
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                ram_mb = get_rss_memory_mb()
                rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
                mae = float(mean_absolute_error(y_te, preds))
                r2 = float(r2_score(y_te, preds)) if len(y_te) > 1 else 0.0

                error_rate = 0.0 if (
                    latency_ms <= cfg["guardrails"]["max_latency_ms"]
                    and ram_mb <= cfg["guardrails"]["max_ram_mb"]
                    and not np.isnan(rmse)
                ) else 1.0

                raw_records.append({
                    "run_id": run_id,
                    "phase": phase,
                    "iteration": i,
                    "model": model_name,
                    "sample_size": int(n_samples),
                    "RMSE": round(rmse, 4),
                    "MAE": round(mae, 4),
                    "R2": round(r2, 4),
                    "latency_ms": round(latency_ms, 3),
                    "ram_mb": round(ram_mb, 2),
                    "error_rate": error_rate,
                })
                run_id += 1

    out_jsonl = Path(cfg["paths"]["raw_jsonl"])
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in raw_records:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(raw_records)} benchmark records to {out_jsonl}")
    return pd.DataFrame(raw_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPA Prediction Benchmark on Real Survey Dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--sample", action="store_true", help="Run on 10-row sample instead of full 256-row dataset")
    parser.add_argument("--full", action="store_true", help="Run on full 256-row dataset (default)")
    args = parser.parse_args()
    use_full_flag = not args.sample
    run_benchmark(args.config, use_full=use_full_flag)
