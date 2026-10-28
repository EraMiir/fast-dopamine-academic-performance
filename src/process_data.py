import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import load_config


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Compute 95% non-parametric bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    boot_means = [
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def process_benchmark_logs(
    raw_jsonl_path: str = "data/raw/test_run.jsonl",
    output_csv_path: str = "data/processed/metrics_summary.csv",
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw .jsonl logs, filter out cold-start warm-up runs, and compute summary statistics including p95 and p99."""
    np.random.seed(seed)
    raw_df = pd.read_json(raw_jsonl_path, lines=True)
    eval_df = raw_df[raw_df["phase"] == "evaluation"].copy()

    summary_rows = []
    for (model, sample_size), grp in eval_df.groupby(["model", "sample_size"]):
        rmse_ci_l, rmse_ci_h = bootstrap_ci(grp["RMSE"].values, n_boot=n_boot, seed=seed)
        lat_ci_l, lat_ci_h = bootstrap_ci(grp["latency_ms"].values, n_boot=n_boot, seed=seed)

        summary_rows.append({
            "model": model,
            "sample_size": int(sample_size),
            "n_runs": int(len(grp)),
            "RMSE_mean": round(float(grp["RMSE"].mean()), 4),
            "RMSE_sd": round(float(grp["RMSE"].std()), 4),
            "RMSE_median": round(float(grp["RMSE"].median()), 4),
            "RMSE_p25": round(float(np.percentile(grp["RMSE"], 25)), 4),
            "RMSE_p75": round(float(np.percentile(grp["RMSE"], 75)), 4),
            "RMSE_p95": round(float(np.percentile(grp["RMSE"], 95)), 4),
            "RMSE_p99": round(float(np.percentile(grp["RMSE"], 99)), 4),
            "RMSE_ci95_low": round(rmse_ci_l, 4),
            "RMSE_ci95_high": round(rmse_ci_h, 4),
            "MAE_mean": round(float(grp["MAE"].mean()), 4),
            "MAE_sd": round(float(grp["MAE"].std()), 4),
            "MAE_median": round(float(grp["MAE"].median()), 4),
            "MAE_p95": round(float(np.percentile(grp["MAE"], 95)), 4),
            "MAE_p99": round(float(np.percentile(grp["MAE"], 99)), 4),
            "latency_mean_ms": round(float(grp["latency_ms"].mean()), 3),
            "latency_sd_ms": round(float(grp["latency_ms"].std()), 3),
            "latency_median_ms": round(float(grp["latency_ms"].median()), 3),
            "latency_p25_ms": round(float(np.percentile(grp["latency_ms"], 25)), 3),
            "latency_p75_ms": round(float(np.percentile(grp["latency_ms"], 75)), 3),
            "latency_p95_ms": round(float(np.percentile(grp["latency_ms"], 95)), 3),
            "latency_p99_ms": round(float(np.percentile(grp["latency_ms"], 99)), 3),
            "latency_ci95_low_ms": round(lat_ci_l, 3),
            "latency_ci95_high_ms": round(lat_ci_h, 3),
            "ram_mean_mb": round(float(grp["ram_mb"].mean()), 2),
            "error_rate": round(float(grp["error_rate"].mean()), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv_path, index=False)
    return raw_df, eval_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw benchmark logs and compute statistics.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", type=str, default=None, help="Override input JSONL path")
    parser.add_argument("--output", type=str, default=None, help="Override output CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = args.input or cfg["paths"].get("raw_jsonl", "data/raw/test_run.jsonl")
    out_path = args.output or cfg["paths"].get("processed_csv", "data/processed/metrics_summary.csv")
    n_boot = int(cfg.get("bootstrap_iterations", 1000))
    seed = int(cfg.get("random_seed", 42))

    _, _, summary_df = process_benchmark_logs(raw_path, out_path, n_boot=n_boot, seed=seed)
    print(f"Processed {len(summary_df)} aggregated rows -> {out_path}")
