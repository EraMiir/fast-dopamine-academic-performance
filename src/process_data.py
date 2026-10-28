import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Compute 95% non-parametric bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    boot_means = [
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def process_benchmark_logs(raw_jsonl_path: str, output_csv_path: str, seed: int = 42) -> pd.DataFrame:
    """Load raw .jsonl logs, filter out cold-start warm-up runs, and aggregate statistics."""
    np.random.seed(seed)
    raw_df = pd.read_json(raw_jsonl_path, lines=True)

    # Filter out cold-start warm-up runs
    eval_df = raw_df[raw_df["phase"] == "evaluation"].copy()

    summary_rows = []
    for (model, sample_size), grp in eval_df.groupby(["model", "sample_size"]):
        rmse_ci_l, rmse_ci_h = bootstrap_ci(grp["RMSE"].values, seed=seed)
        lat_ci_l, lat_ci_h = bootstrap_ci(grp["latency_ms"].values, seed=seed)

        summary_rows.append({
            "model": model,
            "sample_size": int(sample_size),
            "n_runs": len(grp),
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
            "latency_mean_ms": round(float(grp["latency_ms"].mean()), 3),
            "latency_sd_ms": round(float(grp["latency_ms"].std()), 3),
            "latency_median_ms": round(float(grp["latency_ms"].median()), 3),
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
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw benchmark logs and compute statistics.")
    parser.add_argument("--input", type=str, default="data/raw/test_run.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/metrics_summary.csv")
    args = parser.parse_args()
    df_out = process_benchmark_logs(args.input, args.output)
    print(f"Processed {len(df_out)} aggregated benchmark configurations to {args.output}")
