import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon


def bootstrap_ci(
    data: np.ndarray,
    stat_func=np.median,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Непараметрический перцентильный bootstrap 95% доверительный интервал."""
    clean = data[~np.isnan(data)]
    if len(clean) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_samples = rng.choice(clean, size=(n_boot, len(clean)), replace=True)
    boot_stats = stat_func(boot_samples, axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_stats, 100.0 * alpha))
    high = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))
    return low, high


def process_logs(config_path: str) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    paths_cfg = cfg["paths"]
    seed = int(exp_cfg.get("seed", 42))
    n_boot = int(exp_cfg.get("bootstrap_iterations", 2000))
    ci_level = float(exp_cfg.get("confidence_level", 0.95))

    raw_logs_path = Path(paths_cfg["raw_logs"])
    summary_csv_path = Path(paths_cfg["summary_csv"])
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(raw_logs_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    df_all = pd.DataFrame(rows)
    total_rows = len(df_all)
    df_eval = df_all[df_all["phase"] == "evaluation"].copy()
    print(f"[INFO] Прочитано {total_rows} строк. После фильтрации warm-up осталось {len(df_eval)} evaluation-записей.")

    summary_rows = []
    for (model, sample_size), grp in df_eval.groupby(["model", "sample_size"], sort=False):
        n_runs = len(grp)
        error_rate = float(grp["error_rate"].mean())

        valid = grp[grp["error_rate"] == 0]
        rmse_arr = valid["RMSE"].to_numpy()
        base_arr = valid["baseline_rmse"].to_numpy()
        diff_arr = valid["rmse_diff"].to_numpy()  # baseline_rmse - model_rmse
        lat_arr = valid["latency_ms"].to_numpy()
        ram_arr = valid["ram_mb"].to_numpy()

        rmse_ci_low, rmse_ci_high = bootstrap_ci(rmse_arr, np.median, n_boot=n_boot, ci=ci_level, seed=seed)
        lat_ci_low, lat_ci_high = bootstrap_ci(lat_arr, np.median, n_boot=n_boot, ci=ci_level, seed=seed)
        ram_ci_low, ram_ci_high = bootstrap_ci(ram_arr, np.median, n_boot=n_boot, ci=ci_level, seed=seed)

        diff_ci_low, diff_ci_high = bootstrap_ci(diff_arr, np.mean, n_boot=n_boot, ci=ci_level, seed=seed)

        try:
            _, p_val = wilcoxon(base_arr, rmse_arr, alternative="greater")
            p_val = float(p_val)
        except Exception:
            p_val = 1.0

        summary_rows.append({
            "model": model,
            "sample_size": int(sample_size),
            "n_runs": int(n_runs),
            "RMSE_median": round(float(np.median(rmse_arr)), 4),
            "RMSE_mean": round(float(np.mean(rmse_arr)), 4),
            "RMSE_std": round(float(np.std(rmse_arr, ddof=1)), 4),
            "RMSE_p25": round(float(np.percentile(rmse_arr, 25)), 4),
            "RMSE_p75": round(float(np.percentile(rmse_arr, 75)), 4),
            "RMSE_p95": round(float(np.percentile(rmse_arr, 95)), 4),
            "RMSE_iqr": round(float(np.percentile(rmse_arr, 75) - np.percentile(rmse_arr, 25)), 4),
            "RMSE_ci_low": round(rmse_ci_low, 4),
            "RMSE_ci_high": round(rmse_ci_high, 4),
            "baseline_RMSE_median": round(float(np.median(base_arr)), 4),
            "RMSE_diff_mean": round(float(np.mean(diff_arr)), 4),
            "RMSE_diff_ci_low": round(diff_ci_low, 4),
            "RMSE_diff_ci_high": round(diff_ci_high, 4),
            "wilcoxon_p_value": round(p_val, 6),
            "significant_vs_baseline": bool(diff_ci_low > 0.0 and p_val < 0.05),
            "latency_ms_median": round(float(np.median(lat_arr)), 4),
            "latency_ms_p25": round(float(np.percentile(lat_arr, 25)), 4),
            "latency_ms_p75": round(float(np.percentile(lat_arr, 75)), 4),
            "latency_ms_p95": round(float(np.percentile(lat_arr, 95)), 4),
            "latency_ms_ci_low": round(lat_ci_low, 4),
            "latency_ms_ci_high": round(lat_ci_high, 4),
            "ram_mb_median": round(float(np.median(ram_arr)), 4),
            "ram_mb_p95": round(float(np.percentile(ram_arr, 95)), 4),
            "ram_mb_ci_low": round(ram_ci_low, 4),
            "ram_mb_ci_high": round(ram_ci_high, 4),
            "error_rate": round(error_rate, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[OK] Сводная таблица сохранена в {summary_csv_path} ({len(summary_df)} строк).")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw JSONL benchmark logs into summary statistics.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    process_logs(args.config)
