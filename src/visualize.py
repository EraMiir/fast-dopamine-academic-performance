import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from process_data import bootstrap_ci


def generate_all_plots(eval_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: str = "reports/figures"):
    """Generate publication-ready plots (.pdf and 300 DPI .png) with 95% CI and colorblind-safe styles."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    palette = sns.color_palette("colorblind", 3)
    model_styles = {
        "Baseline_Ridge": {"color": palette[0], "marker": "o", "linestyle": "--", "label": "Baseline: Ridge Regression"},
        "GradientBoosting": {"color": palette[1], "marker": "s", "linestyle": "-.", "label": "Gradient Boosting Regressor"},
        "RandomForest": {"color": palette[2], "marker": "^", "linestyle": "-", "label": "Random Forest Regressor"},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, style in model_styles.items():
        sub = summary_df[summary_df["model"] == model_name].sort_values("sample_size")
        x = sub["sample_size"].values

        ax1.plot(x, sub["latency_mean_ms"], marker=style["marker"], linestyle=style["linestyle"],
                 color=style["color"], linewidth=2.2, markersize=7, label=style["label"])
        ax1.fill_between(x, sub["latency_ci95_low_ms"], sub["latency_ci95_high_ms"], color=style["color"], alpha=0.22)

        ax2.plot(x, sub["RMSE_mean"], marker=style["marker"], linestyle=style["linestyle"],
                 color=style["color"], linewidth=2.2, markersize=7, label=style["label"])
        ax2.fill_between(x, sub["RMSE_ci95_low"], sub["RMSE_ci95_high"], color=style["color"], alpha=0.22)

    ax1.set_yscale("log")
    ax1.set_xlabel("Sample Size / Workload Volume (N rows)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Execution Latency (ms, log scale)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Execution Latency vs. Sample Load (with 95% CI)", fontsize=12, fontweight="bold")
    ax1.set_xticks([64, 128, 192, 256])
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(loc="center right", frameon=True)

    ax2.set_xlabel("Sample Size / Workload Volume (N rows)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Prediction Error — RMSE (GPA Points, 0–4 Scale)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Prediction Error (RMSE) vs. Sample Load (with 95% CI)", fontsize=12, fontweight="bold")
    ax2.set_xticks([64, 128, 192, 256])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path / "fig1_latency_vs_load.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(out_path / "fig1_latency_vs_load.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    order = ["Baseline_Ridge", "GradientBoosting", "RandomForest"]
    labels = ["Baseline\n(Ridge)", "Gradient\nBoosting", "Random\nForest"]

    sns.boxplot(data=eval_df, x="model", y="RMSE", order=order, palette=palette,
                width=0.45, ax=ax1, boxprops=dict(alpha=0.75), showfliers=False)
    for idx_m, m_name in enumerate(order):
        vals = eval_df[eval_df["model"] == m_name]["RMSE"].values
        mean_v = np.mean(vals)
        ci_l, ci_h = bootstrap_ci(vals)
        ax1.errorbar(idx_m, mean_v, yerr=[[mean_v - ci_l], [ci_h - mean_v]],
                     fmt="D", color="black", capsize=6, capthick=1.8, markersize=6,
                     label="Mean ± 95% CI" if idx_m == 0 else "")

    ax1.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Evaluated Machine Learning Model", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Prediction Error — RMSE (GPA Points)", fontsize=11, fontweight="bold")
    ax1.set_title("A. RMSE Distribution & 95% Bootstrap CI (Evaluation Runs)", fontsize=11.5, fontweight="bold")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    sns.boxplot(data=eval_df, x="model", y="latency_ms", order=order, palette=palette,
                width=0.45, ax=ax2, boxprops=dict(alpha=0.75), showfliers=False)
    for idx_m, m_name in enumerate(order):
        vals = eval_df[eval_df["model"] == m_name]["latency_ms"].values
        mean_v = np.mean(vals)
        ci_l, ci_h = bootstrap_ci(vals)
        ax2.errorbar(idx_m, mean_v, yerr=[[mean_v - ci_l], [ci_h - mean_v]],
                     fmt="D", color="black", capsize=6, capthick=1.8, markersize=6,
                     label="Mean ± 95% CI" if idx_m == 0 else "")

    ax2.set_yscale("log")
    ax2.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Evaluated Machine Learning Model", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Execution Latency (ms, log scale)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Execution Latency Distribution & 95% CI (Log Scale)", fontsize=11.5, fontweight="bold")
    ax2.grid(True, axis="y", which="both", linestyle=":", alpha=0.6)
    ax2.legend(loc="center right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path / "fig2_distribution_boxplot.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(out_path / "fig2_distribution_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scientific plots for Task 3.")
    parser.add_argument("--raw", type=str, default="data/raw/test_run.jsonl")
    parser.add_argument("--summary", type=str, default="data/processed/metrics_summary.csv")
    parser.add_argument("--outdir", type=str, default="reports/figures")
    args = parser.parse_args()

    raw_df = pd.read_json(args.raw, lines=True)
    eval_df = raw_df[raw_df["phase"] == "evaluation"].copy()
    summary_df = pd.read_csv(args.summary)
    generate_all_plots(eval_df, summary_df, args.outdir)
