import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

STYLE_MAP = {
    "Ridge": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "Random Forest": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
    "Gradient Boosting": {"color": "#009E73", "linestyle": "-.", "marker": "^"},
}


def save_both_formats(fig: plt.Figure, out_dir: Path, base_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{base_name}.pdf"
    png_path = out_dir / f"{base_name}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"[OK] Сохранены графики: {pdf_path} и {png_path} (300 DPI)")


def generate_plots(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    summary_csv = Path(cfg["paths"]["summary_csv"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    df = pd.read_csv(summary_csv)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
    })

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for model, grp in df.groupby("model", sort=False):
        grp = grp.sort_values("sample_size")
        st = STYLE_MAP.get(model, {"color": "#333333", "linestyle": "-", "marker": "o"})
        x = grp["sample_size"].to_numpy()
        y = grp["RMSE_median"].to_numpy()
        ci_low = grp["RMSE_ci_low"].to_numpy()
        ci_high = grp["RMSE_ci_high"].to_numpy()

        ax.plot(
            x, y,
            label=f"{model} (Median ± 95% CI)",
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=2.0,
            markersize=6,
        )
        ax.fill_between(x, ci_low, ci_high, color=st["color"], alpha=0.18)

    base_grp = df.groupby("sample_size", as_index=False)["baseline_RMSE_median"].mean()
    ax.plot(
        base_grp["sample_size"],
        base_grp["baseline_RMSE_median"],
        label="Baseline (Mean DummyRegressor)",
        color="#7F7F7F",
        linestyle=":",
        marker="x",
        linewidth=1.8,
    )

    ax.set_title("Model Prediction Error Across Workload Sizes")
    ax.set_xlabel("Sample Size (N observations)")
    ax.set_ylabel("Prediction Error, RMSE (GPA points)")
    ax.set_xticks([64, 128, 192, 256])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    save_both_formats(fig, figures_dir, "rmse_vs_sample_size")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for model, grp in df.groupby("model", sort=False):
        grp = grp.sort_values("sample_size")
        st = STYLE_MAP.get(model, {"color": "#333333", "linestyle": "-", "marker": "o"})
        x = grp["sample_size"].to_numpy()
        y = grp["latency_ms_median"].to_numpy()
        ci_low = grp["latency_ms_ci_low"].to_numpy()
        ci_high = grp["latency_ms_ci_high"].to_numpy()

        ax.plot(
            x, y,
            label=f"{model} (Median ± 95% CI)",
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=2.0,
            markersize=6,
        )
        ax.fill_between(x, ci_low, ci_high, color=st["color"], alpha=0.18)

    ax.set_title("Training & Inference Latency Across Workload Sizes")
    ax.set_xlabel("Sample Size (N observations)")
    ax.set_ylabel("Execution Latency (ms)")
    ax.set_xticks([64, 128, 192, 256])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    save_both_formats(fig, figures_dir, "latency_vs_sample_size")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    offsets = {"Ridge": -4, "Random Forest": 0, "Gradient Boosting": 4}
    for model, grp in df.groupby("model", sort=False):
        grp = grp.sort_values("sample_size")
        st = STYLE_MAP.get(model, {"color": "#333333", "linestyle": "-", "marker": "o"})
        x = grp["sample_size"].to_numpy() + offsets.get(model, 0)
        y = grp["RMSE_diff_mean"].to_numpy()
        err_low = y - grp["RMSE_diff_ci_low"].to_numpy()
        err_high = grp["RMSE_diff_ci_high"].to_numpy() - y

        ax.errorbar(
            x, y,
            yerr=[err_low, err_high],
            label=f"{model} (ΔRMSE ± 95% Bootstrap CI)",
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            capsize=4,
            linewidth=1.8,
        )

    ax.axhline(0.0, color="#000000", linestyle="--", linewidth=1.2, label="H0 Threshold (ΔRMSE = 0)")
    ax.set_title("Paired RMSE Improvement Over Baseline (ΔRMSE = RMSE_base - RMSE_model)")
    ax.set_xlabel("Sample Size (N observations)")
    ax.set_ylabel("Paired RMSE Reduction, ΔRMSE (GPA points)")
    ax.set_xticks([64, 128, 192, 256])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    save_both_formats(fig, figures_dir, "paired_rmse_difference")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scientific plots in PDF and PNG (300 DPI).")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    generate_plots(args.config)
