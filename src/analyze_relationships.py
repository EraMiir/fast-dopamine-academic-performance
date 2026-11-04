from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

DATA_PATH = Path("data/cleaned_survey.csv")
OUTPUT_PATH = Path("data/processed/feature_gpa_correlations.csv")
README_PATH = Path("README.md")

TARGET = "Last Academic Result (GPA/CGPA)"

FEATURES = {
    "Study time (In Hours)": "Study time (hours)",
    "Attendance rate (In Percentile)": "Attendance rate",
    "Time spent in social media (hours)": "Social media time (hours)",
    "Sleep Disturbance on Sleep Quality": "Sleep disturbance",
    "Social Media Distraction During Academic Activities": "Academic social media distraction",
}

SEED = 42
BOOTSTRAP_ITERATIONS = 2000


def bootstrap_spearman_ci(x, y, iterations=BOOTSTRAP_ITERATIONS, seed=SEED):
    """Percentile bootstrap 95% CI for Spearman correlation."""
    rng = np.random.default_rng(seed)
    values = []

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        rho = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(rho):
            values.append(rho)

    if not values:
        return np.nan, np.nan

    return tuple(np.quantile(values, [0.025, 0.975]))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        raise KeyError(f"Target column not found: {TARGET}")

    rows = []

    for column, label in FEATURES.items():
        if column not in df.columns:
            print(f"SKIPPED: column not found: {column}")
            continue

        pair = df[[column, TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
        x = pair[column].to_numpy(dtype=float)
        y = pair[TARGET].to_numpy(dtype=float)

        if len(pair) < 5 or np.std(x) == 0 or np.std(y) == 0:
            print(f"SKIPPED: insufficient variation/data: {label}")
            continue

        rho, p_value = spearmanr(x, y)
        pearson_r, _ = pearsonr(x, y)
        ci_low, ci_high = bootstrap_spearman_ci(x, y)

        if rho > 0.1:
            direction = "Positive"
        elif rho < -0.1:
            direction = "Negative"
        else:
            direction = "Weak / near zero"

        rows.append({
            "factor": label,
            "source_column": column,
            "n": len(pair),
            "spearman_rho": rho,
            "spearman_ci95_low": ci_low,
            "spearman_ci95_high": ci_high,
            "spearman_p_value": p_value,
            "pearson_r": pearson_r,
            "direction": direction,
        })

    if not rows:
        raise RuntimeError("No valid correlations were calculated.")

    result = pd.DataFrame(rows).sort_values(
        "spearman_rho", key=lambda s: s.abs(), ascending=False
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    # Generate a README section using the actual calculated values.
    lines = [
        "## 5.3. Exploratory Feature–GPA Relationship Analysis",
        "",
        "Spearman rank correlation was used to examine monotonic associations "
        "between selected survey factors and GPA. Pearson correlation is also "
        "reported as a supplementary measure. 95% confidence intervals for "
        "Spearman's rho were estimated using 2,000 bootstrap resamples "
        "(random seed = 42).",
        "",
        "| Factor | N | Spearman's rho | 95% bootstrap CI | p-value | Pearson's r | Direction |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for row in result.to_dict("records"):
        lines.append(
            f"| {row['factor']} | {row['n']} | "
            f"{row['spearman_rho']:.3f} | "
            f"[{row['spearman_ci95_low']:.3f}, {row['spearman_ci95_high']:.3f}] | "
            f"{row['spearman_p_value']:.4f} | "
            f"{row['pearson_r']:.3f} | {row['direction']} |"
        )

    lines += [
        "",
        "**Interpretation:** Positive coefficients indicate that higher factor "
        "values tend to occur with higher GPA; negative coefficients indicate "
        "that higher factor values tend to occur with lower GPA. Coefficients "
        "near zero indicate weak monotonic association.",
        "",
        "**Important limitation:** These are observational correlations, not "
        "evidence of causation. They do not establish that a factor causes GPA "
        "to increase or decrease. Statistical significance should be interpreted "
        "cautiously because several factors are examined.",
        "",
        f"Full results: `{OUTPUT_PATH.as_posix()}`.",
    ]

    section = "\n".join(lines)
    readme = README_PATH.read_text(encoding="utf-8")

    start_marker = "<!-- FEATURE_GPA_ANALYSIS_START -->"
    end_marker = "<!-- FEATURE_GPA_ANALYSIS_END -->"
    wrapped = f"{start_marker}\n{section}\n{end_marker}"

    if start_marker in readme and end_marker in readme:
        before = readme.split(start_marker)[0]
        after = readme.split(end_marker, 1)[1]
        readme = before + wrapped + after
    else:
        readme = readme.rstrip() + "\n\n" + wrapped + "\n"

    README_PATH.write_text(readme, encoding="utf-8")

    print("\nFEATURE–GPA CORRELATION RESULTS")
    print(result[[
        "factor", "n", "spearman_rho",
        "spearman_ci95_low", "spearman_ci95_high",
        "spearman_p_value", "direction"
    ]].round(4).to_string(index=False))

    print(f"\nSaved results: {OUTPUT_PATH}")
    print("README.md updated successfully.")


if __name__ == "__main__":
    main()
