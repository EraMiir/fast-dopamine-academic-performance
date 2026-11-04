# Fast Dopamine & Academic Performance: Predicting Student GPA Based on Social Media Usage and Psychometric Indicators

**Authors:** Sharap Alislam, Miir Yerassyl  
**Repository:** `https://github.com/EraMiir/fast-dopamine-academic-performance`  
**Course / Assignment:** Task 3 — Research Results, Data Analysis, and Scientific Visualization

---

## 1. Overview & Research Questions

This repository contains the end-to-end benchmarking, statistical aggregation, and scientific visualization pipeline evaluating machine learning models that predict student academic performance (`Last Academic Result (GPA/CGPA)`) from behavioral social media ("fast dopamine") metrics and psychometric indicators ($N = 256$ survey responses).

* **RQ1:** To what extent do non-linear ensemble machine learning models (`RandomForestRegressor`, `GradientBoostingRegressor`) reduce the prediction error ($\text{RMSE}$, $\text{MAE}$) of student GPA compared to a linear baseline (`Ridge Regression`) when incorporating social media distraction and psychometric indicators?
* **RQ2:** How do non-linear interactions between late-night social media usage, academic distraction, and sleep disturbance impact GPA prediction accuracy while controlling for study time and attendance rate?

---

## 2. Final Repository Directory Structure

```text
fast-dopamine-academic-performance/
├── configs/
│   └── config.yaml                     # Experiment hyperparameters and guardrail thresholds
├── data/
│   ├── raw/
│   │   └── test_run.jsonl              # Raw benchmark logs (separated into warmup and evaluation runs)
│   ├── processed/
│   │   └── metrics_summary.csv         # Cleaned and aggregated statistical metrics (Mean, SD, Median, p95, p99, 95% CI)
│   ├── sample/
│   │   └── sample_survey.csv           # 10-row micro-sample for smoke testing
│   └── cleaned_survey.csv              # Full preprocessed survey dataset (N = 256, 24 features)
├── docs/
│   └── domain_model.md                 # Formal LaTeX Loss/Metric Equations & C4 Telemetry Diagram
├── notebooks/
│   └── 01_data_analysis.ipynb          # Reproducible Jupyter Notebook (executed in sequential order)
├── reports/
│   └── figures/
│       ├── fig1_latency_vs_load.pdf    # Vector publication plot: Latency & RMSE vs. Load
│       ├── fig1_latency_vs_load.png    # 300 DPI raster plot: Latency & RMSE vs. Load
│       ├── fig2_distribution_boxplot.pdf # Vector publication plot: Metric Distributions & 95% CI
│       └── fig2_distribution_boxplot.png # 300 DPI raster plot: Metric Distributions & 95% CI
├── src/
│   ├── benchmark.py                    # Benchmark execution script
│   ├── process_data.py                 # Warm-up filtering and 1000-iteration bootstrap aggregation module
│   └── visualize.py                    # Scientific visualization generator (colorblind-safe, 300 DPI + PDF)
├── .gitignore                          # Excludes .ipynb_checkpoints, __pycache__, .venv, and raw dumps
├── Dockerfile                          # Reproducible Docker container with exact dependency pins
├── requirements.txt                    # Pinned Python dependencies
├── LICENSE                             # MIT License
└── README.md                           # Final report with Results & Discussion, plots, and domain model
```

## 3. Planned Benchmarks & Metrics Matrix
|Model / Condition|Feature Set|Primary Metric|Secondary Metrics|Guardrail Metrics|
|---|---|---|---|---|
|Baseline: Ridge Regression ($\alpha=1.0$)|Academic + Social Media + Psychometric|RMSE|$MAE$, $R^2$|Latency $\le 2000\text{ ms}$, RAM $\le 256\text{ MB}$|
|Random Forest Regressor|Academic + Social Media + Psychometric|RMSE|$MAE$, $R^2$|Latency $\le 2000\text{ ms}$, RAM $\le 256\text{ MB}$|
|Gradient Boosting Regressor|Academic + Social Media + Psychometric|RMSE|$MAE$, $R^2$|Latency $\le 2000\text{ ms}$, RAM $\le 256\text{ MB}$|

## 4. Domain Model & Formalization

The complete mathematical formalization of the loss functions ($\mathcal{L}_{\text{Ridge}}, \mathcal{L}_{\text{GBR}}$), target metrics ($\text{RMSE}, \text{MAE}$), non-parametric bootstrap 95% confidence intervals, and the C4 Component Telemetry Diagram are documented in docs/domain_model.md.

## 5. Results & Discussion
### 5.1. Summary Benchmark Table ($N = 256$ Full Workload, Cold-Start Warm-up Filtered)

All metrics are computed strictly on post-warmup evaluation runs ($n = 20$ evaluation repetitions per load level; $240$ total evaluation runs across $N \in \{64, 128, 192, 256\}$, np.random.seed(42)).

|Model / Method|RMSE Mean ± SD|RMSE Median (p25–p75)|RMSE 95% CI|RMSE p95 / p99|MAE Mean ± SD|Latency Median (p95 / p99)|Error Rate|
|---|---:|---:|---:|---:|---:|---:|---:|
|Baseline: Ridge Regression|0.3338 ± 0.0245|0.3335 (0.3129–0.3500)|[0.3233, 0.3442]|0.3731 / 0.3766|0.2677 ± 0.0206|1.07 ms (1.49 / 1.60 ms)|0.0%|
|Gradient Boosting Regressor|0.2918 ± 0.0278|0.2919 (0.2850–0.3043)|[0.2793, 0.3038]|0.3402 / 0.3425|0.2192 ± 0.0218|80.64 ms (94.74 / 96.93 ms)|0.0%|
|Random Forest Regressor|0.2866 ± 0.0220|0.2926 (0.2735–0.3009)|[0.2771, 0.2956]|0.3153 / 0.3160|0.2221 ± 0.0172|173.45 ms (186.63 / 187.36 ms)|0.0%|

### 5.2. Key Findings & Methodological Notes

1. **Prediction accuracy:** the lowest mean RMSE at sample size 256 is achieved by **Random Forest Regressor** (0.2866 GPA points; 95% bootstrap CI [0.2771, 0.2956]).
2. **Comparison with baseline:** the reduction in mean RMSE relative to Ridge is 14.1%.
3. **Uncertainty:** confidence intervals and dispersion statistics are reported in the table above. Interval overlap should not be treated as a formal hypothesis test.
4. **Runtime:** latency is reported in milliseconds. The median, 95th and 99th percentiles are shown above to make runtime variability visible.

## 6. Quickstart & Reproducibility Guide

```bash
# 1. Install exact pinned dependencies
pip install -r requirements.txt

# 2. Run benchmark on the full 256-row real survey dataset (or use --sample for the 10-row smoke test)
python src/benchmark.py --config configs/config.yaml --full

# 3. Filter warm-up runs and compute 1,000-iteration bootstrap statistics (including p95 & p99)
python src/process_data.py --config configs/config.yaml

# 4. Generate publication-ready PDF and 300 DPI PNG figures in reports/figures/
python src/visualize.py --config configs/config.yaml
```


## Scientific Visualizations

![Latency vs Load](reports/figures/fig1_latency_vs_load.png)

![Distribution Boxplot](reports/figures/fig2_distribution_boxplot.png)

<!-- FEATURE_GPA_ANALYSIS_START -->
## 5.3. Exploratory Feature–GPA Relationship Analysis

Spearman rank correlation was used to examine monotonic associations between selected survey factors and GPA. Pearson correlation is also reported as a supplementary measure. 95% confidence intervals for Spearman's rho were estimated using 2,000 bootstrap resamples (random seed = 42).

| Factor | N | Spearman's rho | 95% bootstrap CI | p-value | Pearson's r | Direction |
|---|---:|---:|---:|---:|---:|---|
| Study time (hours) | 256 | 0.302 | [0.192, 0.409] | 0.0000 | 0.280 | Positive |
| Attendance rate | 256 | 0.235 | [0.119, 0.349] | 0.0001 | 0.282 | Positive |
| Academic social media distraction | 256 | -0.175 | [-0.298, -0.052] | 0.0051 | -0.154 | Negative |
| Sleep disturbance | 256 | -0.090 | [-0.213, 0.039] | 0.1511 | -0.090 | Weak / near zero |
| Social media time (hours) | 256 | -0.046 | [-0.161, 0.069] | 0.4599 | -0.056 | Weak / near zero |

**Interpretation:** Positive coefficients indicate that higher factor values tend to occur with higher GPA; negative coefficients indicate that higher factor values tend to occur with lower GPA. Coefficients near zero indicate weak monotonic association.

**Important limitation:** These are observational correlations, not evidence of causation. They do not establish that a factor causes GPA to increase or decrease. Statistical significance should be interpreted cautiously because several factors are examined.

Full results: `data/processed/feature_gpa_correlations.csv`.
<!-- FEATURE_GPA_ANALYSIS_END -->
