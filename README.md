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

|Model / Method|RMSE ($\text{Mean} \pm \text{SD}$)|RMSE $\text{Median}$ ($p_{25}\text{--}p_{75}$)|RMSE $95\%\text{ CI}$ (Bootstrap)|RMSE $p_{95} / p_{99}$|MAE ($\text{Mean} \pm \text{SD}$)|Latency $\text{Median}$ ($p_{95} / p_{99}$)|Guardrail Error Rate|
|---|---|---|---|---|---|---|---|
|Baseline: Ridge Regression ($\alpha=1.0$)|$0.3338 \pm 0.0245$|$0.3335$ ($0.3177\text{--}0.3510$)|$[0.3233, 0.3441]$|$0.3731 / 0.3768$|$0.2719 \pm 0.0195$|$1.01\text{ ms}$ ($1.08 / 1.12\text{ ms}$)|0%|
|Gradient Boosting Regressor|$0.2918 \pm 0.0278$|$0.2919$ ($0.2741\text{--}0.3082$)|$[0.2797, 0.3037]$|$0.3402 / 0.3486$|$0.2224 \pm 0.0218$|$75.40\text{ ms}$ ($80.82 / 82.15\text{ ms}$)|0%|
|Random Forest Regressor (Proposed)|$0.2866 \pm 0.0220$|$0.2926$ ($0.2718\text{--}0.3015$)|$[0.2772, 0.2960]$|$0.3152 / 0.3214$|$0.2311 \pm 0.0184$|$141.85\text{ ms}$ ($158.49 / 161.20\text{ ms}$)|0%|

### 5.2. Key Scientific Findings & Hypothesis Verification
1. Confirmation of Alternative Hypothesis ($H_1$): At the full dataset scale ($N = 256$), both non-linear ensemble methods significantly outperform the linear Ridge Regression baseline ($\text{RMSE} = 0.3338$). Specifically, RandomForestRegressor reduces prediction error by $14.1\%$ ($\text{RMSE} = 0.2866$, $95\%\text{ CI } [0.2772, 0.2960]$), while GradientBoostingRegressor achieves a $12.6\%$ reduction ($\text{RMSE} = 0.2918$, $95\%\text{ CI } [0.2797, 0.3037]$) and the lowest Mean Absolute Error ($\text{MAE} = 0.2224$). Because the $95\%$ bootstrap confidence intervals of the ensemble models do not overlap with the baseline mean, we reject the null hypothesis $H_0$ ($p < 0.05$).
2. Accuracy–Latency Trade-off & Guardrail Compliance: While Ridge Regression executes in $\sim 1.01\text{ ms}$, GradientBoosting ($\sim 75.6\text{ ms}$) and RandomForest ($\sim 142.2\text{ ms}$) require two orders of magnitude more compute time due to tree ensemble construction. Nevertheless, all models remain well below the Guardrail threshold ($\text{Latency} \le 2000\text{ ms}$, $\text{RAM} \le 256\text{ MB}$), achieving a $0.0\%$ error rate.

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
