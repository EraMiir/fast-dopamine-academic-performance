# Fast Dopamine & Academic Performance: Predicting Student GPA Based on Social Media Usage and Psychometric Indicators

**Authors:** Sharap Alislam, Miir Yerassyl  
**Repository:** `https://github.com/EraMiir/fast-dopamine-academic-performance`  
**Course / Assignment:** Task 3 — Research Results, Data Analysis, and Scientific Visualization

---

## 1. Overview & Research Questions

This repository contains the end-to-end benchmarking, statistical aggregation, and scientific visualization pipeline evaluating machine learning models that predict student academic performance (`Last Academic Result (GPA/CGPA)`) from behavioral social media ("fast dopamine") metrics and psychometric indicators ($N = 256$ survey responses).

* **RQ1:** To what extent do non-linear ensemble machine learning models (`RandomForestRegressor`, `GradientBoostingRegressor`) reduce the prediction error ($\text{RMSE}$, $\text{MAE}$) of student GPA compared to a linear baseline (`Ridge Regression`) when incorporating social media distraction and psychometric indicators?
* **RQ2 (exploratory):** Which selected survey factors show monotonic associations with GPA? The current analysis reports pairwise Spearman correlations; it does not estimate interaction effects or adjust associations jointly for study time and attendance.

---

## cleaning dataset

|Variable|How did they respond?|What We Did|
|---|---|---|
|Age|**21 years old**, **dd/mm/1999**|The number was extracted by converting to age (2025 − year)|
|StudyH|**3/4h**, **2 hrs**, **30 mins**|Extracted the time, minutes /60, ranges to average|
|Attend|**85%**, **0.9**, **90**|Converted to a fraction between 0 and 1|
|SM_hours|**less than 2**, **3-4 hours**, **20%**|Same as for StudyH|
|GPA|**4.0 and 5.0 scales**, **letters (A, B+, etc.)**|Letter grades are mapped to fixed 4.0-scale values. Numeric values from 0.0 to 4.0 are kept unchanged. Values above 4.0 and below 4.5 are mapped to 3.0; values from 4.5 to 5.0 are mapped to 4.0. Other unrecognized values are removed. This is an approximate harmonization, not a proportional conversion.|

## 2. Final Repository Directory Structure

```text
fast-dopamine-academic-performance/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   ├── survey.csv
│   │   └── test_run.jsonl
│   ├── processed/
│   │   ├── feature_gpa_correlations.csv
│   │   └── metrics_summary.csv
│   ├── sample/
│   │   └── sample_survey.csv
│   └── cleaned_survey.csv
├── docs/
│   ├── README.md
│   └── domain_model.md
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── data_cleaning.ipynb
│   └── README.md
├── reports/
│   └── figures/
│       ├── fig1_latency_vs_load.pdf
│       ├── fig1_latency_vs_load.png
│       ├── fig2_distribution_boxplot.pdf
│       └── fig2_distribution_boxplot.png
├── src/
│   ├── analyze_relationships.py
│   ├── benchmark.py
│   ├── process_data.py
│   └── visualize.py
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## 3. Planned Benchmarks & Metrics Matrix
|Model / Condition|Feature Set|Primary Metric|Secondary Metrics|Guardrail Metrics|
|---|---|---|---|---|
|Baseline: Ridge Regression ($\alpha=1.0$)|Academic + Social Media + Psychometric|RMSE|$MAE$, $R^2$|Latency $\le 2000\text{ ms}$, RAM $\le 512\text{ MB}$|
|Random Forest Regressor|Academic + Social Media + Psychometric|RMSE|$MAE$, $R^2$|Latency $\le 2000\text{ ms}$, RAM $\le 512\text{ MB}$|
|Gradient Boosting Regressor|Academic + Social Media + Psychometric|RMSE|$MAE$, $R^2$|Latency $\le 2000\text{ ms}$, RAM $\le 512\text{ MB}$|

## 4. Domain Model & Formalization

The complete mathematical formalization of the loss functions ($\mathcal{L}_{\text{Ridge}}, \mathcal{L}_{\text{GBR}}$), target metrics ($\text{RMSE}, \text{MAE}$), non-parametric bootstrap 95% confidence intervals, and the C4 Component Telemetry Diagram are documented in docs/domain_model.md.

## 5. Results & Discussion
### 5.1. Benchmark Results at the Full Workload (N = 256)

The corrected benchmark was run on the committed cleaned survey dataset. For each model and workload level, the pipeline used 3 warm-up runs followed by 20 evaluation runs. The table below reports the 20 evaluation runs at the full workload of 256 observations. Latency includes model training and prediction.

|Model|RMSE Mean ± SD|MAE Mean ± SD|RMSE 95% bootstrap CI|Median latency|Mean RAM|Error rate|
|---|---:|---:|---:|---:|---:|---:|
|Standardized Ridge Regression|0.3434 ± 0.0248|0.2744 ± 0.0237|[0.3321, 0.3545]|1.085 ms|160.66 MB|0.0%|
|Random Forest|0.3511 ± 0.0252|0.2875 ± 0.0221|[0.3400, 0.3625]|90.311 ms|161.93 MB|0.0%|
|Gradient Boosting|0.3575 ± 0.0299|0.2871 ± 0.0256|[0.3449, 0.3701]|48.400 ms|163.23 MB|0.0%|

Ridge has the lowest mean RMSE and MAE at N = 256 and is also the fastest model in this experiment. The two ensemble models did not outperform the standardized Ridge baseline on this dataset. All three models remained within the configured latency and memory guardrails during the recorded evaluation runs.

### 5.2. Academic-only vs Full Model

The second experiment used the same 30 reproducible 80/20 train-test splits for both feature sets. Academic-only contains age, study time, and attendance. Full Model adds social-media time, sleep disturbance, mood modification, anxiety, depression, self-esteem, and academic social-media distraction.

|Model|Feature set|RMSE Mean ± SD|MAE Mean ± SD|R² Mean|
|---|---|---:|---:|---:|
|Ridge|Academic-only|0.3393 ± 0.0253|0.2760 ± 0.0224|0.1207|
|Ridge|Full Model|0.3469 ± 0.0277|0.2800 ± 0.0248|0.0801|
|Random Forest|Academic-only|0.3524 ± 0.0228|0.2857 ± 0.0182|0.0501|
|Random Forest|Full Model|0.3545 ± 0.0239|0.2913 ± 0.0203|0.0404|
|Gradient Boosting|Academic-only|0.3582 ± 0.0238|0.2868 ± 0.0185|0.0173|
|Gradient Boosting|Full Model|0.3610 ± 0.0248|0.2903 ± 0.0217|0.0014|

In this repeated holdout comparison, adding the social-media and psychometric variables did not improve average prediction metrics for any of the three models. This does not prove that these factors are unrelated to GPA. It means that, with this dataset, feature set, and evaluation procedure, the additional variables did not provide better out-of-sample predictions. The low average R² values also indicate that the models explain only a limited part of GPA variation; predictive performance should therefore be described as modest rather than strong.

## 6. Academic-only vs Full Model Comparison — Method

`src/compare_feature_sets.py` evaluates three regressors using identical 80/20 train-test splits across 30 reproducible repetitions. The Academic-only set uses age, study time, and attendance; the Full Model adds social-media time, sleep disturbance, mood modification, anxiety, depression, self-esteem, and academic social-media distraction. It writes run-level results to `data/processed/feature_set_comparison.csv` and aggregate results to `data/processed/feature_set_comparison_summary.csv`. The aggregate comparison table is committed in `data/processed/feature_set_comparison_summary.csv`. The full run-level comparison and regenerated benchmark figures are available in the successful GitHub Actions artifact linked below. This is a predictive comparison, not a causal analysis.

Run it with:

```bash
python src/compare_feature_sets.py
```

## 7. Quickstart & Reproducibility Guide

```bash
# 1. Install exact pinned dependencies
pip install -r requirements.txt

# 2. Run benchmark on the full 256-row real survey dataset (or use --sample for the 10-row smoke test)
python src/benchmark.py --config configs/config.yaml --full

# Compare academic-only and full feature sets
python src/compare_feature_sets.py

# 3. Filter warm-up runs and compute 1,000-iteration bootstrap statistics (including p95 & p99)
python src/process_data.py --config configs/config.yaml

# 4. Generate publication-ready PDF and 300 DPI PNG figures in reports/figures/
python src/visualize.py --config configs/config.yaml

# 5. Run the benchmark helper unit tests
python -m unittest test_benchmark -v
```


## Scientific Visualizations

The corrected experiment regenerates both figures in PDF and 300-DPI PNG formats. Download the **reproducible-experiment-results** artifact from the successful [GitHub Actions run](https://github.com/EraMiir/fast-dopamine-academic-performance/actions/runs/42986932908) to view the figures produced by the corrected benchmark. The previously committed images are not presented here as current results.

<!-- FEATURE_GPA_ANALYSIS_START -->
## 8. Exploratory Feature–GPA Relationship Analysis

Spearman rank correlation was used to examine monotonic associations between selected survey factors and GPA. Pearson correlation is also reported as a supplementary measure. 95% confidence intervals for Spearman's rho were estimated using 2,000 bootstrap resamples (random seed = 42).

| Factor | N | Spearman's rho | 95% bootstrap CI | p-value | Pearson's r | Direction |
|---|---:|---:|---:|---:|---:|---|
| Study time (hours) | 256 | 0.302 | [0.192, 0.409] | <0.0001 | 0.280 | Positive |
| Attendance rate | 256 | 0.235 | [0.119, 0.349] | 0.0001 | 0.282 | Positive |
| Academic social media distraction | 256 | -0.175 | [-0.298, -0.052] | 0.0051 | -0.154 | Negative |
| Sleep disturbance | 256 | -0.090 | [-0.213, 0.039] | 0.1511 | -0.090 | Weak / near zero |
| Social media time (hours) | 256 | -0.046 | [-0.161, 0.069] | 0.4599 | -0.056 | Weak / near zero |

**Interpretation:** Positive coefficients indicate that higher factor values tend to occur with higher GPA; negative coefficients indicate that higher factor values tend to occur with lower GPA. Coefficients near zero indicate weak monotonic association.

**Important limitation:** These are observational, pairwise correlations, not evidence of causation. They do not test nonlinear interactions and do not control simultaneously for study time and attendance. Statistical significance should be interpreted cautiously because several factors are examined.

Full results: `data/processed/feature_gpa_correlations.csv`.
<!-- FEATURE_GPA_ANALYSIS_END -->
