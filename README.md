# Predicting Student Academic Performance (GPA/CGPA) Based on Social Media Usage Patterns and Psychometric Indicators

**Authors:** [Your Name / Partner's Name]  
**Course / Assignment:** Task 2 — Methodology Passport & Benchmarking Repository

---

## 1. Research Overview & Questions

This repository contains the reproducible benchmarking pipeline for evaluating machine learning models predicting student academic performance (`Last Academic Result (GPA/CGPA)`) from behavioral social media metrics and psychological scales ($N = 256$ survey responses).

* **RQ1:** To what extent do non-linear ensemble machine learning models (`RandomForestRegressor`, `GradientBoostingRegressor`) reduce the prediction error ($RMSE$, $MAE$) of student GPA compared to a linear baseline (`Ridge Regression`) when incorporating social media distraction and psychometric indicators?
* **RQ2:** How do non-linear interactions between late-night social media usage, academic distraction, and sleep disturbance impact GPA prediction accuracy while controlling for study time and attendance rate?

---

## 2. Repository Directory Structure

```text
├── configs/
│   └── config.yaml               # Experiment hyperparameters, feature lists, and guardrails
├── data/
│   ├── sample/
│   │   └── sample_survey.csv     # 10-row micro-sample for pipeline smoke testing
│   └── cleaned_survey.csv        # Full preprocessed dataset (N = 256, 24 features)
├── docs/
│   └── README.md                 # Methodology passport and research documentation
├── notebooks/
│   └── README.md                 # Exploratory Data Analysis (EDA) and correlation plots
├── src/
│   └── benchmark.py              # Executable benchmarking script (RMSE, MAE, Latency, RAM)
├── .gitignore                    # Python, Jupyter, and temporary artifact exclusions
├── Dockerfile                    # Reproducible environment with exact dependency pins
├── LICENSE                       # MIT Open Source License
└── README.md                     # Project documentation and Quickstart guide
```

### Run via Docker (100% Reproducible)
docker build -t gpa-benchmark .
docker run --rm gpa-benchmark

### Run Locally with Python
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 pyyaml==6.0.1 psutil==5.9.8
python src/benchmark.py --config configs/config.yaml
