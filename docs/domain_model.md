# Domain Model & Mathematical Formalization

## 1. Machine Learning Mathematical Formulation (LaTeX)

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ denote the student survey dataset ($N = 256$), where $\mathbf{x}_i \in \mathbb{R}^{d}$ ($d = 10$) represents the behavioral social media ("fast dopamine") and psychometric feature vector, and $y_i \in [0.0, 4.0]$ represents the target student academic performance (`Last Academic Result (GPA/CGPA)`).

### 1.1. Loss Functions ($\mathcal{L}$)

1. **Baseline Ridge Regression ($L_2$-Regularized Objective):**
   $$\mathcal{L}_{\text{Ridge}}(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - (\mathbf{w}^\top \mathbf{x}_i + b) \right)^2 + \alpha \|\mathbf{w}\|_2^2$$

2. **Gradient Boosting Regressor (Additive Stagewise Optimization):**
   $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x}), \quad \mathcal{L}_{\text{GBR}}(y_i, F(\mathbf{x}_i)) = \frac{1}{2}\left(y_i - F(\mathbf{x}_i)\right)^2$$

3. **Random Forest Ensemble Predictor:**
   $$\hat{y}_i = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x}_i; \Theta_b)$$

### 1.2. Target Evaluation Metrics & Bootstrap Confidence Intervals

* **Primary Metric — Root Mean Squared Error ($\text{RMSE}$):**
  $$\text{RMSE} = \sqrt{\frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} (y_i - \hat{y}_i)^2}$$

* **Secondary Metric — Mean Absolute Error ($\text{MAE}$):**
  $$\text{MAE} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} |y_i - \hat{y}_i|$$

* **Non-parametric Bootstrap 95% Confidence Interval ($B_{\text{boot}} = 1000$):**
  $$\text{CI}_{95\%}(\hat{\theta}) = \left[ Q_{0.025}(\hat{\theta}^*_1, \dots, \hat{\theta}^*_{1000}), \; Q_{0.975}(\hat{\theta}^*_1, \dots, \hat{\theta}^*_{1000}) \right]$$

---

## 2. Mathematical Notation Table

| Symbol | Domain / Units | Description |
| :--- | :---: | :--- |
| $\mathcal{D}$ | $\mathbb{R}^{N \times (d+1)}$ | Preprocessed student survey dataset ($N = 256$ students) |
| $\mathbf{x}_i$ | $\mathbb{R}^{10}$ | Input feature vector (Study Time, Attendance, Social Media Hours, Distraction, Sleep, Anxiety, Depression, Mood, Self-esteem, Age) |
| $y_i, \hat{y}_i$ | $[0.0, 4.0]$ GPA | Ground-truth and predicted cumulative Grade Point Average (GPA/CGPA) |
| $\alpha$ | $\mathbb{R}^+$ | Ridge $L_2$ regularization penalty parameter ($\alpha = 1.0$) |
| $B, \eta$ | $\mathbb{N}, (0, 1]$ | Number of ensemble trees ($B = 100$) and boosting learning rate ($\eta = 0.05$) |
| $T_{\text{lat}}$ | $\text{ms}$ | End-to-end training and inference execution latency |
| $M_{\text{RAM}}$ | $\text{MB}$ | Peak Resident Set Size (RSS) memory footprint |

---

## 3. System Architecture & Metric Collection Pipeline (C4 Component Diagram)

```mermaid
flowchart LR
    subgraph DataLayer ["1. Data Layer"]
        RawCSV["cleaned_survey.csv\n(N = 256)"]
        RawJSONL["data/raw/test_run.jsonl\n(Warm-up + Evaluation Logs)"]
    end

    subgraph BenchmarkEngine ["2. Benchmarking & Telemetry (src/)"]
        Runner["benchmark.py\n(Model Training & Inference)"]
        Probe["Telemetry Probe\n[Metric Node: Latency (ms), RAM (MB), RMSE, MAE]"]
        Processor["process_data.py\n(Warm-up Filter & 1000-iter Bootstrap CI)"]
    end

    subgraph ReportingLayer ["3. Artifacts & Visualization"]
        SummaryCSV["data/processed/metrics_summary.csv"]
        Visualizer["visualize.py\n(Colorblind-safe PDF & 300 DPI PNG)"]
    end

    RawCSV --> Runner
    Runner --> Probe
    Probe --> RawJSONL
    RawJSONL --> Processor
    Processor --> SummaryCSV
    SummaryCSV --> Visualizer
