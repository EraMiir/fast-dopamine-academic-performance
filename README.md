# Impact of "Fast Dopamine" Consumption Patterns on Academic Performance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

**Authors:** Sharap Alislam, Miir Yerassyl  
**Affiliation:** School of Digital Technologies, Narxoz University  

---

## Project Overview & Research Aim
This repository contains the research codebase and methodology configuration investigating the relationship between daily "fast dopamine" consumption triggers (screen time, short-form videos, gaming) and overall academic performance (GPA) of undergraduate students.

* **Research Question (RQ):** To what extent do daily screen time and video gaming frequency negatively correlate with university students' GPA?
* **Key Hypothesis ($H_1$):** High daily engagement in short-form media and gaming leads to a statistically significant decrease in academic performance ($p < 0.05$).

---

## Repository Structure
```text
fast-dopamine-academic-performance/
├── configs/
│   └── config.yaml          # Experiment parameters & feature definitions
├── data/
│   └── sample/
│       └── survey.csv       # Primary survey dataset for pipeline execution
├── src/
│   └── benchmark.py         # Executable validation and test script
├── .gitignore               # Excludes sensitive data, caches, and checkpoints
├── LICENSE                  # MIT Open Access License
├── requirements.txt         # Pinned dependency manifest
└── README.md                # Project documentation
```

Quickstart & Reproducibility Guide
# 1. Clone the repository
git clone [https://github.com/EraMiir/fast-dopamine-academic-performance.git](https://github.com/EraMiir/fast-dopamine-academic-performance.git)
cd fast-dopamine-academic-performance

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows use: venv\Scripts\activate
# On macOS/Linux use: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the benchmark script
python src/benchmark.py

Citation & License
Licensed under the MIT License.
