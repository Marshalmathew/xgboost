# XGBoost Mastery: From Foundations to Distributed Systems

A comprehensive, end-to-end guide and project repository for mastering XGBoost. This repository covers everything from the theoretical foundations and core mathematical mechanics to advanced tuning, production deployment, and distributed training. It also includes a dedicated suite of enterprise financial machine learning projects, regulatory compliance frameworks (SR 11-7 / ECOA), and distributed architectures.

---

## 🗺️ Master Curriculum & Documentation Navigation

This repository is organized into a **3-Tier Documentation Standard**:

1. **[Master Curriculum Roadmap (`CURRICULUM.md`)](./CURRICULUM.md)**: Master syllabus, prerequisite flowchart, learning progression, and the **5 Lethal Enterprise Production Footguns**.
2. **[Enterprise XGBoost Playbook (`XGBoost_Playbook.md`)](./XGBoost_Playbook.md)**: Institutional governance standard with three production proposals ready for Model Risk / Architecture sign-off.
3. **[Comprehensive Study Guide (`XGBoost_Mastery_Study_Guide.md`)](./XGBoost_Mastery_Study_Guide.md)**: Monolithic 2,000-line textbook with complete mathematical derivations (exportable to HTML/PDF via [`export_study_guide_pdf.py`](./export_study_guide_pdf.py)).

---

## 🗂️ Modular Curriculum Structure

The repository is organized progressively into the following sections:

- **[`01_theory_foundations/`](./01_theory_foundations/README.md)**: Fundamental math behind CART, bagging vs boosting, and Friedman GBM (from scratch).
- **[`02_xgboost_core_mechanics/`](./02_xgboost_core_mechanics/README.md)**: 2nd-order Taylor expansion, Newton-Raphson leaf weights, Gain formula, and Weighted Quantile Sketch.
- **[`03_basic_usage_and_tuning/`](./03_basic_usage_and_tuning/README.md)**: Hyperparameter science, TPE Bayesian optimization (Optuna), MedianPruner, and learning curve diagnostics.
- **[`04_advanced_features/`](./04_advanced_features/README.md)**: Monotonic constraints, interaction constraints (ECOA Reg B), TreeSHAP game theory, and **native Multi-Output Vector Trees**.
- **[`05_production_and_quirks/`](./05_production_and_quirks/README.md)**: Custom loss derivations (Asymmetric AML loss $k=10$), ONNX serving, probability calibration, and **Model Governance (PSI/CSI) & Fair Lending Compliance**.
- **[`06_projects_finance/`](./06_projects_finance/README.md)**: Real-world financial machine learning projects: fraud detection, credit scorecards, marketing propensity, massive scale data handling, and **Survival Analysis (AFT for Lifetime ECL)**.
- **[`07_distributed_xgboost/`](./07_distributed_xgboost/README.md)**: Scaling XGBoost training using the Rabit ring AllReduce topology across PySpark, Dask, and Ray.

---

## 🚀 Getting Started

### Prerequisites

This project uses `uv` for lightning-fast dependency management:

```bash
# Clone the repository
git clone https://github.com/Marshalmathew/xgboost.git
cd xgboost

# Install dependencies and sync virtual environment
uv sync
```

### Instant Data Setup: Synthetic vs Kaggle

You can start training immediately without external accounts:

1. **Option A: Instant Deterministic Synthetic Data (Zero Setup)**
   ```bash
   uv run python scripts/generate_synthetic_data.py --dataset all --output-dir data/
   ```

2. **Option B: Real-World Kaggle Competitions**
   - Place your `kaggle.json` API key in the repository root.
   - Run the automated downloader:
     ```bash
     uv run python download_kaggle.py
     ```

### Running the Test Suite

```bash
uv run pytest -v
```

## 📜 License
This project is licensed under the terms of the included [LICENSE](./LICENSE) file.

