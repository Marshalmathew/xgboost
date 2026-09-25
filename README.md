# XGBoost Mastery: From Foundations to Distributed Systems

A comprehensive, end-to-end guide and project repository for mastering XGBoost. This repository covers everything from the theoretical foundations and core mathematical mechanics to advanced tuning, production deployment, and distributed training. It also includes a dedicated section for real-world finance projects using Kaggle datasets.

## 🗂️ Repository Structure

The repository is organized progressively into the following sections:

- **`01_theory_foundations/`**: The fundamental math and theory behind decision trees, ensemble learning, and gradient boosting.
- **`02_xgboost_core_mechanics/`**: A deep dive into XGBoost's specific implementation, covering Taylor expansions, custom loss/objective functions, and core algorithms.
- **`03_basic_usage_and_tuning/`**: Practical guides on how to train, evaluate, and fine-tune XGBoost models (hyperparameter optimization, cross-validation).
- **`04_advanced_features/`**: Leveraging advanced capabilities such as monotonicity constraints, feature interactions, SHAP interpretability, and **native Multi-Output Vector Trees**.
- **`05_production_and_quirks/`**: Best practices for model serialization (JSON/UBJ/ONNX), deployment strategies, and **Model Governance (PSI/CSI) & Fair Lending Compliance (Four-Fifths Rule)**.
- **`06_projects_finance/`**: Real-world financial machine learning projects, including fraud detection, credit risk scorecards, marketing propensity, massive scale data handling, and **Survival Analysis (Accelerated Failure Time / AFT for Lifetime ECL)**.
- **`07_distributed_xgboost/`**: Scaling XGBoost training for massive datasets using distributed frameworks (PySpark, Dask, Ray).

## 🚀 Getting Started

### Prerequisites

This project uses `uv` for dependency management and running scripts. Make sure you have it installed, or adapt the commands to your preferred Python package manager (like `pip` or `poetry`).

### Downloading Datasets (Kaggle)

The projects in `06_projects_finance` rely on several Kaggle datasets. We have provided a script to automate the download process.

1. Obtain your Kaggle API key (`kaggle.json`) from your Kaggle account settings.
2. Place the `kaggle.json` file in the root directory of this repository.
3. Run the download script:

```bash
python download_kaggle.py
```
*(Note: If you run into a `403 Forbidden` error, you may need to visit the specific Kaggle competition page and accept the rules before downloading the data via the API.)*

## 📜 License
This project is licensed under the terms of the included [LICENSE](./LICENSE) file.
