# XGBoost Mastery: From Foundations to Distributed Systems

A comprehensive, end-to-end guide and project repository for mastering XGBoost. This repository covers everything from the theoretical foundations and core mathematical mechanics to advanced tuning, production deployment, and distributed training. It also includes a dedicated section for real-world finance projects using Kaggle datasets.

## 🗂️ Repository Structure

The repository is organized progressively into the following sections:

- **`01_theory_foundations/`**: The fundamental math and theory behind decision trees, ensemble learning, and gradient boosting.
- **`02_xgboost_core_mechanics/`**: A deep dive into XGBoost's specific implementation, covering Taylor expansions, custom loss/objective functions, and core algorithms.
- **`03_basic_usage_and_tuning/`**: Practical guides on how to train, evaluate, and fine-tune XGBoost models (hyperparameter optimization, cross-validation).
- **`04_advanced_features/`**: Leveraging advanced capabilities such as monotonicity constraints, feature interactions, missing data handling, and custom evaluation metrics.
- **`05_production_and_quirks/`**: Best practices for model serialization (JSON/PMML/ONNX), deployment strategies, and understanding the specific quirks and edge cases of the XGBoost library.
- **`06_projects_finance/`**: Real-world, hands-on financial machine learning projects. Includes fraud detection, credit scoring, and marketing propensity modeling using large-scale banking datasets.
- **`07_distributed_xgboost/`**: Scaling XGBoost training for massive datasets using distributed frameworks.

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
