# 06 - Capstone Projects: Finance Focus

To cement your authority on XGBoost, you will apply the theory and mechanics to real-world financial problems. These projects will challenge your ability to handle data, tune hyperparameters, interpret results with SHAP, and apply production quirks.

## Project 1: Stock Prediction / Financial Forecasting
- **Goal**: Predict the future movement of a stock (Classification) or the future price (Regression).
- **Data Source**: Programmatically downloaded via `yfinance`.
- **Focus**: Time-series validation (no standard K-Fold CV, must use time-based splitting) and handling noisy, non-stationary data.

## Project 2: Fraud Detection
- **Goal**: Identify fraudulent credit card/bank transactions.
- **Data Sources**: 
  - Synthetic Imbalanced Data (used in notebook)
  - [Bank Account Fraud Dataset NeurIPS 2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) (Highly recommended real-world dataset)
  - [IEEE Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Classic Kaggle competition with complex relational data)
- **Focus**: Extreme class imbalance (0.1% positives). We will focus on `scale_pos_weight`, customizing the evaluation metric (AUC-PR), and explaining predictions to non-technical stakeholders (SHAP).

## Project 3: Credit Scoring (Loan Default)
- **Goal**: Predict the probability of a customer defaulting on a loan or assess credit risk.
- **Data Sources**: 
  - Simulated Credit Data (used in notebook)
  - [Leading Indian Bank and CIBIL Real World Dataset](https://www.kaggle.com/datasets/saurabhbadole/leading-indian-bank-and-cibil-real-world-dataset)
- **Focus**: Monotonic constraints (e.g., higher income should strictly decrease default probability) and handling high-cardinality categorical features.

## Project 4: Marketing Propensity
- **Goal**: Predict whether a customer will subscribe to a term deposit or respond to a marketing campaign.
- **Data Source**: [Banking Dataset - Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
- **Focus**: Maximizing recall/precision tradeoffs for business ROI, identifying key drivers of customer conversion using SHAP.

## Project 5: Massive Data Scaling
- **Goal**: Push the limits of a single machine or practice distributed computing on a massive dataset.
- **Data Source**: [Massive Bank Dataset - 1 Million Rows](https://www.kaggle.com/datasets/ksabishek/massive-bank-dataset-1-million-rows)
- **Focus**: Memory management with DMatrix/Device DMatrix, utilizing `tree_method='hist'` or `gpu_hist` for ultra-fast training, and hyperparameter tuning at scale.
