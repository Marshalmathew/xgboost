# 06 - Capstone Projects: Finance Focus

---
[⬅️ Prev: 05 - Production & Quirks](../05_production_and_quirks/README.md) | [🏠 Master Curriculum](../CURRICULUM.md) | [Next: 07 - Distributed XGBoost ➡️](../07_distributed_xgboost/README.md)
---

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

## Project 4: Marketing Propensity & Causal Uplift Decisioning
- **Goal**: Move beyond standard response propensity $P(Y=1 \mid X)$ to estimate Conditional Average Treatment Effects $\tau(X)$ and optimize capital allocation for retail banking campaigns.
- **Data Source**: Deterministic synthetic continuous CATE cohort (`scripts/generate_synthetic_data.py --dataset uplift`) and [Banking Dataset - Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets).
- **Deliverables**:
  - [`marketing_propensity.ipynb`](./04_marketing_propensity/marketing_propensity.ipynb): Classical predictive propensity modeling with Optuna Bayesian profit maximization.
  - [`causal_uplift_engine.py`](./04_marketing_propensity/causal_uplift_engine.py): Standalone production causal ML engine implementing S-Learner, T-Learner, and cross-fitted X-Learner (Künzel et al. PNAS 2019), Qini curves, AUUC, bootstrap permutation tests, and budget-constrained argmax policy optimization.
  - [`uplift_modeling.ipynb`](./04_marketing_propensity/uplift_modeling.ipynb): Interactive lab demonstrating the "Sure Things" / "Sleeping Dogs" propensity trap, Unconfoundedness identification assumption & hidden confounder audits, TreeSHAP model attribution caveats under SR 11-7, and campaign profit frontier optimization (+20% to +40% net dollar gain).

## Project 5: Massive Data Scaling
- **Goal**: Push the limits of a single machine or practice distributed computing on a massive dataset.
- **Data Source**: [Massive Bank Dataset - 1 Million Rows](https://www.kaggle.com/datasets/ksabishek/massive-bank-dataset-1-million-rows)
- **Focus**: Memory management with DMatrix/Device DMatrix, utilizing `tree_method='hist'` or `gpu_hist` for ultra-fast training, and hyperparameter tuning at scale.

## Project 6: Survival Analysis & AFT (Time-to-Default)
- **Goal**: Estimate continuous time-to-default on right-censored multi-year loan cohorts to compute IFRS 9 / CECL Lifetime Expected Credit Loss.
- **Directory**: `06_survival_credit_risk/`
- **Focus**: Accelerated Failure Time (`survival:aft`), interval-censored DMatrix bounds (`label_lower_bound`, `label_upper_bound`), parametric distribution comparison (`normal`, `logistic`, `extreme`), Harrell's Concordance Index (C-Index), and individual survival curves $S(t | \mathbf{x})$.

