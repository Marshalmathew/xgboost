# Project 06: Survival Analysis & Accelerated Failure Time (AFT) for Credit Risk

In institutional lending and retail banking, standard binary classification predicts **whether** a borrower will default within a fixed time window (e.g. 12 months). However, loans have varying amortization terms (36 to 360 months), and predicting **when** a borrower defaults (Time-to-Default) with right-censored observation data is critical for computing **Lifetime Expected Credit Loss (IFRS 9 / CECL)**.

---

## 🎯 Core Concepts

1. **Right-Censored Loan Portfolios**:
   - For defaulted loans: the exact time of default $T$ is observed ($[t_i, t_i]$).
   - For active/performing loans: at observation date $t_{\text{obs}}$, default has not occurred. The true default time is right-censored in the interval $[t_{\text{obs}}, +\infty)$.
2. **Accelerated Failure Time (AFT) Formulation**:
   - Log-linear model: $\ln(T) = f(\mathbf{x}) + \sigma \epsilon$, where $\epsilon$ follows an extreme value, normal, or logistic distribution.
   - Objective: `survival:aft` in XGBoost with lower and upper bound bounding constraints (`label_lower_bound`, `label_upper_bound`).
3. **IFRS 9 / CECL Lifetime Loss Estimation**:
   - Integrating cumulative survival curves $S(t | \mathbf{x}) = P(T > t | \mathbf{x})$ with loan balance amortization curves over time.
4. **Concordance Index (C-Index)**:
   - Ranking discrimination evaluation on censored survival durations.

---

## 💻 Contents

- [survival_analysis_aft.ipynb](./survival_analysis_aft.ipynb): Complete end-to-end survival modeling pipeline in XGBoost using synthetic 5-year loan cohorts, interval bounds, AFT distribution comparisons, survival curve visualizations, and C-Index benchmarking.
