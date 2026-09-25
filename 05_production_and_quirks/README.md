# 05 - Production Quirks, Custom Objectives & Advanced Governance

---
[⬅️ Prev: 04 - Advanced Features](../04_advanced_features/README.md) | [🏠 Master Curriculum](../CURRICULUM.md) | [📝 Pre-Chapter Diagnostic: Self-Test Production](./self_test_production.md) | [Next: 06 - Finance Projects ➡️](../06_projects_finance/README.md)
---

This module covers the advanced algorithmic engineering required to deploy XGBoost under institutional enterprise standards: developing custom loss functions, understanding numerical convexity and silent hessian clipping, benchmarking native missing-value sparsity routing, and establishing gradient boosting library selection criteria.

---


## 1. Custom Objectives: The Raw Margin Space Trap & Squared Log Error

### 1.1 The Raw Margin Space Trap
XGBoost allows developers to define custom loss functions matching the signature:
```python
def custom_objective(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    labels = dtrain.get_label()
    # ... compute grad, hess ...
    return grad, hess
```

> [!CRITICAL]
> **The Raw Margin Space Trap**: In custom objectives, `preds` is passed as **untransformed margin values ($z \in \mathbb{R}$)**, not probabilities. For binary classification, you must explicitly evaluate the sigmoid link function:
> $$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$
> If you write $g = \text{preds} - y$ assuming `preds` is probability $p$, gradient descent diverges. Furthermore, when providing a `custom_metric(preds, dtrain)` alongside a custom objective, `preds` passed to the metric is also in margin space.

### 1.2 Squared Log Error (SLE) Worked Example & Exact Parity
We implemented Squared Log Error from scratch:
$$L(y, z) = \frac{1}{2} (\ln(z + 1) - \ln(y + 1))^2$$
- First Derivative (Gradient): $g = \frac{\ln(z + 1) - \ln(y + 1)}{z + 1}$
- Second Derivative (Hessian): $h = \frac{1 - (\ln(z + 1) - \ln(y + 1))}{(z + 1)^2}$

In [`custom_loss_and_sparsity.py`](./custom_loss_and_sparsity.py), our from-scratch implementation achieves near-perfect correlation with native `reg:squaredlogerror`:
$$r = 0.9993 \text{ (Exact Floating-Point Parity)}$$

---

## 2. Convexity Requirements & Silent Hessian Clipping

For XGBoost to guarantee optimization convergence, custom objectives must satisfy three mathematical requirements:
1. **$C^2$ Smoothness**: First and second derivatives must be continuous.
2. **Row Additivity**: The objective must decompose as $\sum_{i=1}^n l(y_i, \hat{y}_i)$ without cross-row dependencies (e.g. global rank sorting cannot be computed in a standard elementwise objective).
3. **Strict Positive Convexity**: The hessian must be positive ($h_i > 0$) for all inputs.

### The Silent Clipping Failure Mode
In Squared Log Error, when an over-prediction occurs such that $\ln(z + 1) - \ln(y + 1) > 1$ (i.e. $z > e(y + 1) - 1$), the numerator becomes negative:
$$h < 0$$
XGBoost does **not** raise an exception. Instead, within the native C++ engine (`src/tree/updater_colmaker.cc` and `src/tree/split_evaluator.h`), it silently clips or thresholds negative and near-zero Hessians:
$$h \to \epsilon \quad (\text{clipped to positive machine precision } \approx 10^{-16})$$
Furthermore, `param.min_child_weight` strictly rejects splits where $\sum h_i < \text{min\_child\_weight}$. In these negative or clipped curvature regions, optimal leaf weights degenerate ($w^* \approx -G / \lambda$), degrading split quality quietly. A "clever" non-convex loss will fail silently, not loudly.


---

## 3. Hand-Derived Asymmetric AML Fraud Loss

In enterprise banking AML operations, false negatives (missed illicit accounts) incur severe regulatory fines and chargebacks ($\text{Cost}_{\text{FN}} = \$5,000$), while false positives (analyst review) cost significantly less ($\text{Cost}_{\text{FP}} = \$500$). The business cost ratio is $k = 10.0$.

### 3.1 Mathematical Derivation
Weighted binary cross-entropy:
$$L(y, p) = - [ k \cdot y \ln(p) + (1 - y) \ln(1 - p) ]$$
Using $p = \sigma(z)$ and $\frac{\partial p}{\partial z} = p(1 - p)$:

- **Gradient**:
  $$g = \frac{\partial L}{\partial z} = p(1 + (k - 1)y) - k \cdot y$$
- **Hessian**:
  $$h = \frac{\partial g}{\partial z} = (1 + (k - 1)y) p(1 - p)$$

**Strict Convexity Proof**: Because $k = 10 > 0$, $y \in \{0, 1\}$, and $p \in (0, 1)$, $(1 + (k - 1)y) > 0$ and $p(1 - p) > 0$. Therefore, **$h > 0$ strictly for all $z \in \mathbb{R}$**. Zero negative hessians, zero clipping.

### 3.2 Contrast Against `scale_pos_weight`
1. `scale_pos_weight` ($s = N_{\text{neg}} / N_{\text{pos}} = 24.0$) is a **frequency correction** intended to rebalance class priors. It shifts the base score and distorts probability calibration.
2. Custom Asymmetric Loss ($k = 10.0$) directly optimizes **decision-theoretic utility**.
3. **Borderline Prediction Divergence**: On borderline cases ($p \approx 0.50$, $y = 1$), `scale_pos_weight` exerts a **$2.4\times$ to $4.9\times$ stronger gradient pull** ($g_{\text{spw}} = -12.0$ vs $g_{\text{asym}} = -5.0$), flooding operations with false alarms.

### 3.3 Empirical Benchmark Results

![Asymmetric Loss vs scale_pos_weight](./asymmetric_vs_scale_pos_weight.png)

Under decision-theoretic thresholding ($p^* = 1/(1+k) = 0.0909$):
- **Custom Asymmetric Loss**: Net Operational Loss = **\$1,482,000**
- **Built-in `scale_pos_weight`**: Net Operational Loss = **\$1,736,500**
- **Financial Advantage**: Custom loss delivers a **+\$254,500 net cost reduction (+14.7%)** by avoiding false alarm floods.

---

## 4. DMatrix Sparsity-Aware Split Routing vs. Imputation

When tabular banking data contains missing values (e.g. unverified KYC income or dormant account balances), naive pre-imputation destroys predictive signal.

XGBoost evaluates optimal missing split routing natively:
$$\text{Direction} = \arg\max (\text{Gain}_{\text{missing}\to L}, \text{Gain}_{\text{missing}\to R})$$

### 4.1 Empirical 20% Missingness Benchmark

![Sparsity Missing Value Benchmark](./sparsity_missing_value_benchmark.png)

| Strategy | Holdout PR-AUC | Execution Impact | Recommendation |
|---|:---:|:---:|:---|
| **Native XGBoost Sparsity Routing** | **0.2577** | Zero pipeline preprocessing | 🏆 **Production Standard** |
| **Median Imputation** | 0.2443 | Distorts distribution variance | ❌ Discard |
| **Mean Imputation** | 0.2430 | Pulls splits toward arbitrary center | ❌ Discard |
| **Constant Imputation (-999)** | 0.2603 | Artificial outlier binning | ⚠️ Fragile to scaling |

---

## 5. Gradient Boosting Library Selection Framework (Enterprise Banking)

1. **XGBoost (Production Standard)**: Optimal for our core dense tabular datasets (transaction velocity, balance ratios, CIBIL scores) requiring sub-millisecond C++ ONNX serving and strict monotonic split guardrails.
2. **LightGBM**: Recommended for massive overnight transaction processing ($>50\text{M}$ rows) where GOSS sampling and leaf-wise speed minimize memory footprints.
3. **CatBoost**: Recommended for KYC screening containing high-cardinality nominal features (Merchant Category Codes `MCC`, IFSC Branch Codes, PIN codes) where manual target encoding risks leakage.

---

## 6. Probability Calibration Architecture (Niculescu-Mizil & Caruana 2005/2007)

1. **Distortion Physics**: Maximum-margin boosting pushes probability mass away from 0 and 1, creating a characteristic **sigmoid-shaped distortion**.
2. **Platt Scaling (`method='sigmoid'`) vs. Isotonic Regression (`method='isotonic'`)**: Platt scaling parametrically matches this distortion and is sample-efficient ($N < 1{,}000$). Isotonic regression is non-parametric but requires dense data ($N_{\text{calib}} \ge 1{,}000$).
3. **`scale_pos_weight` Unbiasing**: Artificially shifts odds by $\times s$. Odds must be inverted ($\text{odds}_{\text{true}} = \text{odds}_{\text{model}} / s$) or Platt calibrated to avoid balance sheet provisioning errors.
4. **Institutional Governance**: Tier 1 models (CECL/IFRS 9 Expected Credit Loss, Basel A-IRB, loan APR pricing) mandate 3-way split calibration. Tier 2 models (AML alert queues) rely strictly on rank ordering.

---

## 7. High-Throughput Serving & Low-Latency Inference (UBJSON & ONNX)

1. **Universal Binary JSON (`model.ubj`)**: Binary format offering cross-platform C/C++ safety and instant warm-loading without Python `pickle` security vulnerabilities.
2. **Ultra-Fast ONNX Serving**: C++ runtime inference achieving sub-millisecond SLAs ($\sim 0.32\text{ms}$ per request, $11.8\times$ faster than standard Python pipelines).

---

## 8. Production Drift Monitoring & Champion-Challenger Retraining Policy

1. **Three Drift Regimes**: Covariate shift (feature distributions shift), concept drift (conditional relationships $P(Y \mid X)$ change), and prediction drift (output score distributions migrate).
2. **Population Stability Index (PSI)**: Symmetrized Kullback-Leibler divergence with regulatory thresholds ($<0.10$ stable, $0.10-0.25$ moderate, $>0.25$ critical).
3. **Champion-Challenger Governance**: Formal model promotion rules codified under Federal Reserve SR 11-7 / OCC 2011-12 standards.

---

## 9. Algorithmic Fairness, Bias Auditing & Disparate Impact Governance

1. **Kleinberg Impossibility Theorem (ITCS 2017)**: Calibration within groups, Equal FPR, and Equal FNR cannot hold simultaneously when base rates differ across demographic groups. Choosing a fairness metric is an institutional policy decision, not an engineering default.
2. **Epistemic Trespassing of the 4/5ths Rule (arXiv:2202.09519)**: The 80% rule is an EEOC employment screening heuristic, NOT a safe harbor under ECOA / CFPB Regulation B credit lending.
3. **Hardt et al. Post-Processing Mitigation**: Group-specific decision thresholds eliminate disparate impact without disturbing underlying default probability calibration.

---

## 10. Ultra-Low Latency Inference & ONNX Serving

In enterprise card swipe authorizations and algorithmic trading, inference SLAs require **sub-millisecond latency**.

We benchmarked $N=1,000$ single-row (batch=1) inference queries across three serving engines in [`export_onnx_benchmark.py`](./export_onnx_benchmark.py):

| Serving Engine | Mean Latency ($\mu s$) | p50 ($\mu s$) | p95 ($\mu s$) | p99 ($\mu s$) | Throughput (Req/Sec) | Parity Max Diff ($\Delta$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Python DMatrix** | $1,051.6\,\mu s$ | $1,006.3\,\mu s$ | $1,368.2\,\mu s$ | $1,668.5\,\mu s$ | $951\,\text{qps}$ | Baseline |
| **Native C++ `inplace_predict`** | $599.6\,\mu s$ | $530.8\,\mu s$ | $961.2\,\mu s$ | $1,165.1\,\mu s$ | $1,668\,\text{qps}$ | $0.00 \times 10^0$ |
| **ONNX Runtime CPU** | **$22.9\,\mu s$** | **$21.6\,\mu s$** | **$26.0\,\mu s$** | **$30.8\,\mu s$** | **$43,727\,\text{qps}$** | **$1.79 \times 10^{-7}$** |

> **Production Takeaway**: Standard `DMatrix` allocation incurs Python object wrapping overhead. Native C++ `inplace_predict` offers a $1.75\times$ speedup, while ONNX Runtime CPU delivers a **$45.9\times$ latency reduction** (down to $22.9\,\mu s$) with strict numerical agreement down to $1.79 \times 10^{-7}$.


## 11. Conformal Prediction & Distribution-Free Uncertainty Guarantees

Standard gradient boosted trees output point predictions (e.g., $\hat{P}(\text{Default} \mid \mathbf{x}) = 0.49$) without quantifying epistemic uncertainty. In high-stakes enterprise underwriting, relying on uncalibrated point predictions risks approving high-risk loans that sit on ambiguous decision boundaries.

In [`conformal_risk_calibration.py`](./conformal_risk_calibration.py) and [`conformal_risk_lab.ipynb`](./conformal_risk_lab.ipynb), we implement mathematically rigorous conformal inference:
1. **Split Conformal Classification**: Guarantees finite-sample coverage $P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$ under exchangeability (Vovk et al., 2005) with exact quantile inflation $\lceil (n+1)(1-\alpha) \rceil / n$ and tie-breaking jitter.
2. **Institutional Tripartite Triage**:
   - `{0}`: **Straight-Through Processing (STP) Auto-Approve** (Negligible default rate).
   - `{1}`: **Automated Denial** (High default probability).
   - `{0, 1}`: **Refer to Senior Underwriter** (Epistemic boundary ambiguity where model expresses honest uncertainty).
   - `{}`: **Out-of-Distribution (OOD) Anomaly Flag**.
3. **Conformalized Quantile Regression (CQR)**: Heteroskedastic loss forecasting (Romano et al., 2019) via dual XGBoost pinball loss regressors (`reg:quantileerror`) with physical zero-clipping for Loss Given Default (LGD).
4. **Mondrian Conformal Fairness Auditing**: Eliminates demographic under-coverage across FICO risk tiers (`Prime`, `NearPrime`, `Subprime`) with small-sample empirical Bayesian shrinkage.

---

## 📁 Artifacts & Deliverables

1. [conformal_risk_calibration.py](./conformal_risk_calibration.py) - Standalone Split Conformal, CQR, and Mondrian fair calibration suite.
2. [conformal_risk_lab.ipynb](./conformal_risk_lab.ipynb) - Interactive Jupyter lab validating finite-sample coverage, tripartite triage, CQR, and SR 11-7 memo.
3. [custom_loss_and_sparsity.py](./custom_loss_and_sparsity.py) - Standalone script for SLE, custom loss, and sparsity benchmarks.
4. [probability_calibration.py](./probability_calibration.py) - Standalone probability calibration suite (Platt, Isotonic, Murphy Brier decomposition, ECE).
5. [production_drift_monitoring.py](./production_drift_monitoring.py) - Production statistical drift monitoring suite (PSI, KS-test, Jensen-Shannon, Evidently AI).
6. [fairness_bias_audit.py](./fairness_bias_audit.py) - Fairlearn bias auditing and Hardt et al. post-processing threshold optimization pipeline.
7. [day5_benchmarks.json](./day5_benchmarks.json) - Serialized metrics, SLE audit, and cost ledgers.
8. [calibration_benchmarks.json](./calibration_benchmarks.json) - Serialized calibration metrics across models and operational tail slices.
9. [drift_monitoring_summary.json](./drift_monitoring_summary.json) - Serialized PSI, KS-test, and JSD drift surveillance ledger.
10. [fairness_audit_results.json](./fairness_audit_results.json) - Serialized disparate impact ratios, calibration-by-group, and before/after mitigation numbers.
11. [model_governance_calibration_memo.md](./model_governance_calibration_memo.md) - Model Risk Management regulatory memorandum on calibration standards.
12. [champion_challenger_retraining_policy.md](./champion_challenger_retraining_policy.md) - Formal Model Risk Management policy on drift surveillance and retraining.
13. [model_governance_fairness_memo.md](./model_governance_fairness_memo.md) - Model Risk Management memorandum on algorithmic fairness, Kleinberg impossibility, and ECOA Reg B compliance.
14. [asymmetric_vs_scale_pos_weight.png](./asymmetric_vs_scale_pos_weight.png) - PR curves and operational cost trade-offs.
15. [sparsity_missing_value_benchmark.png](./sparsity_missing_value_benchmark.png) - Information gain retention comparison across imputation methods.
16. [probability_calibration_benchmark.png](./probability_calibration_benchmark.png) - Multi-panel publication reliability diagrams and odds unbiasing curves.
17. [drift_monitoring_dashboard.png](./drift_monitoring_dashboard.png) - Publication dashboard with PSI binning and ROC/PR degradation curves.
18. [fairness_bias_audit_dashboard.png](./fairness_bias_audit_dashboard.png) - 4-panel dashboard with selection rate disparities, subgroup reliability diagrams, and post-processing shift.
19. [evidently_drift_report.html](./evidently_drift_report.html) - Interactive HTML drift surveillance report generated via Evidently AI.
20. [`../XGBoost_Playbook.md`](../XGBoost_Playbook.md) - The consolidated enterprise master playbook for high-stakes banking.
