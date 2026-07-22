# XGBoost Mastery: From Theoretical Foundations to High-Throughput Production

**Author**: Marshal Mathew | Senior Manager & Lead Data Scientist (7+ Years Banking ML)  
**Repository**: [C:\Users\marsh\agy2-projects\xgboost](file:///C:/Users/marsh/agy2-projects/xgboost)  

---

## Table of Contents
1. [Module 01: Theoretical Foundations (CART, Bagging, Boosting, GBM)](#module-01-theoretical-foundations)
2. [Module 02: XGBoost Core Mechanics (Taylor Expansion, Gradients, Hessians, Split Gain)](#module-02-xgboost-core-mechanics)
3. [Module 03: Basic Usage & Tuning (GridSearch vs. Optuna Bayesian Optimization)](#module-03-basic-usage--tuning)
4. [Module 04: Advanced Features & Diagnostics (SHAP Interpretability, Monotonicity, Feature Importance)](#module-04-advanced-features--diagnostics)
5. [Module 05: Production Deployment & Serving (JSON/UBJ Serialization, ONNX Runtime Benchmarking)](#module-05-production-deployment--serving)
6. [Module 06: Banking & Finance Projects (Credit Scoring, AML Mule Detection, Fraud)](#module-06-banking--finance-projects)
7. [Module 07: Distributed XGBoost Architecture (PySpark, Dask, Ray Scaling)](#module-07-distributed-xgboost-architecture)

---

<a id="module-01-theoretical-foundations"></a>
## Module 01: Theoretical Foundations

Before diving into Extreme Gradient Boosting (XGBoost), it is crucial to understand the foundational algorithms it builds upon: **Decision Trees** and **Ensemble Learning (Bagging vs. Boosting)**.

### 1. Decision Trees (CART)
XGBoost is an ensemble of Classification and Regression Trees (CART). Unlike standard decision trees that output categorical classes (like "Fraud" or "Not Fraud"), CARTs contain real-valued scores in each of their leaves, regardless of whether the task is classification or regression.

#### Anatomy of a Decision Tree
- **Root Node**: The entire dataset.
- **Internal Nodes**: Splitting rules (e.g., `Age > 30`).
- **Leaf Nodes**: The final output values (scores).

#### Splitting Criteria
Trees learn by finding splits that maximize information gain (or minimize impurity):
- **Gini Impurity** (Classification):
  $$\text{Gini} = 1 - \sum (p_i)^2$$
- **Mean Squared Error (MSE)** (Regression):
  $$\text{MSE} = \frac{1}{N} \sum (y_i - \bar{y})^2$$

### 2. Ensemble Learning: The Wisdom of the Crowd
A single decision tree is prone to **overfitting** (high variance, low bias). Ensemble methods combine multiple weak learners to create a strong learner.

- **Bagging (Bootstrap Aggregating)**: Train multiple independent trees in *parallel* on random subsets of the data with replacement. Reduces variance (e.g., Random Forest).
- **Boosting**: Train trees *sequentially*. Each new tree tries to correct the errors (residuals) made by previous trees. Reduces bias and variance with regularization.

### 3. Gradient Boosting Machines (GBM)
GBM fits new models to the **residuals** (the errors) of previous models:
1. Predict target $Y$ with baseline model $F_1(x)$ (e.g., mean).
2. Calculate residual error: $h_1(x) = Y - F_1(x)$.
3. Train new tree $T_1(x)$ to predict residual $h_1(x)$.
4. Update model: $F_2(x) = F_1(x) + \eta T_1(x)$, where $\eta$ is learning rate.

---

<a id="module-02-xgboost-core-mechanics"></a>
## Module 02: XGBoost Core Mechanics

XGBoost optimizes a regularized objective function using a 2nd-order Taylor expansion.

### 1. The Regularized Objective Function
$$\mathcal{O}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \Omega(f_t)$$
Where the regularization penalty $\Omega(f_t)$ is defined as:
$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|$$
- $\gamma$: Complexity penalty per added leaf node (controls pruning).
- $\lambda$: L2 regularization penalty on leaf weights.
- $\alpha$: L1 regularization penalty on leaf weights.

### 2. 2nd-Order Taylor Expansion (Gradients & Hessians)
Using Taylor expansion up to the 2nd order around $\hat{y}_i^{(t-1)}$:
$$\mathcal{O}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
Where:
- First Derivative (Gradient): $g_i = \frac{\partial l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}$
- Second Derivative (Hessian): $h_i = \frac{\partial^2 l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)2}}$

### 3. Optimal Leaf Weight & Split Gain
The optimal weight $w_j^*$ for leaf $j$ containing sample set $I_j$ is solved analytically:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

The Loss Reduction (Gain) evaluated for splitting a node into Left ($L$) and Right ($R$) children is:
$$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$
- **Pruning Rule**: A split is kept ONLY if $\text{Gain} > 0$. $\gamma$ acts as the threshold barrier for tree expansion!

---

<a id="module-03-basic-usage--tuning"></a>
## Module 03: Basic Usage & Tuning

### Step 3A: Baseline Tuning via `GridSearchCV`
```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("GridSearch Best Params:", grid_search.best_params_)
```

### Step 3B: Advanced Tuning via `Optuna` (Bayesian Optimization)
```python
import optuna
import xgboost as xgb

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True)
    }
    
    cv_res = xgb.cv(params, dtrain, nfold=3, num_boost_round=150, early_stopping_rounds=15)
    return cv_res['test-logloss-mean'].min()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
print("Optuna Best Parameters:", study.best_trial.params)
```

---

<a id="module-04-advanced-features--diagnostics"></a>
## Module 04: Advanced Features & Diagnostics

### 1. Monotonic Constraints
Enforces domain business rules (e.g. Higher Income $\rightarrow$ Lower Credit Risk):
```python
params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'monotone_constraints': '(-1, 0, 0)' # Income (-1: decreasing), Age (0: unconstrained)
}
```

### 2. SHAP (SHapley Additive exPlanations) Interpretability
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global Summary Plot
shap.summary_plot(shap_values, X_test)

# Local Waterfall Prediction Explanation
shap.plots.waterfall(explainer(X_test)[0])
```

---

<a id="module-05-production-deployment--serving"></a>
## Module 05: Production Deployment & Serving

### Step 5A: Native Model Serialization (JSON & UBJ)
```python
# Save as Universal Binary JSON
model.save_model("model.ubj")

# Load model
booster = xgb.Booster()
booster.load_model("model.ubj")
```

### Step 5B: ONNX Runtime Conversion & Benchmark (`onnxruntime`)
```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# Convert to ONNX format
initial_type = [('float_input', FloatTensorType([None, X_test.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Ultra-fast C++ ONNX Runtime Inference
sess = rt.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
preds = sess.run(None, {"float_input": X_test.astype(np.float32)})
```

---

<a id="module-06-banking--finance-projects"></a>
## Module 06: Banking & Finance Projects

### Credit Risk Scorecard Implementation (`06_projects_finance/03_credit_scoring/credit_scoring.py`)
- Incorporates CIBIL score features (`External_Cibil_Dataset.xlsx`).
- Enforces strict monotonic constraints on financial ratios to meet regulatory audit rules.
- Computes WoE (Weight of Evidence) and PSI (Population Stability Index).

---

<a id="module-07-distributed-xgboost-architecture"></a>
## Module 07: Distributed XGBoost Architecture

For multi-gigabyte/terabyte datasets:
- **PySpark Integration**: `xgboost.spark.SparkXGBClassifier` distributes training across Spark clusters.
- **Dask Integration**: `dask_xgboost` handles out-of-core data frames on local or cloud worker pools.
