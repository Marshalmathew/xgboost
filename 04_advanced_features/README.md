# 04 - Advanced Features & Diagnostics

Mastery of XGBoost requires knowing how to evaluate it, interpret it, and hook into its training lifecycle.

## 1. Early Stopping and Cross Validation
Instead of blindly guessing `num_boost_round`, we can tell XGBoost to stop training if the validation metric doesn't improve for `N` consecutive rounds. 
- Use `xgb.cv()` for robust evaluation. It returns a Pandas DataFrame with the evaluation history.

## 2. Feature Importance vs. SHAP
XGBoost provides built-in feature importance, but they can be misleading:
- **Weight**: Number of times a feature is used to split data across all trees. (Biased towards continuous features).
- **Gain**: The average gain of splits which use the feature. (Most reliable native metric).
- **Cover**: The average coverage (number of instances) of splits which use the feature.

**The Authority approach: SHAP (SHapley Additive exPlanations)**
SHAP values provide consistent, game-theoretic feature attributions. It explains *why* a specific prediction was made, and aggregate SHAP values give global feature importance without the biases of native metrics.

## 3. Custom Objective and Evaluation Functions
If standard Log Loss or MSE isn't what your business cares about, you can pass custom functions to XGBoost.
- **Custom Objective**: Must return the First (Gradient) and Second (Hessian) order derivatives of your loss function.
- **Custom Metric**: Must return a single float value representing the score.
