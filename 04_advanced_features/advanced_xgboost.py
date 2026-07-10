# %% [markdown]
# # Advanced Features and Diagnostics
# 
# In this notebook, we will explore Feature Importance, SHAP values for interpretability, and Cross-Validation (Early Stopping).

# %%
import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## 1. Data Prep and Training
# We'll use the California Housing dataset for regression.

# %%
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'eta': 0.1,
    'max_depth': 4
}
bst = xgb.train(params, dtrain, num_boost_round=100)

# %% [markdown]
# ## 2. Native Feature Importance
# XGBoost provides three types of native feature importance. Notice how they can contradict each other!

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

xgb.plot_importance(bst, importance_type='weight', ax=axes[0], title='Weight (Frequency)')
xgb.plot_importance(bst, importance_type='gain', ax=axes[1], title='Gain (Loss Reduction)')
xgb.plot_importance(bst, importance_type='cover', ax=axes[2], title='Cover (Instances)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. The Authority Method: SHAP
# SHAP (SHapley Additive exPlanations) is the gold standard for tree interpretability. It breaks down individual predictions and provides consistent global importance.

# %%
# Initialize JS visualization for SHAP
shap.initjs()

# Create a Tree Explainer for our XGBoost model
explainer = shap.TreeExplainer(bst)

# Calculate SHAP values for the test set
# (We pass the raw pandas dataframe, not the DMatrix)
shap_values = explainer.shap_values(X_test)

# %% [markdown]
# ### Global Importance (Summary Plot)
# This plot replaces native feature importance. It shows the magnitude of impact and the direction (e.g., higher MedInc -> higher SHAP value -> higher house price).

# %%
shap.summary_plot(shap_values, X_test)

# %% [markdown]
# ### Local Importance (Explaining a single prediction)
# Let's explain why the 1st house in the test set got its specific prediction.

# %%
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True)

# %% [markdown]
# ## 4. Cross Validation (xgb.cv)
# When tuning, we should use cross-validation rather than a single validation set to prevent overfitting the validation set.

# %%
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=500,
    nfold=5,
    early_stopping_rounds=20,
    metrics='rmse',
    as_pandas=True
)

print(f"Optimal number of trees: {cv_results.shape[0]}")
print(f"Best CV RMSE: {cv_results['test-rmse-mean'].iloc[-1]:.4f}")

cv_results[['train-rmse-mean', 'test-rmse-mean']].plot()
plt.title('CV Learning Curve')
plt.ylabel('RMSE')
plt.xlabel('Boosting Round')
plt.show()
