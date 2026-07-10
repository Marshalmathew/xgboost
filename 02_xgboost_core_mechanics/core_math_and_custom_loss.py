# %% [markdown]
# # Core Math and Custom Loss Functions in Python
# 
# In this notebook, we will implement the mathematics of XGBoost from scratch using pure Python/NumPy, and then hook a custom objective function into the actual XGBoost API.

# %%
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %% [markdown]
# ## 1. Simulating Data
# Let's create a simple regression dataset.

# %%
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# %% [markdown]
# ## 2. Standard Training (MSE)
# Let's train a standard model first to see the baseline performance.

# %%
params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1
}

bst_standard = xgb.train(params, dtrain, num_boost_round=50)
preds_standard = bst_standard.predict(dtest)
print(f"Standard MSE: {mean_squared_error(y_test, preds_standard):.4f}")

# %% [markdown]
# ## 3. Custom Objective Function
# The XGBoost objective function must return the **Gradient** (1st derivative) and **Hessian** (2nd derivative) of the loss function with respect to the predictions.
# 
# For Mean Squared Error (MSE), the loss is:
# $$L(y, \hat{y}) = \frac{1}{2} (y - \hat{y})^2$$
# 
# The Gradient is:
# $$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$
# 
# The Hessian is:
# $$\frac{\partial^2 L}{\partial \hat{y}^2} = 1$$

# %%
def custom_mse_objective(preds, dtrain):
    labels = dtrain.get_label()
    # Gradient: pred - label
    grad = preds - labels
    # Hessian: constant 1
    hess = np.ones(labels.shape)
    return grad, hess

# %% [markdown]
# ## 4. Custom Evaluation Metric
# We also need a custom metric to evaluate during training.

# %%
def custom_mse_metric(preds, dtrain):
    labels = dtrain.get_label()
    mse = np.mean((preds - labels)**2)
    return 'custom_mse', mse

# %% [markdown]
# ## 5. Training with Custom Objective
# Now we pass our custom functions to XGBoost. Note that we set `objective='reg:squarederror'` in params just as a placeholder (or 'reg:pseudohubererror', etc. but we override it via `obj=` parameter in `.train()`).
# Actually, it is better to omit the objective in params if we are overriding it, or set `disable_default_eval_metric=True`.

# %%
params_custom = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'disable_default_eval_metric': 1
}

# Train using custom objective and custom metric
bst_custom = xgb.train(
    params_custom, 
    dtrain, 
    num_boost_round=50, 
    obj=custom_mse_objective, 
    custom_metric=custom_mse_metric,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    verbose_eval=10
)

preds_custom = bst_custom.predict(dtest)
print(f"Custom Objective MSE: {mean_squared_error(y_test, preds_custom):.4f}")

# %% [markdown]
# ## Conclusion
# You have just implemented the core math of XGBoost! By defining the Gradient and Hessian, you can create loss functions for any business problem (e.g., asymmetric loss where under-predicting costs 10x more than over-predicting).
