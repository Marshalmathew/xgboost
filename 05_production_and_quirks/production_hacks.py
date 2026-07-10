# %% [markdown]
# # Production Quirks and Hacks
# 
# Learn how to enforce Monotonic Constraints, handle imbalanced data, and natively use categorical features.

# %%
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Monotonic Constraints
# Imagine a scenario where you are predicting car insurance risk based on the age of the driver. Domain knowledge says that generally, older drivers are less risky up to a certain point. Let's create a synthetic dataset where risk decreases with age, but with lots of noise.

# %%
np.random.seed(42)
ages = np.random.uniform(18, 80, 500)
# Risk goes down as age goes up, plus noise
risk = -2.0 * ages + np.random.normal(0, 30, 500)

X = pd.DataFrame({'Age': ages})
y = risk

dtrain = xgb.DMatrix(X, label=y)

# %% [markdown]
# ### Training Without Constraints
# The model will overfit the noise and produce a jagged, non-monotonic curve.

# %%
params_standard = {'objective': 'reg:squarederror', 'max_depth': 3, 'eta': 0.1}
bst_standard = xgb.train(params_standard, dtrain, num_boost_round=50)

# %% [markdown]
# ### Training With Constraints
# We enforce that as Age increases, the prediction must strictly DECREASE (-1). 
# (Use 1 for strictly INCREASING).

# %%
params_constrained = {
    'objective': 'reg:squarederror', 
    'max_depth': 3, 
    'eta': 0.1,
    'monotone_constraints': '(-1)' # Tuple/string corresponding to feature columns
}
bst_constrained = xgb.train(params_constrained, dtrain, num_boost_round=50)

# %% [markdown]
# ### Visualization

# %%
X_plot = pd.DataFrame({'Age': np.linspace(18, 80, 100)})
dplot = xgb.DMatrix(X_plot)

plt.scatter(ages, risk, alpha=0.3, label='Data (Noisy)')
plt.plot(X_plot['Age'], bst_standard.predict(dplot), color='red', label='Standard (Overfit)')
plt.plot(X_plot['Age'], bst_constrained.predict(dplot), color='green', linewidth=3, label='Monotonic (Robust)')
plt.legend()
plt.title("Monotonic Constraints in XGBoost")
plt.show()

# %% [markdown]
# ## 2. Native Categorical Features (XGBoost >= 1.5)
# One-Hot Encoding inflates tree depth. XGBoost now supports categorical splits natively.

# %%
# Synthetic Data
df = pd.DataFrame({
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'London', 'Paris', 'New York'],
    'Value': [100, 200, 150, 300, 210, 140, 110]
})

# Crucial Step: Convert to Pandas category type!
df['City'] = df['City'].astype('category')

X_cat = df[['City']]
y_cat = df['Value']

# Enable categorical support
dtrain_cat = xgb.DMatrix(X_cat, label=y_cat, enable_categorical=True)

params_cat = {
    'tree_method': 'hist', # MUST use 'hist' or 'gpu_hist' for categoricals
    'max_depth': 2
}
bst_cat = xgb.train(params_cat, dtrain_cat, num_boost_round=10)

# See how it splits internally
xgb.plot_tree(bst_cat, num_trees=0)
plt.show()

# %% [markdown]
# ## 3. Saving and Loading Models
# Use `.json` or `.ubj` for safe, cross-platform model serialization.

# %%
# Save
bst_cat.save_model("my_model.json")

# Load
loaded_bst = xgb.Booster()
loaded_bst.load_model("my_model.json")
print("Model loaded successfully!")
