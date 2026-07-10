# %% [markdown]
# # Project 3: Credit Scoring (Loan Default)
# 
# In this project, we predict whether someone will default on their loan. We will use native categorical features and monotonic constraints to make the model robust and explainable to regulators.

# %%
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Simulating Credit Data
# We'll create a dataset with Income, Age, and a categorical 'Employment_Type'.
# Higher income = Lower Default.

# %%
np.random.seed(42)
n_samples = 10000

income = np.random.uniform(20000, 150000, n_samples)
age = np.random.uniform(18, 70, n_samples)
employment_types = ['Salaried', 'Self-Employed', 'Unemployed', 'Student']
employment = np.random.choice(employment_types, n_samples)

# Generate target (Default = 1)
# Higher income -> lower probability
# Unemployed -> higher probability
base_risk = 0.5 - (income / 300000)
base_risk = np.clip(base_risk, 0.05, 0.95)

# Add employment effect
emp_effect = {'Salaried': 0, 'Self-Employed': 0.1, 'Unemployed': 0.4, 'Student': 0.2}
risk = base_risk + np.array([emp_effect[e] for e in employment])
risk = np.clip(risk, 0, 1)

y = np.random.binomial(1, risk)

df = pd.DataFrame({
    'Income': income,
    'Age': age,
    'Employment': employment
})

df['Employment'] = df['Employment'].astype('category')

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

# %% [markdown]
# ## 2. Training with Monotonicity and Categoricals
# We enforce that Income MUST be monotonically decreasing (-1) with respect to default risk. Age is unconstrained (0). Employment is categorical (cannot have monotonic constraints).

# %%
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist', # Required for categoricals
    'max_depth': 4,
    'eta': 0.1,
    'monotone_constraints': '(-1, 0, 0)' # (Income, Age, Employment) - Note: index matches column order
}

bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], verbose_eval=20)

# %% [markdown]
# ## 3. Evaluation

# %%
preds = bst.predict(dtest)
print(f"ROC-AUC: {roc_auc_score(y_test, preds):.4f}")

# Plot a single tree to verify categorical splits
xgb.plot_tree(bst, num_trees=0)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()
