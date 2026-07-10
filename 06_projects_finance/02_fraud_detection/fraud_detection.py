# %% [markdown]
# # Project 2: Fraud Detection
# 
# Fraud detection suffers from extreme class imbalance (e.g., 99.9% legitimate, 0.1% fraud). We will simulate this and show how to tune XGBoost to catch the minority class using `scale_pos_weight` and `AUC-PR`.

# %%
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, classification_report
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Simulating Highly Imbalanced Data

# %%
X, y = make_classification(n_samples=50000, n_features=20, n_informative=5, 
                           n_redundant=2, weights=[0.99, 0.01], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Calculate scale_pos_weight
num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
scale_weight = num_neg / num_pos
print(f"Negative/Positive Ratio: {scale_weight:.2f}")

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# %% [markdown]
# ## 2. Training with scale_pos_weight
# We will use `aucpr` (Area under Precision-Recall curve) as our metric. Standard `auc` can be misleading on imbalanced data.

# %%
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr', # CRITICAL for imbalance
    'scale_pos_weight': scale_weight, # Balances the gradient
    'max_depth': 4,
    'eta': 0.1
}

bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], verbose_eval=20)

# %% [markdown]
# ## 3. Evaluation (Precision-Recall Curve)

# %%
preds_prob = bst.predict(dtest)

precision, recall, thresholds = precision_recall_curve(y_test, preds_prob)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, marker='.')
plt.title(f'Precision-Recall Curve (AUC={pr_auc:.3f})')
plt.xlabel('Recall (True Positive Rate)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.show()

# Convert to binary using a custom threshold (e.g., 0.5)
# In reality, you'd pick a threshold based on the PR curve above that balances cost of False Positives vs False Negatives.
preds_binary = (preds_prob > 0.5).astype(int)
print(classification_report(y_test, preds_binary))
