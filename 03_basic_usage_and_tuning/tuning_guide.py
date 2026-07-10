# %% [markdown]
# # Basic Usage and Hyperparameter Tuning
# 
# This notebook covers standard training, understanding the DMatrix, and tuning hyperparameters using `Optuna`.

# %%
import xgboost as xgb
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %% [markdown]
# ## 1. Data Preparation (DMatrix)
# We will use the Breast Cancer classification dataset.

# %%
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# %% [markdown]
# ## 2. Basic Training

# %%
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 3
}

bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrain, 'train'), (dtest, 'test')], verbose_eval=20)
preds_prob = bst.predict(dtest)
preds_binary = (preds_prob > 0.5).astype(int)
print(f"Baseline Accuracy: {accuracy_score(y_test, preds_binary):.4f}")

# %% [markdown]
# ## 3. Hyperparameter Tuning with Optuna
# Optuna uses Bayesian optimization to find the best hyperparameters. It's much faster and smarter than Grid Search.

# %%
def objective(trial):
    # Suggest hyperparameters
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True)
    }
    
    # Pruning (Early Stopping) callback for Optuna is available in `optuna.integration.XGBoostPruningCallback`
    # For simplicity, we will just use XGBoost's built-in early stopping
    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'test-logloss')
    
    bst_cv = xgb.cv(
        param, 
        dtrain, 
        nfold=3, 
        num_boost_round=200, 
        early_stopping_rounds=20, 
        metrics='logloss',
        callbacks=[pruning_callback]
    )
    
    return bst_cv['test-logloss-mean'].min()

# %%
# Run the optimization
# Note: we suppress output here for brevity, but in practice you can watch Optuna explore the space!
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20) # Usually 100+ for real projects

print("Best trial:")
trial = study.best_trial
print(f"  Value (LogLoss): {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# %% [markdown]
# ## 4. Visualizing the Tuning Process
# Optuna provides excellent visualizations to understand parameter importance.

# %%
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title("Hyperparameter Importances")
plt.tight_layout()
plt.show()
