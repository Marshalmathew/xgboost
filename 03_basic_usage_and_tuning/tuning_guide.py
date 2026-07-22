# %% [markdown]
# # Basic Usage and Hyperparameter Tuning
# 
# This notebook covers standard XGBoost training, DMatrix usage, GridSearch baseline tuning, and advanced Bayesian Optimization with `Optuna`.

# %%
import xgboost as xgb
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Disable optuna logging verbosity for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
# ## 3A. Baseline Tuning: GridSearchCV (Scikit-Learn API)
# GridSearch systematically searches through a manually specified parameter grid.

# %%
xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_classifier,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("GridSearchCV Best Parameters:", grid_search.best_params_)
print(f"GridSearchCV Best Accuracy: {grid_search.best_score_:.4f}")

# %% [markdown]
# ## 3B. Advanced Tuning: Bayesian Optimization with Optuna
# Optuna uses Bayesian optimization to sample promising hyperparameter regions. It is faster and handles continuous spaces much better than Grid Search.

# %%
def objective(trial):
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

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("\nOptuna Best Trial:")
trial = study.best_trial
print(f"  Best LogLoss: {trial.value:.4f}")
print("  Optimized Parameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# %% [markdown]
# ## 4. Visualizing Optuna Parameter Importance

# %%
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title("Hyperparameter Importances (Optuna)")
plt.tight_layout()
plt.show()

