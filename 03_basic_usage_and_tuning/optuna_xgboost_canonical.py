"""
Canonical Minimal Optuna + XGBoost with Pruning.

Three Essential Gotchas Often Missed When Copy-Pasting:
1. Eval Key Matching: The callback's metric key format must match XGBoost's evals naming.
   If evals=[(dval, 'val')] and eval_metric='logloss', XGBoost logs 'val-logloss'.
   Passing the wrong string causes the pruning callback to silently do nothing!
2. Warmup Steps: Always set n_warmup_steps (e.g., 10-15) in MedianPruner.
   Without warmup, stochastic early fluctuations kill potentially stellar trials prematurely.
3. Direction Traps: Pruner & Study directions must align with the metric:
   'minimize' for logloss/rmse, 'maximize' for auc/pr_auc. Mismatch will prune good models.
"""

import optuna
import xgboost as xgb

try:
    from optuna_integration import XGBoostPruningCallback
except ImportError:
    from optuna.integration import XGBoostPruningCallback
from optuna.pruners import MedianPruner
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def run_canonical_study(n_trials: int = 15, random_state: int = 42):
    # 1. Dataset setup
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=12,
        weights=[0.95, 0.05],
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 2. Objective function with pruning callback
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": random_state,
            "nthread": -1,
        }

        # Key Gotcha #1: 'val-logloss' must match evals=[(dval, 'val')] + eval_metric='logloss'
        pruning_cb = XGBoostPruningCallback(trial, "val-logloss")

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dval, "val")],
            callbacks=[pruning_cb],
            verbose_eval=False,
        )

        # Using binary logloss evaluation from XGBoost evaluation record
        res = bst.eval(dval, "val")
        # res looks like: "[0]\tval-logloss:0.12345"
        val_loss = float(res.split(":")[-1])
        return val_loss

    # 3. MedianPruner with n_warmup_steps=10 to allow learning to stabilize
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    return study


if __name__ == "__main__":
    print("Executing Canonical Optuna + XGBoost Pruning Study...")
    study = run_canonical_study(n_trials=20)
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Trials completed: {len(complete)}, Trials pruned early: {len(pruned)}")
    print(f"Best trial validation logloss: {study.best_value:.5f}")
    print("Best params:", study.best_params)
