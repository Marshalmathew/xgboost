"""
Staged Hyperparameter Tuning Pipeline for XGBoost.

Instead of unconstrained high-dimensional search over 10+ hyperparameters simultaneously,
this pipeline executes a disciplined 4-stage optimization strategy:
  Stage 1: Tree Architecture (max_depth, min_child_weight, gamma)
  Stage 2: Stochastic Sampling & Regularization (subsample, colsample_bytree, reg_alpha, reg_lambda)
  Stage 3: Learning Rate Shrinkage (eta exploration with fixed capacity)
  Stage 4: Capacity Scaling (discovering optimal n_estimators via xgb.cv with early stopping)

Target Metric: PR-AUC (Average Precision Score) for highly imbalanced fraud detection.
"""

import json
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import xgboost as xgb
from optuna.pruners import MedianPruner
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# Handle Optuna integration imports cleanly across versions
try:
    from optuna_integration import XGBoostPruningCallback
except ImportError:
    from optuna.integration import XGBoostPruningCallback


def generate_fraud_dataset(
    n_samples: int = 10000, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic tabular transaction dataset with 2% fraud prevalence."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=18,
        n_redundant=6,
        n_clusters_per_class=2,
        weights=[0.98, 0.02],
        flip_y=0.01,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)


class StagedXGBoostTuner:
    """Disciplined multi-stage Bayesian hyperparameter tuner for XGBoost."""

    def __init__(self, random_state: int = 42, n_trials_per_stage: int = 15):
        self.random_state = random_state
        self.n_trials = n_trials_per_stage
        self.tuned_params: Dict[str, Any] = {
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": self.random_state,
            "nthread": -1,
        }
        self.stage_history: Dict[str, Dict[str, Any]] = {}

    def stage_1_tree_architecture(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Tune structural capacity: max_depth, min_child_weight, and gamma."""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        def objective(trial: optuna.Trial) -> float:
            params = {
                **self.tuned_params,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }

            pruning_cb = XGBoostPruningCallback(trial, "val-logloss")
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dval, "val")],
                callbacks=[pruning_cb],
                verbose_eval=False,
            )
            preds = bst.predict(dval)
            return float(average_precision_score(y_val, preds))

        pruner = MedianPruner(n_startup_trials=4, n_warmup_steps=10)
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials)

        best_p = study.best_params
        self.tuned_params.update(best_p)
        self.stage_history["stage_1_tree_architecture"] = {
            "params": best_p,
            "best_pr_auc": round(study.best_value, 5),
            "completed_trials": len(study.trials),
        }
        return best_p

    def stage_2_randomness_and_regularization(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """Tune stochastic subsampling and L1/L2 penalties via Stratified 3-Fold CV."""
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        def objective(trial: optuna.Trial) -> float:
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)

            fold_scores = []
            for train_idx, val_idx in skf.split(X, y):
                dtr = xgb.DMatrix(X[train_idx], label=y[train_idx])
                dva = xgb.DMatrix(X[val_idx], label=y[val_idx])

                p = {
                    **self.tuned_params,
                    "learning_rate": 0.1,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "reg_alpha": reg_alpha,
                    "reg_lambda": reg_lambda,
                }
                bst = xgb.train(p, dtr, num_boost_round=100, verbose_eval=False)
                preds = bst.predict(dva)
                fold_scores.append(average_precision_score(y[val_idx], preds))

            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials)

        best_p = study.best_params
        self.tuned_params.update(best_p)
        self.stage_history["stage_2_randomness_and_regularization"] = {
            "params": best_p,
            "best_pr_auc_cv": round(study.best_value, 5),
            "completed_trials": len(study.trials),
        }
        return best_p

    def stage_3_learning_rate_shrinkage(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """Explore learning rate shrinkage for finer gradient descent steps."""
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
            fold_scores = []
            for train_idx, val_idx in skf.split(X, y):
                dtr = xgb.DMatrix(X[train_idx], label=y[train_idx])
                dva = xgb.DMatrix(X[val_idx], label=y[val_idx])
                p = {**self.tuned_params, "learning_rate": lr}
                bst = xgb.train(p, dtr, num_boost_round=150, verbose_eval=False)
                preds = bst.predict(dva)
                fold_scores.append(average_precision_score(y[val_idx], preds))
            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials)

        best_p = study.best_params
        self.tuned_params.update(best_p)
        self.stage_history["stage_3_learning_rate_shrinkage"] = {
            "params": best_p,
            "best_pr_auc_cv": round(study.best_value, 5),
        }
        return best_p

    def stage_4_capacity_scaling(self, X: np.ndarray, y: np.ndarray) -> int:
        """Lock learning rate and discover exact optimal n_estimators via xgb.cv."""
        dtrain_full = xgb.DMatrix(X, label=y)
        cv_results = xgb.cv(
            self.tuned_params,
            dtrain_full,
            num_boost_round=1000,
            nfold=5,
            stratified=True,
            early_stopping_rounds=40,
            metrics=["logloss"],
            seed=self.random_state,
            verbose_eval=False,
        )

        optimal_rounds = int(cv_results["test-logloss-mean"].idxmin()) + 1
        best_cv_loss = float(cv_results["test-logloss-mean"].min())

        self.tuned_params["n_estimators"] = optimal_rounds
        self.stage_history["stage_4_capacity_scaling"] = {
            "optimal_n_estimators": optimal_rounds,
            "best_cv_logloss": round(best_cv_loss, 5),
        }
        return optimal_rounds

    def run_full_pipeline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_json_path: str = None,
    ) -> Dict[str, Any]:
        """Execute all 4 stages sequentially and persist configuration."""
        print("Starting Stage 1: Tree Architecture Tuning...")
        self.stage_1_tree_architecture(X_train, y_train, X_val, y_val)
        print(f"Stage 1 Best PR-AUC: {self.stage_history['stage_1_tree_architecture']['best_pr_auc']}")

        print("Starting Stage 2: Randomness & Regularization Tuning...")
        self.stage_2_randomness_and_regularization(X_train, y_train)
        print(f"Stage 2 Best CV PR-AUC: {self.stage_history['stage_2_randomness_and_regularization']['best_pr_auc_cv']}")

        print("Starting Stage 3: Learning Rate Exploration...")
        self.stage_3_learning_rate_shrinkage(X_train, y_train)
        print(f"Stage 3 Best CV PR-AUC: {self.stage_history['stage_3_learning_rate_shrinkage']['best_pr_auc_cv']}")

        print("Starting Stage 4: Capacity Scaling via 5-Fold Early Stopping...")
        opt_rounds = self.stage_4_capacity_scaling(X_train, y_train)
        print(f"Stage 4 Discovered Optimal n_estimators: {opt_rounds}")

        # Final evaluation on holdout test set
        dtrain_all = xgb.DMatrix(X_train, label=y_train)
        dval_all = xgb.DMatrix(X_val, label=y_val)

        train_params = {k: v for k, v in self.tuned_params.items() if k != "n_estimators"}
        final_bst = xgb.train(
            train_params,
            dtrain_all,
            num_boost_round=self.tuned_params["n_estimators"],
            verbose_eval=False,
        )
        final_preds = final_bst.predict(dval_all)
        final_pr_auc = float(average_precision_score(y_val, final_preds))

        output_payload = {
            "final_holdout_pr_auc": round(final_pr_auc, 5),
            "final_hyperparameters": self.tuned_params,
            "stage_progression": self.stage_history,
        }

        if output_json_path:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=2)

        return output_payload


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "tuned_hyperparameters.json")

    print("Generating Tabular Fraud Dataset (N=10,000, 2% Fraud)...")
    X_tr, X_va, y_tr, y_va = generate_fraud_dataset(n_samples=10000, random_state=42)

    tuner = StagedXGBoostTuner(random_state=42, n_trials_per_stage=15)
    t0 = time.perf_counter()
    summary = tuner.run_full_pipeline(X_tr, y_tr, X_va, y_va, output_json_path=json_path)
    total_time = time.perf_counter() - t0

    print(f"\nPipeline finished in {total_time:.1f}s")
    print(f"Final Holdout PR-AUC: {summary['final_holdout_pr_auc']:.5f}")
    print(f"Optimal n_estimators: {summary['final_hyperparameters']['n_estimators']}")
    print(f"Saved locked config to {json_path}")
