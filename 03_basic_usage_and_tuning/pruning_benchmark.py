"""
Pruning Efficiency Benchmark for XGBoost.

Compares 25 trials WITH Optuna MedianPruner against 25 trials WITHOUT pruning (NopPruner).
Measures:
  1. Rounds executed per trial (full 100 rounds vs early stopping at round 10-30).
  2. Cumulative boosting rounds consumed across trial progression.
  3. Total wall-clock duration and speedup factor.
Generates:
  03_basic_usage_and_tuning/pruning_efficiency.png (two-panel publication-grade figure).
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import xgboost as xgb
from optuna.pruners import MedianPruner, NopPruner
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

try:
    from optuna_integration import XGBoostPruningCallback
except ImportError:
    from optuna.integration import XGBoostPruningCallback


def generate_benchmark_data(n_samples: int = 8000, random_state: int = 42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=25,
        n_informative=15,
        weights=[0.97, 0.03],
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)


class RoundTrackingCallback(xgb.callback.TrainingCallback):
    """Tracks how many boosting iterations were actually executed before pruning/finish."""

    def __init__(self):
        super().__init__()
        self.rounds_executed = 0

    def after_iteration(self, model, epoch, evals_log):
        self.rounds_executed = epoch + 1
        return False


def run_pruning_study(use_pruner: bool, n_trials: int = 25, random_state: int = 42):
    X_train, X_val, y_train, y_val = generate_benchmark_data(n_samples=8000, random_state=random_state)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    trial_records = []
    pruner = (
        MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
        if use_pruner
        else NopPruner()
    )

    def objective(trial: optuna.Trial) -> float:
        tracker = RoundTrackingCallback()
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 11),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "random_state": random_state,
            "nthread": -1,
        }

        callbacks = [tracker]
        if use_pruner:
            callbacks.append(XGBoostPruningCallback(trial, "val-logloss"))

        start_time = time.perf_counter()
        try:
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dval, "val")],
                callbacks=callbacks,
                verbose_eval=False,
            )
            elapsed = time.perf_counter() - start_time
            eval_res = bst.eval(dval, "val")
            val_loss = float(eval_res.split(":")[-1])
            trial_records.append({
                "trial_idx": trial.number,
                "rounds": tracker.rounds_executed,
                "time_sec": elapsed,
                "pruned": False,
                "val_loss": val_loss,
            })
            return val_loss
        except optuna.TrialPruned:
            elapsed = time.perf_counter() - start_time
            trial_records.append({
                "trial_idx": trial.number,
                "rounds": tracker.rounds_executed,
                "time_sec": elapsed,
                "pruned": True,
                "val_loss": np.nan,
            })
            raise

    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    total_duration = time.perf_counter() - t0

    return {
        "study": study,
        "records": trial_records,
        "total_duration": total_duration,
        "best_value": study.best_value,
    }


def plot_and_save_benchmark(records_no_prune, records_with_prune, out_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Left panel: Rounds executed per trial
    trials_no = [r["trial_idx"] for r in records_no_prune]
    rounds_no = [r["rounds"] for r in records_no_prune]

    rounds_pr = [r["rounds"] for r in records_with_prune]
    colors_pr = ["#dc2626" if r["pruned"] else "#2563eb" for r in records_with_prune]

    x = np.arange(len(trials_no))
    width = 0.38

    ax1.bar(x - width / 2, rounds_no, width, label="Without Pruning (Full 100 rds)", color="#94a3b8", alpha=0.85)
    ax1.bar(x + width / 2, rounds_pr, width, label="With MedianPruner", color=colors_pr, alpha=0.9)

    ax1.set_xlabel("Trial Index", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Boosting Rounds Executed", fontsize=11, fontweight="bold")
    ax1.set_title("Per-Trial Execution Depth (Red = Pruned Early)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels([str(i) for i in trials_no[::2]])
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", frameon=True)

    # Right panel: Cumulative boosting rounds consumed
    cum_rounds_no = np.cumsum(rounds_no)
    cum_rounds_pr = np.cumsum(rounds_pr)

    ax2.plot(x, cum_rounds_no, marker="o", color="#475569", lw=2.2, label="Without Pruning (Linear 100x)")
    ax2.plot(x, cum_rounds_pr, marker="s", color="#059669", lw=2.5, label="With MedianPruner (Aggressive Compute Savings)")
    ax2.fill_between(x, cum_rounds_pr, cum_rounds_no, color="#10b981", alpha=0.15, label="Compute Budget Saved")

    ax2.set_xlabel("Trial Progression (Count)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Boosting Rounds Consumed", fontsize=11, fontweight="bold")
    ax2.set_title("Cumulative Computational Investment", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, "pruning_efficiency.png")

    print("Running 25 trials WITHOUT pruning...")
    res_noprune = run_pruning_study(use_pruner=False, n_trials=25, random_state=42)
    print(f"Without pruning: {res_noprune['total_duration']:.2f}s, Best LogLoss: {res_noprune['best_value']:.5f}")

    print("\nRunning 25 trials WITH MedianPruner...")
    res_prune = run_pruning_study(use_pruner=True, n_trials=25, random_state=42)
    print(f"With pruning: {res_prune['total_duration']:.2f}s, Best LogLoss: {res_prune['best_value']:.5f}")

    tot_rounds_no = sum(r["rounds"] for r in res_noprune["records"])
    tot_rounds_pr = sum(r["rounds"] for r in res_prune["records"])
    pruned_count = sum(1 for r in res_prune["records"] if r["pruned"])
    rounds_saved = tot_rounds_no - tot_rounds_pr
    speedup = res_noprune["total_duration"] / res_prune["total_duration"]

    print("\n=== Pruning Benchmark Results ===")
    print(f"Unpruned Total Rounds: {tot_rounds_no} | Pruned Total Rounds: {tot_rounds_pr}")
    print(f"Trials Pruned: {pruned_count}/25 ({pruned_count/25*100:.1f}%)")
    print(f"Rounds Saved: {rounds_saved} ({rounds_saved/tot_rounds_no*100:.1f}%)")
    print(f"Duration: {res_noprune['total_duration']:.2f}s -> {res_prune['total_duration']:.2f}s ({speedup:.2f}x speedup)")

    plot_and_save_benchmark(res_noprune["records"], res_prune["records"], fig_path)
    print(f"Saved benchmark figure to {fig_path}")
