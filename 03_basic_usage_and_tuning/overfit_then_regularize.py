"""
XGBoost Overfit-Then-Regularize Ablation & Imbalance Diagnostic Suite
====================================================================

Implements the Day 2 curriculum requirements:
1. Multi-stage ablation experiment starting from an unregularized overfit model
   down through depth, hessian guard, gain pruning, stochastic sampling, and L2 shrinkage.
2. Generates a 3x3 multi-panel learning curve plot with Stage 0 overlaid as reference.
3. Imbalance evaluation comparing scale_pos_weight=1.0 vs 49.0 across operational
   fraud alert thresholds (Precision, Recall, Alert Volume <= 5%).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split


def generate_fraud_dataset(
    n_samples: int = 5000,
    n_features: int = 20,
    n_informative: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Generates synthetic credit/fraud dataset with 98% / 2% class imbalance."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=4,
        weights=[0.98, 0.02],
        flip_y=0.005,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    scale_pos_weight = neg_count / pos_count
    return X_train, X_val, y_train, y_val, scale_pos_weight


def get_ablation_stages() -> List[Dict[str, Any]]:
    """Defines the 9 ablation stages (0 through 5b)."""
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "learning_rate": 0.1,
        "tree_method": "hist",
        "seed": 42,
    }

    stages = [
        {
            "id": "Stage 0",
            "name": "Overfit Baseline",
            "delta_label": "max_depth=9, gamma=0, lambda=1, mcw=1, sub=1.0",
            "params": {
                **base_params,
                "max_depth": 9,
                "min_child_weight": 1.0,
                "gamma": 0.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
            },
        },
        {
            "id": "Stage 1",
            "name": "Depth Constrained",
            "delta_label": "max_depth 9 -> 4",
            "params": {
                **base_params,
                "max_depth": 4,
                "min_child_weight": 1.0,
                "gamma": 0.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
            },
        },
        {
            "id": "Stage 2a",
            "name": "Hessian Guard (mcw=5)",
            "delta_label": "min_child_weight 1 -> 5",
            "params": {
                **base_params,
                "max_depth": 4,
                "min_child_weight": 5.0,
                "gamma": 0.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
            },
        },
        {
            "id": "Stage 2b",
            "name": "Aggressive Hessian (mcw=20)",
            "delta_label": "min_child_weight 5 -> 20",
            "params": {
                **base_params,
                "max_depth": 4,
                "min_child_weight": 20.0,
                "gamma": 0.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
            },
        },
        {
            "id": "Stage 3a",
            "name": "Gain Pruning (gamma=1.0)",
            "delta_label": "gamma 0 -> 1.0 (mcw=5)",
            "params": {
                **base_params,
                "max_depth": 4,
                "min_child_weight": 5.0,
                "gamma": 1.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
            },
        },
        {
            "id": "Stage 3b",
            "name": "Aggressive Pruning (gamma=5.0)",
            "delta_label": "gamma 1.0 -> 5.0 (mcw=5)",
            "params": {
                **base_params,
                "max_depth": 4,
                "min_child_weight": 5.0,
                "gamma": 5.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
            },
        },
        {
            "id": "Stage 4",
            "name": "Stochastic Subsampling",
            "delta_label": "subsample=0.8, colsample=0.8",
            "params": {
                **base_params,
                "max_depth": 4,
                "min_child_weight": 5.0,
                "gamma": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
        },
        {
            "id": "Stage 5a",
            "name": "L2 Shrinkage (lambda=5.0)",
            "delta_label": "reg_lambda 1.0 -> 5.0",
            "params": {
                **base_params,
                "max_depth": 4,
                "min_child_weight": 5.0,
                "gamma": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
            },
        },
        {
            "id": "Stage 5b",
            "name": "Aggressive L2 (lambda=10.0)",
            "delta_label": "reg_lambda 5.0 -> 10.0",
            "params": {
                **base_params,
                "max_depth": 4,
                "min_child_weight": 5.0,
                "gamma": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 10.0,
            },
        },
    ]
    return stages


def run_ablation_experiment(
    num_boost_round: int = 120,
    output_dir: str = "03_basic_usage_and_tuning",
) -> Dict[str, Any]:
    """Runs all 9 stages, logs evals_result, and generates learning_curves_ablation.png."""
    X_train, X_val, y_train, y_val, spw = generate_fraud_dataset()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evallist = [(dtrain, "train"), (dval, "val")]

    stages = get_ablation_stages()
    stage_results: List[Dict[str, Any]] = []

    print(f"Starting Day 2 Ablation Experiment ({len(stages)} stages, {num_boost_round} rounds)...")

    for stage in stages:
        evals_result: Dict[str, Dict[str, List[float]]] = {}
        bst = xgb.train(
            stage["params"],
            dtrain,
            num_boost_round=num_boost_round,
            evals=evallist,
            evals_result=evals_result,
            verbose_eval=False,
        )

        train_loss = evals_result["train"]["logloss"]
        val_loss = evals_result["val"]["logloss"]
        val_auc = evals_result["val"]["auc"]

        min_val_loss = float(np.min(val_loss))
        best_round = int(np.argmin(val_loss)) + 1
        final_train_loss = float(train_loss[-1])
        final_val_loss = float(val_loss[-1])
        final_gap = final_val_loss - final_train_loss
        max_auc = float(np.max(val_auc))

        stage_results.append({
            "stage": stage,
            "booster": bst,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "min_val_loss": min_val_loss,
            "best_round": best_round,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_gap": final_gap,
            "max_auc": max_auc,
        })

        print(
            f"  {stage['id']} ({stage['name']}): "
            f"Min Val LogLoss={min_val_loss:.4f} (Round {best_round}) | "
            f"Final Gap={final_gap:.4f} | Max AUC={max_auc:.4f}"
        )

    # Plot Multi-Panel Figure
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    rounds = np.arange(1, num_boost_round + 1)
    s0_train = stage_results[0]["train_loss"]
    s0_val = stage_results[0]["val_loss"]

    for i, res in enumerate(stage_results):
        ax = axes_flat[i]
        stage_info = res["stage"]

        # Stage 0 reference lines (grey) on stages 1 through 5b
        if i > 0:
            ax.plot(rounds, s0_train, color="#94a3b8", linestyle=":", alpha=0.6, label="Stage 0 Train (ref)")
            ax.plot(rounds, s0_val, color="#64748b", linestyle="--", alpha=0.6, label="Stage 0 Val (ref)")

        # Current stage curves
        ax.plot(rounds, res["train_loss"], color="#0284c7", linewidth=2.0, label="Train LogLoss")
        ax.plot(rounds, res["val_loss"], color="#ea580c", linewidth=2.0, linestyle="--", label="Val LogLoss")

        # Mark minimum val loss
        best_r = res["best_round"]
        min_v = res["min_val_loss"]
        ax.scatter([best_r], [min_v], color="#dc2626", s=45, zorder=5, label=f"Min Val ({min_v:.3f})")

        ax.set_title(f"{stage_info['id']}: {stage_info['name']}\n[{stage_info['delta_label']}]", fontsize=11, fontweight="bold")
        ax.set_ylabel("LogLoss", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

        gap_text = f"Final Gap: {res['final_gap']:.4f}\nBest Rd: {best_r}"
        ax.text(
            0.96, 0.94, gap_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#cbd5e1")
        )

        if i == 0:
            ax.legend(loc="upper left", fontsize=8.5)

    for ax in axes[-1]:
        ax.set_xlabel("Boosting Round", fontsize=10)

    fig.suptitle(
        "XGBoost Overfit-Then-Regularize Ablation Suite\n"
        "(Tracking Generalization Gap Across Tree Depth, Hessian Guard, Gain Pruning, Sampling & L2)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "learning_curves_ablation.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved multi-panel ablation plot to {plot_path}")

    return {
        "stage_results": stage_results,
        "plot_path": plot_path,
        "X_val": X_val,
        "y_val": y_val,
        "scale_pos_weight": spw,
    }


def run_imbalance_suite(
    output_dir: str = "03_basic_usage_and_tuning",
) -> Dict[str, Any]:
    """Compares scale_pos_weight=1.0 vs 49.0 on Stage 4 architecture across thresholds."""
    X_train, X_val, y_train, y_val, spw = generate_fraud_dataset()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Base configuration: Stage 4 parameters
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "learning_rate": 0.1,
        "tree_method": "hist",
        "max_depth": 4,
        "min_child_weight": 5.0,
        "gamma": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "seed": 42,
    }

    print("\nRunning Imbalance Suite (scale_pos_weight Comparison)...")
    # Config A: scale_pos_weight = 1.0
    params_A = {**base_params, "scale_pos_weight": 1.0}
    bst_A = xgb.train(params_A, dtrain, num_boost_round=100)
    preds_A = bst_A.predict(dval)

    # Config B: scale_pos_weight = spw (~49.0)
    params_B = {**base_params, "scale_pos_weight": spw}
    bst_B = xgb.train(params_B, dtrain, num_boost_round=100)
    preds_B = bst_B.predict(dval)

    thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    n_val = len(y_val)

    def evaluate_thresholds(preds: np.ndarray, label: str) -> List[Dict[str, float]]:
        metrics_list = []
        for t in thresholds:
            pred_binary = (preds >= t).astype(int)
            flagged = int(np.sum(pred_binary == 1))
            alert_vol = flagged / n_val
            prec = float(precision_score(y_val, pred_binary, zero_division=0))
            rec = float(recall_score(y_val, pred_binary, zero_division=0))
            f1 = float(f1_score(y_val, pred_binary, zero_division=0))
            metrics_list.append({
                "threshold": t,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "alert_volume": alert_vol,
                "flagged_count": flagged,
            })
            print(
                f"  [{label}] t={t:.2f} -> AlertVol={alert_vol*100:.1f}% ({flagged}/{n_val}) | "
                f"Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}"
            )
        return metrics_list

    print("\n--- Config A: scale_pos_weight = 1.0 (Default) ---")
    metrics_A = evaluate_thresholds(preds_A, "spw=1.0")

    print(f"\n--- Config B: scale_pos_weight = {spw:.1f} (Canonical) ---")
    metrics_B = evaluate_thresholds(preds_B, f"spw={spw:.1f}")

    # Identify operational performance at alert volume <= 5%
    # Search fine-grained threshold grid for each config to find t where alert_vol <= 0.05
    fine_grid = np.linspace(0.01, 0.99, 1000)

    def find_ops_metrics(preds: np.ndarray, max_alert_vol: float = 0.05) -> Dict[str, float]:
        for t in fine_grid:
            bin_preds = (preds >= t).astype(int)
            vol = np.sum(bin_preds == 1) / n_val
            if vol <= max_alert_vol:
                prec = float(precision_score(y_val, bin_preds, zero_division=0))
                rec = float(recall_score(y_val, bin_preds, zero_division=0))
                f1 = float(f1_score(y_val, bin_preds, zero_division=0))
                return {
                    "threshold": float(t),
                    "alert_volume": float(vol),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                }
        return {"threshold": 1.0, "alert_volume": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    ops_A = find_ops_metrics(preds_A, 0.05)
    ops_B = find_ops_metrics(preds_B, 0.05)

    print("\n--- Operational Evaluation: Capacity-Constrained Alert Volume <= 5% ---")
    print(
        f"Config A (spw=1.0):   t={ops_A['threshold']:.3f} | AlertVol={ops_A['alert_volume']*100:.2f}% | "
        f"Precision={ops_A['precision']:.4f} | Recall={ops_A['recall']:.4f}"
    )
    print(
        f"Config B (spw={spw:.1f}):  t={ops_B['threshold']:.3f} | AlertVol={ops_B['alert_volume']*100:.2f}% | "
        f"Precision={ops_B['precision']:.4f} | Recall={ops_B['recall']:.4f}"
    )

    # Plot Precision-Recall & Alert Volume vs Recall Curves
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: PR Curves
    prec_curve_A, rec_curve_A, _ = precision_recall_curve(y_val, preds_A)
    prec_curve_B, rec_curve_B, _ = precision_recall_curve(y_val, preds_B)
    ax1.plot(rec_curve_A, prec_curve_A, color="#0284c7", linewidth=2.0, label="Config A (spw=1.0)")
    ax1.plot(rec_curve_B, prec_curve_B, color="#ea580c", linewidth=2.0, linestyle="--", label=f"Config B (spw={spw:.1f})")
    ax1.set_title("Precision-Recall Curve Comparison", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Recall", fontsize=10)
    ax1.set_ylabel("Precision", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=10)

    # Subplot 2: Alert Volume vs Recall across fine sweep
    vols_A = [np.mean(preds_A >= t) for t in fine_grid]
    recs_A = [recall_score(y_val, (preds_A >= t).astype(int), zero_division=0) for t in fine_grid]
    vols_B = [np.mean(preds_B >= t) for t in fine_grid]
    recs_B = [recall_score(y_val, (preds_B >= t).astype(int), zero_division=0) for t in fine_grid]

    ax2.plot(vols_A, recs_A, color="#0284c7", linewidth=2.0, label="Config A (spw=1.0)")
    ax2.plot(vols_B, recs_B, color="#ea580c", linewidth=2.0, linestyle="--", label=f"Config B (spw={spw:.1f})")
    ax2.axvline(0.05, color="#dc2626", linestyle=":", linewidth=1.8, label="5% Ops Capacity Cap")

    ax2.set_title("Alert Volume vs. Captured Recall", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Alert Volume (% Total Cases Flagged)", fontsize=10)
    ax2.set_ylabel("Recall (% Fraud Detected)", fontsize=10)
    ax2.set_xlim(0, 0.20)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=10)

    fig.suptitle(
        "XGBoost Imbalance Diagnostics: Default vs. scale_pos_weight under Operational Alert Budgets",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    pr_plot_path = os.path.join(output_dir, "fraud_threshold_metrics.png")
    plt.savefig(pr_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved threshold metrics plot to {pr_plot_path}")

    return {
        "metrics_A": metrics_A,
        "metrics_B": metrics_B,
        "ops_A": ops_A,
        "ops_B": ops_B,
        "plot_path": pr_plot_path,
        "scale_pos_weight": spw,
    }


if __name__ == "__main__":
    ablation_out = run_ablation_experiment(num_boost_round=120)
    imbalance_out = run_imbalance_suite()
    print("\nExecution complete! Generated both figures and empirical metrics.")
