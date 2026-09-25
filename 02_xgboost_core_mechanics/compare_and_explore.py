"""
XGBoost Numerical Parity & Hyperparameter Exploration
=====================================================

Validates exact mathematical parity between our pure Python 2nd-order engine
(xgboost_scratch.py) and official C++ XGBoost (`xgboost.train`).

Covers:
1. XGBoost >= 2.0 base_score handling and margin extraction.
2. 1-round single tree exact margin comparison (atol <= 1e-4).
3. Multi-round (3 trees, eta=0.3) boosting ensemble margin comparison.
4. Systematic exploration of gamma (pruning), min_child_weight, and lambda (L2).
"""

from __future__ import annotations

import json

import numpy as np
import xgboost as xgb
from packaging import version
from sklearn.datasets import make_classification
from xgboost_scratch import XGBoostScratch


def get_xgboost_base_score(bst: xgb.Booster) -> float:
    """Extract actual base_score used by XGBoost booster across versions."""
    # Method 1: Check booster attribute
    attr_val = bst.attr("base_score")
    if attr_val is not None:
        try:
            return float(attr_val)
        except ValueError:
            pass

    # Method 2: Inspect internal JSON learner config
    try:
        cfg = json.loads(bst.save_config())
        learner_param = cfg.get("learner", {}).get("learner_model_param", {})
        raw_val = learner_param.get("base_score")
        if raw_val is not None:
            # Often formatted as "[5E-1]" or "0.5"
            cleaned = str(raw_val).strip("[]")
            return float(cleaned)
    except Exception:
        pass

    return 0.5


def run_parity_comparison() -> None:
    print("=" * 78)
    print("XGBOOST EXACT PARITY VALIDATION (SCRATCH vs OFFICIAL C++ ENGINE)")
    print("=" * 78)

    xgb_ver = xgb.__version__
    print(f"[*] Detected Official XGBoost Version: {xgb_ver}")

    if version.parse(xgb_ver) >= version.parse("2.0.0"):
        print(
            "[*] NOTE [XGBoost >= 2.0]: In modern XGBoost, base_score defaults dynamically\n"
            "    if not set explicitly. We pass base_score=0.5 explicitly and extract the\n"
            "    internal learner base_score from model config to guarantee exact margin parity."
        )

    # 1. Generate reproducible dataset
    X, y = make_classification(
        n_samples=50,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    dtrain = xgb.DMatrix(X, label=y)

    # ---------------------------------------------------------
    # Test 1: Single Round (1 Tree, eta=1.0)
    # ---------------------------------------------------------
    print("\n--- Test 1: Single Round (1 Tree, eta=1.0, max_depth=2, lambda=1.0) ---")
    params_single = {
        "max_depth": 2,
        "eta": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "base_score": 0.5,
        "tree_method": "exact",
        "objective": "binary:logistic",
    }
    bst_single = xgb.train(params_single, dtrain, num_boost_round=1)
    actual_base_score = get_xgboost_base_score(bst_single)

    official_margins_1 = bst_single.predict(dtrain, output_margin=True)

    scratch_1 = XGBoostScratch(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=2,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1.0,
        base_score=actual_base_score,
    )
    scratch_1.fit(X, y)
    scratch_margins_1 = scratch_1.predict_raw_margin(X)

    # Display side-by-side comparison for first 10 samples
    print(f"{'Sample':<8}{'Official XGBoost Margin':<26}{'Scratch Engine Margin':<26}{'Abs Difference':<18}")
    print("-" * 78)
    for i in range(10):
        diff = abs(official_margins_1[i] - scratch_margins_1[i])
        print(f"{i:<8}{official_margins_1[i]:<26.7f}{scratch_margins_1[i]:<26.7f}{diff:<18.2e}")

    max_diff_1 = float(np.max(np.abs(official_margins_1 - scratch_margins_1)))
    print("-" * 78)
    print(f"Max Absolute Error (50 samples): {max_diff_1:.2e}")
    assert np.allclose(official_margins_1, scratch_margins_1, atol=1e-4), "Parity failed for 1 tree!"
    print(">>> PASS: Single-tree predictions match official C++ XGBoost within 1e-4 tolerance!\n")

    # ---------------------------------------------------------
    # Test 2: Multi-Round Boosting (3 Trees, eta=0.3)
    # ---------------------------------------------------------
    print("--- Test 2: Multi-Round Ensemble (3 Trees, eta=0.3, max_depth=2) ---")
    params_multi = {
        "max_depth": 2,
        "eta": 0.3,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "base_score": 0.5,
        "tree_method": "exact",
        "objective": "binary:logistic",
    }
    bst_multi = xgb.train(params_multi, dtrain, num_boost_round=3)
    official_margins_3 = bst_multi.predict(dtrain, output_margin=True)

    scratch_3 = XGBoostScratch(
        n_estimators=3,
        learning_rate=0.3,
        max_depth=2,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1.0,
        base_score=actual_base_score,
    )
    scratch_3.fit(X, y)
    scratch_margins_3 = scratch_3.predict_raw_margin(X)

    max_diff_3 = float(np.max(np.abs(official_margins_3 - scratch_margins_3)))
    print(f"Max Absolute Error across 3 rounds: {max_diff_3:.2e}")
    assert np.allclose(official_margins_3, scratch_margins_3, atol=1e-4), "Parity failed for 3 rounds!"
    print(">>> PASS: Multi-round boosting loop matches official C++ XGBoost within 1e-4 tolerance!\n")

    # ---------------------------------------------------------
    # Exploration: How Hyperparameters Impact Tree Structure
    # ---------------------------------------------------------
    print("=" * 78)
    print("HYPERPARAMETER MECHANICS EXPLORATION")
    print("=" * 78)

    # Effect of Gamma (Tree Pruning)
    print("\n1. Effect of Gamma (Minimum Gain Threshold for Split Acceptance):")
    for g_val in [0.0, 1.0, 5.0, 20.0]:
        m = XGBoostScratch(n_estimators=1, max_depth=3, gamma=g_val, min_child_weight=1.0)
        m.fit(X, y)
        root = m.trees[0].root
        is_split = (root is not None) and (not root.is_leaf)
        root_gain = root.gain if (root and not root.is_leaf) else 0.0
        print(f"   gamma = {g_val:>4.1f} | Root split accepted? {str(is_split):<5} | Best Gain after -gamma: {root_gain:.4f}")

    # Effect of Lambda (L2 Regularization on Leaf Weights)
    print("\n2. Effect of Lambda (L2 Weight Shrinkage on Leaves):")
    for l_val in [0.01, 1.0, 10.0, 100.0]:
        m = XGBoostScratch(n_estimators=1, max_depth=1, reg_lambda=l_val, min_child_weight=0.1)
        m.fit(X, y)
        preds = m.predict_raw_margin(X)
        margin_spread = float(np.max(preds) - np.min(preds))
        print(f"   lambda = {l_val:>5.2f} | Margin Range: [{np.min(preds):+.4f}, {np.max(preds):+.4f}] | Spread: {margin_spread:.4f}")

    # Effect of min_child_weight
    print("\n3. Effect of min_child_weight (Minimum Sum of Hessian Required in Children):")
    for mcw in [0.1, 1.0, 5.0, 10.0]:
        m = XGBoostScratch(n_estimators=1, max_depth=2, min_child_weight=mcw)
        m.fit(X, y)
        root = m.trees[0].root
        is_split = (root is not None) and (not root.is_leaf)
        print(f"   min_child_weight = {mcw:>4.1f} | Root split allowed? {str(is_split):<5}")

    print("\n" + "=" * 78)
    print("All Parity Validations and Diagnostic Checks Completed Successfully!")
    print("=" * 78)


if __name__ == "__main__":
    run_parity_comparison()
