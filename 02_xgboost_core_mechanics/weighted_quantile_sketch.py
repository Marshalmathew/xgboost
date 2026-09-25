"""
Weighted Quantile Sketch & Approximate Split Finding Engine
============================================================

Mathematical Derivation and Implementation from Chen & Guestrin (2016):
"XGBoost: A Scalable Tree Boosting System" (arXiv:1603.02754), Section 3.2, 3.3 & Appendix A.

Theoretical Foundation:
-----------------------
1. Why Hessian Weighting?
   Rewriting the second-order Taylor expansion of the objective around step (t-1):
   L^(t) approx sum_i [ g_i * f_t(x_i) + 0.5 * h_i * f_t^2(x_i) ] + Omega(f_t)
         = sum_i [ 0.5 * h_i * (f_t(x_i) - (-g_i / h_i))^2 ] + Omega(f_t) + const

   This is literally a WEIGHTED SQUARED ERROR loss with target (-g_i / h_i) and
   instance weight h_i. The hessian h_i is the local curvature of the loss,
   representing the 'information content' or 'certainty' of sample i.
   Therefore, candidate split points must be distributed evenly by HESSIAN MASS,
   not by instance count or raw feature value range.

2. Rank Functions (Appendix A):
   Let multiset D_k = {(x_1k, h_1), (x_2k, h_2), ..., (x_nk, h_n)}.
   Total weight H = sum_i h_i.
   r_k^-(y) = sum_{x_i < y} h_i / H
   r_k^+(y) = sum_{x_i <= y} h_i / H

   A valid eps-approximation sketch proposes candidates S_k = {s_k1, s_k2, ..., s_kl}
   such that: |r_k(s_{kj}) - r_k(s_{k, j-1})| <= eps = 1 / l.
"""

from __future__ import annotations

import json
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb


class WeightedQuantileSketch:
    """
    Implements the Weighted Quantile Sketch algorithm for proposing split candidates
    based on cumulative hessian mass distribution.
    """

    def __init__(self, eps: float = 0.1) -> None:
        """
        Parameters
        ----------
        eps : float
            Approximation factor (default 0.1 means ~10 candidate split buckets).
        """
        self.eps = eps

    @staticmethod
    def compute_ranks(
        feature_vals: np.ndarray, hessians: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Computes unnormalized rank functions r^-(y) and r^+(y) for sorted feature values.

        Returns
        -------
        sorted_x : np.ndarray
            Unique sorted feature values.
        r_minus : np.ndarray
            Sum of hessians for all x < y.
        r_plus : np.ndarray
            Sum of hessians for all x <= y.
        total_hessian : float
            Total sum of hessians H.
        """
        order = np.argsort(feature_vals)
        x_sorted = feature_vals[order]
        h_sorted = hessians[order]

        # Aggregate duplicate feature values
        unique_x, unique_idx = np.unique(x_sorted, return_inverse=True)
        aggregated_h = np.zeros_like(unique_x, dtype=float)
        np.add.at(aggregated_h, unique_idx, h_sorted)

        total_hessian = float(np.sum(aggregated_h))

        # Cumulative sums
        cum_h = np.cumsum(aggregated_h)
        r_plus = cum_h
        r_minus = np.roll(cum_h, 1)
        r_minus[0] = 0.0

        return unique_x, r_minus, r_plus, total_hessian

    def get_split_candidates(
        self, feature_vals: np.ndarray, hessians: np.ndarray
    ) -> np.ndarray:
        """
        Proposes candidate split points s_k such that consecutive candidates differ
        in cumulative hessian rank by at most eps * Total_Hessian.

        According to Section 3.3:
        |r_k(s_{k,j}) - r_k(s_{k,j-1})| <= eps
        """
        unique_x, _, r_plus, total_h = self.compute_ranks(feature_vals, hessians)

        if total_h <= 1e-12 or len(unique_x) <= 1:
            return unique_x

        # Step size in normalized rank space
        normalized_r_plus = r_plus / total_h

        candidates = [unique_x[0]]
        current_target = self.eps

        for i, (x_val, rank_val) in enumerate(zip(unique_x, normalized_r_plus)):
            if rank_val >= current_target:
                candidates.append(x_val)
                # Advance target to next quantile boundary
                while current_target <= rank_val:
                    current_target += self.eps

        # Ensure maximum feature value is covered
        if candidates[-1] != unique_x[-1]:
            candidates.append(unique_x[-1])

        return np.unique(np.array(candidates))


def compare_split_proposals(
    feature_vals: np.ndarray, hessians: np.ndarray, n_bins: int = 10
):
    """
    Compares 3 candidate split strategies on a skewed hessian dataset:
    1. Uniform Value-Spaced Bins (naive linear interpolation)
    2. Uniform Count-Spaced Bins (unweighted quantiles)
    3. Weighted Quantile Sketch (XGBoost hessian-mass quantiles)
    """
    eps = 1.0 / n_bins
    sketch = WeightedQuantileSketch(eps=eps)
    weighted_candidates = sketch.get_split_candidates(feature_vals, hessians)

    # 1. Uniform value-spaced
    val_min, val_max = np.min(feature_vals), np.max(feature_vals)
    value_candidates = np.linspace(val_min, val_max, n_bins + 1)

    # 2. Uniform count-spaced (unweighted empirical percentiles)
    count_candidates = np.percentile(feature_vals, np.linspace(0, 100, n_bins + 1))

    return {
        "weighted_hessian_candidates": weighted_candidates.tolist(),
        "value_spaced_candidates": value_candidates.tolist(),
        "count_spaced_candidates": count_candidates.tolist(),
    }


def run_approx_vs_exact_benchmark():
    """
    Empirically benchmarks:
    1. The candidate split clustering on a dataset with deliberately skewed hessians.
    2. Real XGBoost `tree_method='approx'` vs `tree_method='exact'` convergence and trees.
    """
    np.random.seed(42)
    n_samples = 4000

    # Feature x is uniform continuous [0, 100]
    X_feat = np.random.uniform(0.0, 100.0, n_samples)

    # Skewed Hessian Profile:
    # Samples with x in [40, 60] are near the decision boundary -> high uncertainty -> p ~ 0.5 -> h = 0.25
    # Samples with x < 40 or x > 60 are very confident -> p ~ 0.99 or 0.01 -> h = 0.01
    is_borderline = (X_feat >= 40.0) & (X_feat <= 60.0)
    hessians = np.where(is_borderline, 0.25, 0.01)

    # Generate labels with non-linear threshold at x=50
    prob_true = 1.0 / (1.0 + np.exp(-(X_feat - 50.0) / 5.0))
    y = np.random.binomial(1, prob_true)

    # 1. Evaluate candidate split clustering
    proposals = compare_split_proposals(X_feat, hessians, n_bins=10)

    # Calculate density of candidates inside high-hessian region [40, 60]
    def count_in_region(cand_list, low=40.0, high=60.0):
        return sum(1 for c in cand_list if low <= c <= high)

    w_count = count_in_region(proposals["weighted_hessian_candidates"])
    val_count = count_in_region(proposals["value_spaced_candidates"])
    cnt_count = count_in_region(proposals["count_spaced_candidates"])

    clustering_summary = {
        "region_high_hessian": "[40.0, 60.0]",
        "total_candidates_weighted": len(proposals["weighted_hessian_candidates"]),
        "candidates_in_high_hessian_weighted": w_count,
        "percentage_in_high_hessian_weighted": w_count / len(proposals["weighted_hessian_candidates"]),
        "candidates_in_high_hessian_value_spaced": val_count,
        "percentage_in_high_hessian_value_spaced": val_count / len(proposals["value_spaced_candidates"]),
        "candidates_in_high_hessian_count_spaced": cnt_count,
        "percentage_in_high_hessian_count_spaced": cnt_count / len(proposals["count_spaced_candidates"]),
    }

    # 2. Benchmark real XGBoost 'approx' vs 'exact'
    X_matrix = np.column_stack([X_feat, np.random.normal(0, 1, (n_samples, 4))])
    dtrain = xgb.DMatrix(X_matrix, label=y, feature_names=["f0", "f1", "f2", "f3", "f4"])

    params_exact = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "exact",
        "max_depth": 3,
        "learning_rate": 0.1,
        "seed": 42,
    }
    bst_exact = xgb.train(params_exact, dtrain, num_boost_round=15)
    preds_exact = bst_exact.predict(dtrain)

    params_approx = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "approx",
        "max_depth": 3,
        "learning_rate": 0.1,
        "seed": 42,
    }
    bst_approx = xgb.train(params_approx, dtrain, num_boost_round=15)
    preds_approx = bst_approx.predict(dtrain)

    correlation = float(np.corrcoef(preds_exact, preds_approx)[0, 1])
    max_abs_diff = float(np.max(np.abs(preds_exact - preds_approx)))
    mean_abs_diff = float(np.mean(np.abs(preds_exact - preds_approx)))

    # Compare tree structure dumps
    dump_exact = bst_exact.get_dump()[0]
    dump_approx = bst_approx.get_dump()[0]

    benchmark_results = {
        "clustering_summary": clustering_summary,
        "real_xgboost_comparison": {
            "correlation_exact_vs_approx": correlation,
            "max_absolute_prediction_diff": max_abs_diff,
            "mean_absolute_prediction_diff": mean_abs_diff,
            "root_split_exact": dump_exact.split("\n")[0],
            "root_split_approx": dump_approx.split("\n")[0],
        },
        "proposals": proposals,
    }

    # 3. Generate Publication Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Subplot 1: Data distribution & Hessian mass density
    ax1.scatter(X_feat, hessians, color="#1f77b4", alpha=0.3, s=12, label="Instance Hessians h_i")
    ax1.set_ylabel("Hessian Weight h_i", fontsize=11)
    ax1.set_title("Instance Hessian Curvature Distribution (Dense Uncertainty in [40, 60])", fontsize=12, fontweight="bold")
    ax1.axvspan(40, 60, color="orange", alpha=0.15, label="High-Information Region [40, 60]")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Candidate split locations
    y_weighted = np.full_like(proposals["weighted_hessian_candidates"], 3.0)
    y_count = np.full_like(proposals["count_spaced_candidates"], 2.0)
    y_val = np.full_like(proposals["value_spaced_candidates"], 1.0)

    ax2.plot(proposals["weighted_hessian_candidates"], y_weighted, "ro-", label="Weighted Quantile Sketch (XGBoost)", markersize=8, linewidth=1.5)
    ax2.plot(proposals["count_spaced_candidates"], y_count, "bs-", label="Uniform Count Bins (Classical Quantile)", markersize=7, linewidth=1.5)
    ax2.plot(proposals["value_spaced_candidates"], y_val, "g^-", label="Uniform Value Bins (Equal Width)", markersize=7, linewidth=1.5)

    ax2.axvspan(40, 60, color="orange", alpha=0.15)
    ax2.set_yticks([1.0, 2.0, 3.0])
    ax2.set_yticklabels(["Value-Spaced", "Count-Spaced", "Hessian-Weighted"])
    ax2.set_xlabel("Feature Value X", fontsize=11)
    ax2.set_title("Candidate Split Proposals Comparison: Hessian-Weighted Bins Cluster in High-Curvature Region", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = "02_xgboost_core_mechanics/weighted_quantile_sketch_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # 4. Serialize JSON
    json_path = "02_xgboost_core_mechanics/quantile_sketch_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=4)

    return benchmark_results


if __name__ == "__main__":
    print("Running Weighted Quantile Sketch & Approximate Split Finding Benchmark...")
    results = run_approx_vs_exact_benchmark()
    print("\n--- Benchmark Complete ---")
    print("Candidates in High-Hessian [40, 60]:")
    print(f"  - Hessian-Weighted Sketch: {results['clustering_summary']['candidates_in_high_hessian_weighted']} / {results['clustering_summary']['total_candidates_weighted']} ({results['clustering_summary']['percentage_in_high_hessian_weighted']:.1%})")
    print(f"  - Equal-Width Value Bins:  {results['clustering_summary']['candidates_in_high_hessian_value_spaced']} / 11 ({results['clustering_summary']['percentage_in_high_hessian_value_spaced']:.1%})")
    print(f"  - Count-Spaced Bins:        {results['clustering_summary']['candidates_in_high_hessian_count_spaced']} / 11 ({results['clustering_summary']['percentage_in_high_hessian_count_spaced']:.1%})")
    print(f"\nReal XGBoost exact vs approx correlation: r = {results['real_xgboost_comparison']['correlation_exact_vs_approx']:.6f}")
    print(f"Mean absolute prediction difference: {results['real_xgboost_comparison']['mean_absolute_prediction_diff']:.6f}")
    print("Artifacts serialized to 02_xgboost_core_mechanics/quantile_sketch_benchmark.json")
