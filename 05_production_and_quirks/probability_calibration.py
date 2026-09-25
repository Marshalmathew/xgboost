"""
05_production_and_quirks/probability_calibration.py
Enterprise XGBoost Probability Calibration & Reliability Engineering Suite

Derived from:
1. Niculescu-Mizil & Caruana (AAAI 2007, arXiv:1207.1403): "Obtaining Calibrated Probabilities from Boosting"
2. Niculescu-Mizil & Caruana (ICML 2005): "Predicting Good Probabilities With Supervised Learning"
3. Murphy (1973): "A New Vector Partition of the Probability Score" (Brier Score Decomposition)
4. Scikit-Learn Probability Calibration Guide & CalibratedClassifierCV

Key Insights:
- Boosted trees push probability mass away from 0 and 1, creating a characteristic
  sigmoid-shaped distortion on reliability diagrams.
- Platt scaling (method='sigmoid') parametrically matches this exact sigmoid deformation.
- Isotonic regression (method='isotonic') is non-parametric and flexible, but requires >1000 samples
  and can overfit small positive slices in imbalanced banking data.
- When scale_pos_weight is used, probabilities are artificially shifted; calibration must be
  evaluated with and without analytical odds unbiasing.
- Models requiring calibration (Expected Loss = PD * LGD * EAD, CECL / IFRS 9) vs models
  requiring ranking only (rank-ordered alert queues).
"""

from __future__ import annotations

import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. METRICS: BRIER SCORE, MURPHY DECOMPOSITION, AND ECE
# =====================================================================

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
    MCE = max_b |acc(B_b) - conf(B_b)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n_samples = len(y_true)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        bin_counts.append(bin_size)

        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            diff = np.abs(bin_acc - bin_conf)
            ece += (bin_size / n_samples) * diff
            mce = max(mce, diff)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)

    return float(ece), float(mce), np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


def brier_score_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """
    Murphy (1973) Brier Score 3-Component Decomposition:
    Brier Score = Reliability - Resolution + Uncertainty
    - Reliability (Calibration Error): sum (n_k / N) * (p_k - o_k)^2  [lower is better, 0 is perfect]
    - Resolution: sum (n_k / N) * (o_k - base_rate)^2               [higher is better]
    - Uncertainty: base_rate * (1 - base_rate)                       [inherent dataset difficulty]
    """
    base_rate = float(np.mean(y_true))
    uncertainty = base_rate * (1.0 - base_rate)
    n_samples = len(y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    within_bin_var = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            o_k = np.mean(y_true[in_bin])  # observed fraction
            p_k = np.mean(y_prob[in_bin])  # mean confidence
            reliability += (bin_size / n_samples) * ((p_k - o_k) ** 2)
            resolution += (bin_size / n_samples) * ((o_k - base_rate) ** 2)
            within_bin_var += np.sum((y_prob[in_bin] - p_k) ** 2) / n_samples

    total_brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "brier_score": total_brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "within_bin_var": float(within_bin_var),
        "reconstructed_brier": float(reliability - resolution + uncertainty + within_bin_var),
    }


def unbias_odds_probabilities(probs: np.ndarray, scale_pos_weight: float) -> np.ndarray:
    """
    Inverts the analytical logit shift induced by scale_pos_weight.
    When training with scale_pos_weight = s:
        logit_model = logit_true + ln(s)
        odds_model = odds_true * s
        odds_true = odds_model / s
        p_true = odds_true / (1 + odds_true)
    """
    eps = 1e-15
    p = np.clip(probs, eps, 1.0 - eps)
    odds_model = p / (1.0 - p)
    odds_true = odds_model / scale_pos_weight
    return odds_true / (1.0 + odds_true)


# =====================================================================
# 2. SYNTHETIC REALISTIC BANKING CREDIT / AML DATASET
# =====================================================================

def make_banking_risk_dataset(n_samples: int = 25000, seed: int = 42) -> pd.DataFrame:
    """
    Synthesizes realistic banking credit risk & AML default portfolio dataset.
    N = 25,000 accounts, default rate = 3.5% (875 positive defaults).
    Features reflect real retail underwriting and AML transaction vectors.
    """
    np.random.seed(seed)

    annual_income = np.random.lognormal(10.8, 0.65, n_samples)
    dti_ratio = np.random.beta(2, 5, n_samples) * 0.8
    revolving_utilization = np.random.beta(2, 3, n_samples)
    bureau_score = np.random.normal(680, 55, n_samples).clip(300, 850)
    delinquency_count_2yr = np.random.poisson(0.35, n_samples)
    tx_velocity_1h = np.random.exponential(1.2, n_samples)
    cash_spike_ratio = np.random.beta(1, 8, n_samples) * 4.0
    account_age_months = np.random.uniform(3, 140, n_samples)

    # True latent default log-odds (logistic link)
    log_odds = (
        -4.2
        - 0.55 * (np.log(annual_income) - 10.8)
        + 1.80 * (dti_ratio - 0.28)
        + 2.10 * (revolving_utilization - 0.40)
        - 0.012 * (bureau_score - 680)
        + 0.65 * delinquency_count_2yr
        + 0.45 * tx_velocity_1h
        + 0.75 * cash_spike_ratio
        - 0.015 * (account_age_months - 40)
    )

    true_prob = 1.0 / (1.0 + np.exp(-log_odds))
    target = (np.random.uniform(0, 1, n_samples) < true_prob).astype(int)

    df = pd.DataFrame({
        "annual_income": annual_income,
        "dti_ratio": dti_ratio,
        "revolving_utilization": revolving_utilization,
        "bureau_score": bureau_score,
        "delinquency_count_2yr": delinquency_count_2yr,
        "tx_velocity_1h": tx_velocity_1h,
        "cash_spike_ratio": cash_spike_ratio,
        "account_age_months": account_age_months,
        "true_prob": true_prob,
        "target": target,
    })

    return df


def make_calibrated_classifier(estimator, method: str = "sigmoid"):
    """
    Constructs a CalibratedClassifierCV on a pre-fitted estimator.
    Handles scikit-learn >= 1.4 FrozenEstimator API while retaining backwards compatibility.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(estimator=FrozenEstimator(estimator), method=method)
    except ImportError:
        try:
            from sklearn.calibration import FrozenEstimator
            return CalibratedClassifierCV(estimator=FrozenEstimator(estimator), method=method)
        except ImportError:
            return CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")


# =====================================================================
# 3. COMPLETE 3-WAY SPLIT CALIBRATION WORKFLOW
# =====================================================================

def run_calibration_suite() -> dict:
    """
    Executes rigorous 3-way split calibration benchmark:
    - 60% Train (15,000 samples): Trains base XGBoost model
    - 20% Calibration (5,000 samples): Fits Platt Scaling and Isotonic Regression
    - 20% Test (5,000 samples): Evaluates all models out-of-sample with zero leakage
    """
    print("=" * 80)
    print("XGBOOST PROBABILITY CALIBRATION BENCHMARK (NICULESCU-MIZIL & CARUANA 2005/2007)")
    print("=" * 80)

    df = make_banking_risk_dataset(n_samples=25000, seed=42)
    feature_cols = [c for c in df.columns if c not in ["target", "true_prob"]]
    X = df[feature_cols]
    y = df["target"]

    print(f"\n[Dataset] Total Instances: {len(df):,}")
    print(f"  - Positive Default Rate: {y.mean():.2%} ({y.sum():,} events)")
    print(f"  - Features: {feature_cols}")

    # 3-Way Stratified Split: Train (60%), Calib (20%), Test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"  - Train Set:       {len(X_train):,} samples (Default rate: {y_train.mean():.2%})")
    print(f"  - Calibration Set: {len(X_calib):,} samples (Default rate: {y_calib.mean():.2%})")
    print(f"  - Test Set:        {len(X_test):,} samples (Default rate: {y_test.mean():.2%})")

    # -----------------------------------------------------------------
    # Model 1: Standard XGBoost (Uncalibrated Baseline)
    # -----------------------------------------------------------------
    print("\n[Step 1] Training Base XGBoost Model (Objective: binary:logistic, tree_method='hist')...")
    xgb_base = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
    )
    xgb_base.fit(X_train, y_train)

    probs_test_raw = xgb_base.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------------------
    # Model 2: Platt Scaling (Sigmoid Calibrated on Calibration Set)
    # -----------------------------------------------------------------
    print("[Step 2] Fitting Platt Scaling (method='sigmoid') on Held-Out Calibration Set...")
    calib_platt = make_calibrated_classifier(xgb_base, method="sigmoid")
    calib_platt.fit(X_calib, y_calib)
    probs_test_platt = calib_platt.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------------------
    # Model 3: Isotonic Regression (Non-parametric on Calibration Set)
    # -----------------------------------------------------------------
    print("[Step 3] Fitting Isotonic Regression (method='isotonic') on Held-Out Calibration Set...")
    calib_iso = make_calibrated_classifier(xgb_base, method="isotonic")
    calib_iso.fit(X_calib, y_calib)
    probs_test_iso = calib_iso.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------------------
    # Model 4: scale_pos_weight Model (Imbalanced Distortion Experiment)
    # -----------------------------------------------------------------
    spw_value = float((1.0 - y_train.mean()) / y_train.mean())
    print(f"\n[Step 4] Training XGBoost with scale_pos_weight={spw_value:.1f}x...")
    xgb_spw = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw_value,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
    )
    xgb_spw.fit(X_train, y_train)

    probs_test_spw_raw = xgb_spw.predict_proba(X_test)[:, 1]
    probs_test_spw_unbiased = unbias_odds_probabilities(probs_test_spw_raw, scale_pos_weight=spw_value)

    calib_spw_platt = make_calibrated_classifier(xgb_spw, method="sigmoid")
    calib_spw_platt.fit(X_calib, y_calib)
    probs_test_spw_platt = calib_spw_platt.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------------------
    # Comprehensive Metric Evaluation
    # -----------------------------------------------------------------
    print("\n[Step 5] Evaluating Out-of-Sample Calibration & Discriminative Performance...")

    models = {
        "Raw XGBoost (Uncalibrated)": probs_test_raw,
        "Platt Scaling (Sigmoid)": probs_test_platt,
        "Isotonic Regression": probs_test_iso,
        "Raw scale_pos_weight": probs_test_spw_raw,
        "scale_pos_weight (Odds-Unbiased)": probs_test_spw_unbiased,
        "scale_pos_weight + Platt Scaling": probs_test_spw_platt,
    }

    results = {}
    for name, p in models.items():
        brier = brier_score_loss(y_test, p)
        ece, mce, _, _, _ = calculate_ece(y_test.values, p, n_bins=10)
        ll = log_loss(y_test, p)
        roc = roc_auc_score(y_test, p)
        pr = average_precision_score(y_test, p)
        decomp = brier_score_decomposition(y_test.values, p, n_bins=10)

        # Operational slice metric: Tail calibration for high-risk accounts (p >= 0.08)
        tail_mask = p >= 0.08
        tail_count = int(np.sum(tail_mask))
        if tail_count >= 10:
            tail_brier = float(brier_score_loss(y_test[tail_mask], p[tail_mask]))
            tail_ece, _, _, _, _ = calculate_ece(y_test.values[tail_mask], p[tail_mask], n_bins=5)
            tail_actual_rate = float(y_test[tail_mask].mean())
            tail_pred_mean = float(p[tail_mask].mean())
        else:
            tail_brier, tail_ece, tail_actual_rate, tail_pred_mean = None, None, None, None

        results[name] = {
            "brier_score": float(brier),
            "ece": float(ece),
            "mce": float(mce),
            "log_loss": float(ll),
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "reliability_murphy": float(decomp["reliability"]),
            "resolution_murphy": float(decomp["resolution"]),
            "uncertainty_murphy": float(decomp["uncertainty"]),
            "tail_slice": {
                "threshold": 0.08,
                "n_accounts": tail_count,
                "tail_brier": tail_brier,
                "tail_ece": tail_ece,
                "tail_actual_default_rate": tail_actual_rate,
                "tail_mean_predicted_prob": tail_pred_mean,
            }
        }

    # Print Formatted Table
    print("\n" + "=" * 95)
    print(f"{'Model Architecture':<34}{'Brier':<10}{'ECE':<10}{'LogLoss':<10}{'ROC-AUC':<10}{'PR-AUC':<10}{'Reliability':<12}")
    print("-" * 95)
    for name, m in results.items():
        print(f"{name:<34}{m['brier_score']:<10.5f}{m['ece']:<10.4f}{m['log_loss']:<10.4f}{m['roc_auc']:<10.4f}{m['pr_auc']:<10.4f}{m['reliability_murphy']:<12.6f}")
    print("=" * 95)

    # -----------------------------------------------------------------
    # Step 6: Generate Publication Multi-Panel Diagnostic Visual
    # -----------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, "probability_calibration_benchmark.png")
    json_path = os.path.join(out_dir, "calibration_benchmarks.json")

    print(f"\n[Step 6] Rendering Multi-Panel Reliability Visual to: {os.path.basename(fig_path)}...")
    render_calibration_figure(y_test.values, models, results, fig_path)

    # Save benchmark metrics to JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark ledger to: {os.path.basename(json_path)}")

    return results


def render_calibration_figure(y_test: np.ndarray, models: dict, results: dict, fig_path: str):
    """
    Renders 4-panel publication visualization:
    1. Overall Reliability Curves (Raw vs Platt vs Isotonic)
    2. Predicted Probability Histograms (showing boosting's mass dispersion away from 0 and 1)
    3. Operational Tail Reliability Curve (Zoomed in on high-risk alert queue p >= 0.05)
    4. scale_pos_weight Shift & Unbiasing Reliability Curves
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    colors = {
        "Raw XGBoost (Uncalibrated)": "#e11d48",
        "Platt Scaling (Sigmoid)": "#0284c7",
        "Isotonic Regression": "#059669",
        "Raw scale_pos_weight": "#d97706",
        "scale_pos_weight (Odds-Unbiased)": "#8b5cf6",
        "scale_pos_weight + Platt Scaling": "#06b6d4",
    }

    # -----------------------------------------------------------------
    # Panel 1: Overall Reliability Curves (0 to 1)
    # -----------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", linewidth=1.8, alpha=0.7)

    for name in ["Raw XGBoost (Uncalibrated)", "Platt Scaling (Sigmoid)", "Isotonic Regression"]:
        prob_true, prob_pred = calibration_curve(y_test, models[name], n_bins=10, strategy="uniform")
        ece_val = results[name]["ece"]
        brier_val = results[name]["brier_score"]
        ax1.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2.2,
            label=f"{name} (ECE={ece_val:.3f}, Brier={brier_val:.4f})",
            color=colors[name],
        )

    ax1.set_title("A. Full-Spectrum Reliability Diagram (10 Bins)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax1.set_ylabel("Empirical True Fraction of Positives", fontsize=11)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------------------------------------------
    # Panel 2: Probability Mass Dispersion Histogram
    # -----------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 0.40, 40)
    ax2.hist(
        models["Raw XGBoost (Uncalibrated)"],
        bins=bins,
        alpha=0.45,
        color=colors["Raw XGBoost (Uncalibrated)"],
        label="Raw XGBoost (Pushed away from 0)",
        density=True,
    )
    ax2.hist(
        models["Platt Scaling (Sigmoid)"],
        bins=bins,
        alpha=0.55,
        color=colors["Platt Scaling (Sigmoid)"],
        label="Platt Calibrated (Restored Mass at Tail)",
        density=True,
    )

    ax2.set_title("B. Probability Distribution (Niculescu-Mizil & Caruana 2005)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Predicted Probability (Zoomed: [0, 0.40])", fontsize=11)
    ax2.set_ylabel("Empirical Density", fontsize=11)
    ax2.set_xlim([0, 0.40])
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------------------------------------------
    # Panel 3: High-Risk Alert Queue Region (Operational Focus)
    # -----------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot([0, 0.35], [0, 0.35], "k--", label="Ideal Diagonal", linewidth=1.8, alpha=0.7)

    for name in ["Raw XGBoost (Uncalibrated)", "Platt Scaling (Sigmoid)", "Isotonic Regression"]:
        prob_true, prob_pred = calibration_curve(y_test, models[name], n_bins=12, strategy="quantile")
        # Filter for operational focus
        mask = prob_pred <= 0.35
        ax3.plot(
            prob_pred[mask],
            prob_true[mask],
            marker="s",
            linewidth=2.2,
            label=f"{name}",
            color=colors[name],
        )

    ax3.axvline(0.08, color="#f59e0b", linestyle=":", linewidth=2.0, label="Alert Threshold (p=0.08)")
    ax3.set_title("C. Operational Risk Tail Calibration (Quantile Binned)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Mean Predicted Probability (p <= 0.35)", fontsize=11)
    ax3.set_ylabel("Observed Default Frequency", fontsize=11)
    ax3.set_xlim([0, 0.35])
    ax3.set_ylim([0, 0.35])
    ax3.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------------------------------------------
    # Panel 4: scale_pos_weight Shift & Unbiasing
    # -----------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot([0, 1], [0, 1], "k--", label="Ideal Diagonal", linewidth=1.8, alpha=0.7)

    spw_eval_names = [
        "Raw scale_pos_weight",
        "scale_pos_weight (Odds-Unbiased)",
        "scale_pos_weight + Platt Scaling",
    ]
    for name in spw_eval_names:
        prob_true, prob_pred = calibration_curve(y_test, models[name], n_bins=10, strategy="uniform")
        ece_val = results[name]["ece"]
        ax4.plot(
            prob_pred,
            prob_true,
            marker="^",
            linewidth=2.2,
            label=f"{name} (ECE={ece_val:.3f})",
            color=colors[name],
        )

    ax4.set_title("D. scale_pos_weight Distortion & Odds Unbiasing Recovery", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax4.set_ylabel("Empirical True Fraction of Positives", fontsize=11)
    ax4.set_xlim([-0.02, 1.02])
    ax4.set_ylim([-0.02, 1.02])
    ax4.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Enterprise XGBoost Probability Calibration Architecture\nEmpirical Validation on Held-Out Test Set (N=5,000, 3-Way Split)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    fig.subplots_adjust(top=0.91, bottom=0.06, left=0.07, right=0.96, hspace=0.28, wspace=0.22)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Figure rendered successfully: {os.path.basename(fig_path)}")


if __name__ == "__main__":
    run_calibration_suite()
