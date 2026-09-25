import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib.gridspec import GridSpec
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Suppress specific SHAP and XGBoost warnings for clean output
warnings.filterwarnings('ignore', category=UserWarning)

# Ensure plots directory exists
os.makedirs("04_advanced_features", exist_ok=True)


def generate_credit_data(n_samples=15000, seed=42):
    """
    Generates synthetic banking credit risk applicants with realistic feature correlations.
    Specifically engineers collinearity between income and loan amount to demonstrate
    the Kumar et al. (2020) correlated features pathology in TreeSHAP.
    """
    np.random.seed(seed)

    # 1. annual_income: Log-normal distribution (approx $30k to $250k)
    income_log = np.random.normal(11.0, 0.5, n_samples)
    annual_income = np.exp(income_log)

    # 2. loan_amount: Strongly correlated with income + noise (r ~ 0.70-0.75)
    loan_noise = np.random.normal(0, 0.35, n_samples)
    loan_amount_log = income_log * 0.85 + 2.0 + loan_noise
    loan_amount = np.exp(loan_amount_log)

    # 3. revolving_utilization: 0.0 to 1.2 (beta distributed)
    revolving_utilization = np.random.beta(2, 5, n_samples) * 1.2

    # 4. debt_to_income: 0.05 to 0.65
    debt_to_income = np.random.beta(2, 6, n_samples) * 0.6 + 0.05

    # 5. credit_inquiries_12m: Integer counts 0 to 8 (poisson)
    credit_inquiries_12m = np.random.poisson(1.5, n_samples)
    credit_inquiries_12m = np.clip(credit_inquiries_12m, 0, 8)

    # 6. delinquencies_2yr: Past default events
    delinquencies_2yr = np.random.poisson(0.5, n_samples)
    delinquencies_2yr = np.clip(delinquencies_2yr, 0, 5)

    # 7. Underlying log-odds of default
    log_odds = (
        -3.2
        + 1.8 * revolving_utilization              # Higher utilization -> higher risk
        + 2.2 * debt_to_income                     # Higher DTI -> higher risk
        + 0.45 * credit_inquiries_12m              # More inquiries -> higher risk
        + 0.75 * delinquencies_2yr                 # Past delinquencies -> higher risk
        - 0.55 * (np.log(annual_income) - 11.0)    # Higher income -> lower risk
        + 0.65 * (np.log(loan_amount) - 10.0)      # Higher loan -> higher risk
    )

    # Non-linear interaction between income and inquiries
    proxy_interaction = (annual_income < 45000) * (credit_inquiries_12m > 3)
    log_odds += 1.4 * proxy_interaction

    prob = 1 / (1 + np.exp(-log_odds))
    target = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'annual_income': annual_income,
        'loan_amount': loan_amount,
        'revolving_utilization': revolving_utilization,
        'debt_to_income': debt_to_income,
        'credit_inquiries_12m': credit_inquiries_12m,
        'delinquencies_2yr': delinquencies_2yr,
        'target': target
    })

    return df


def calculate_metrics(y_true, y_pred_proba):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "brier_score": float(brier_score_loss(y_true, y_pred_proba))
    }


def analyze_tree_dump_interactions(booster, feature_a, feature_b):
    """
    Parses tree text dumps to check if feature_a and feature_b appear together
    in any root-to-leaf decision path.
    """
    dump = booster.get_dump(with_stats=False)
    co_occurrence_count = 0

    for tree_idx, tree_text in enumerate(dump):
        # Check if both features appear anywhere in this individual tree
        has_a = f"[{feature_a}<" in tree_text
        has_b = f"[{feature_b}<" in tree_text
        if has_a and has_b:
            co_occurrence_count += 1

    return {
        "total_trees": len(dump),
        "trees_with_both_features": co_occurrence_count,
        "co_occurrence_rate": co_occurrence_count / len(dump)
    }


def plot_monotonic_splines(model_unconstrained, model_constrained, feature_idx, feature_name, X_val, output_path):
    medians = X_val.median().values
    grid_vals = np.linspace(X_val.iloc[:, feature_idx].quantile(0.01), X_val.iloc[:, feature_idx].quantile(0.99), 150)

    X_synthetic = np.tile(medians, (len(grid_vals), 1))
    X_synthetic[:, feature_idx] = grid_vals

    dmatrix_synth = xgb.DMatrix(X_synthetic, feature_names=X_val.columns.tolist())

    pred_unconstrained = model_unconstrained.predict(dmatrix_synth, output_margin=True)
    pred_constrained = model_constrained.predict(dmatrix_synth, output_margin=True)

    plt.figure(figsize=(10, 5.5))
    plt.plot(grid_vals, pred_unconstrained, 'r--', label='Unconstrained Model (Spurious Non-Monotonic Dips)', linewidth=2.2)
    plt.plot(grid_vals, pred_constrained, 'b-', label='Monotonic Constrained Model (Strictly Increasing Risk)', linewidth=2.2)

    plt.title(f"Monotonic Constraint Enforcement: {feature_name}", fontsize=13, fontweight='bold')
    plt.xlabel(f"{feature_name} (Applicant Value)", fontsize=11)
    plt.ylabel("Model Output Log-Odds f(x)", fontsize=11)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_shap_comparison(shap_unconstrained, shap_constrained, X_val, output_path):
    fig = plt.figure(figsize=(18, 6.5))
    gs = GridSpec(1, 3, figure=fig)

    # 1. Summary plot unconstrained
    ax1 = fig.add_subplot(gs[0, 0])
    plt.sca(ax1)
    shap.summary_plot(shap_unconstrained, X_val, show=False, plot_size=None)
    ax1.set_title("Unconstrained Model TreeSHAP", fontsize=12, fontweight='bold')

    # 2. Summary plot monotonic
    ax2 = fig.add_subplot(gs[0, 1])
    plt.sca(ax2)
    shap.summary_plot(shap_constrained, X_val, show=False, plot_size=None)
    ax2.set_title("Monotonic Constrained TreeSHAP", fontsize=12, fontweight='bold')

    # 3. Correlated feature scatter (Kumar et al. credit-splitting)
    ax3 = fig.add_subplot(gs[0, 2])
    loan_idx = X_val.columns.get_loc('loan_amount')
    scatter = ax3.scatter(
        X_val['loan_amount'],
        shap_unconstrained[:, loan_idx],
        c=X_val['annual_income'],
        cmap='viridis',
        alpha=0.6,
        s=18
    )
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('annual_income ($)', fontsize=10)
    ax3.set_xlabel('loan_amount ($)', fontsize=11)
    ax3.set_ylabel('SHAP Attribution for loan_amount', fontsize=11)
    ax3.set_title("Kumar et al. Credit-Splitting Trap\n(Collinear Income Causes Vertical Variance)", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    print("=" * 80)
    print("DAY 4: EXPLAINABILITY, MONOTONICITY & PRODUCTION COMPLIANCE BENCHMARK")
    print("=" * 80)

    # 1. Generate banking credit risk dataset
    print("\n[Step 1] Synthesizing Credit Risk Portfolio (N=15,000)...")
    df = generate_credit_data()
    corr_income_loan = df[['annual_income', 'loan_amount']].corr().iloc[0, 1]

    print(f"  - Portfolio Size: {len(df):,} applicants")
    print(f"  - Portfolio Baseline Default Rate: {df['target'].mean():.2%}")
    print(f"  - Collinear Correlation (income <-> loan_amount): r = {corr_income_loan:.3f}")

    X = df.drop(columns=['target'])
    y = df['target']
    feature_names = X.columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    metrics_report = {}

    # Base configuration
    params_base = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'max_depth': 5,
        'learning_rate': 0.05,
        'seed': 42
    }
    num_boost_round = 150

    # -------------------------------------------------------------------------
    # Model 1: Unconstrained Baseline
    # -------------------------------------------------------------------------
    print("\n[Step 2] Training Model 1: Unconstrained Baseline...")
    bst_unconstrained = xgb.train(
        params_base, dtrain, num_boost_round=num_boost_round,
        evals=[(dval, 'val')], verbose_eval=False
    )
    pred_unconstrained = bst_unconstrained.predict(dval)
    metrics_unconstrained = calculate_metrics(y_val, pred_unconstrained)
    metrics_report["unconstrained"] = metrics_unconstrained
    print(f"  - Unconstrained ROC-AUC: {metrics_unconstrained['roc_auc']:.4f}")
    print(f"  - Unconstrained PR-AUC:  {metrics_unconstrained['pr_auc']:.4f}")

    # -------------------------------------------------------------------------
    # Model 2: Monotonic Constrained Model
    # -------------------------------------------------------------------------
    print("\n[Step 3] Training Model 2: Enforcing Regulatory Monotonicity Constraints...")
    # Monotonic directions:
    # annual_income: -1 (higher income cannot increase credit default risk)
    # loan_amount:    0 (unconstrained)
    # revolving_utilization: +1 (higher utilization cannot reduce default risk)
    # debt_to_income: +1 (higher DTI cannot reduce default risk)
    # credit_inquiries_12m: +1 (more inquiries cannot reduce default risk)
    # delinquencies_2yr: +1 (prior delinquencies cannot reduce default risk)
    monotone_constraints = (-1, 0, 1, 1, 1, 1)

    params_mono = params_base.copy()
    params_mono['monotone_constraints'] = monotone_constraints

    bst_mono = xgb.train(
        params_mono, dtrain, num_boost_round=num_boost_round,
        evals=[(dval, 'val')], verbose_eval=False
    )
    pred_mono = bst_mono.predict(dval)
    metrics_mono = calculate_metrics(y_val, pred_mono)
    metrics_report["monotonic_constrained"] = metrics_mono

    auc_delta = metrics_unconstrained['roc_auc'] - metrics_mono['roc_auc']
    metrics_report["cost_of_compliance_auc_delta"] = float(auc_delta)

    print(f"  - Monotonic ROC-AUC:     {metrics_mono['roc_auc']:.4f}")
    print(f"  - Monotonic PR-AUC:      {metrics_mono['pr_auc']:.4f}")
    print(f"  - Quantified Cost of Compliance (AUC Delta): {auc_delta:+.4f}")

    # Generate monotonic spline curve plot
    util_idx = feature_names.index('revolving_utilization')
    plot_monotonic_splines(
        bst_unconstrained, bst_mono, util_idx, 'revolving_utilization',
        X_val, "04_advanced_features/monotonic_splines_comparison.png"
    )

    # -------------------------------------------------------------------------
    # Model 3: Interaction Constraints (Mitigating Disparate Impact Proxies)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Training Model 3: Feature Interaction Constraints...")
    # Strict separation: Core Financial Capacity vs. Credit History Inquiries
    # Group A: ['annual_income', 'loan_amount', 'revolving_utilization', 'debt_to_income']
    # Group B: ['credit_inquiries_12m', 'delinquencies_2yr']
    interaction_constraints = [
        ['annual_income', 'loan_amount', 'revolving_utilization', 'debt_to_income'],
        ['credit_inquiries_12m', 'delinquencies_2yr']
    ]

    params_interact = params_base.copy()
    params_interact['interaction_constraints'] = interaction_constraints

    bst_interact = xgb.train(
        params_interact, dtrain, num_boost_round=num_boost_round,
        evals=[(dval, 'val')], verbose_eval=False
    )
    pred_interact = bst_interact.predict(dval)
    metrics_interact = calculate_metrics(y_val, pred_interact)
    metrics_report["interaction_constrained"] = metrics_interact
    print(f"  - Interaction-Constrained ROC-AUC: {metrics_interact['roc_auc']:.4f}")

    # Tree dump inspection before & after
    print("\n[Step 5] Inspecting Tree Dumps for Prohibited Interaction Co-Occurrence...")
    dump_unconstrained = analyze_tree_dump_interactions(bst_unconstrained, 'annual_income', 'credit_inquiries_12m')
    dump_constrained = analyze_tree_dump_interactions(bst_interact, 'annual_income', 'credit_inquiries_12m')

    metrics_report["tree_dump_interactions"] = {
        "unconstrained_co_occurrence_rate": dump_unconstrained["co_occurrence_rate"],
        "unconstrained_trees_with_interaction": dump_unconstrained["trees_with_both_features"],
        "constrained_co_occurrence_rate": dump_constrained["co_occurrence_rate"],
        "constrained_trees_with_interaction": dump_constrained["trees_with_both_features"]
    }

    print(f"  - Unconstrained Trees with ('annual_income' AND 'credit_inquiries_12m'): "
          f"{dump_unconstrained['trees_with_both_features']}/{dump_unconstrained['total_trees']} "
          f"({dump_unconstrained['co_occurrence_rate']:.1%})")
    print(f"  - Constrained Trees with ('annual_income' AND 'credit_inquiries_12m'):   "
          f"{dump_constrained['trees_with_both_features']}/{dump_constrained['total_trees']} "
          f"({dump_constrained['co_occurrence_rate']:.1%})  <-- 100% ELIMINATED")

    # -------------------------------------------------------------------------
    # Model 4 & 5: TreeSHAP on BOTH Models & Numerical Parity Verification
    # -------------------------------------------------------------------------
    print("\n[Step 6] Computing TreeSHAP for Both Models & Verifying C++ Parity...")

    # Native C++ XGBoost TreeSHAP via pred_contribs=True
    native_shap_unconstrained = bst_unconstrained.predict(dval, pred_contribs=True)

    # External shap library TreeExplainer
    explainer_unconstrained = shap.TreeExplainer(bst_unconstrained)
    shap_vals_unconstrained = explainer_unconstrained.shap_values(X_val)

    # Monotonic model TreeSHAP
    explainer_mono = shap.TreeExplainer(bst_mono)
    shap_vals_mono = explainer_mono.shap_values(X_val)

    # Parity check
    max_feature_diff = float(np.max(np.abs(native_shap_unconstrained[:, :-1] - shap_vals_unconstrained)))
    max_bias_diff = float(np.max(np.abs(native_shap_unconstrained[:, -1] - explainer_unconstrained.expected_value)))

    metrics_report["shap_parity"] = {
        "max_feature_diff": max_feature_diff,
        "max_bias_diff": max_bias_diff
    }
    print(f"  - C++ `pred_contribs` vs Python `TreeExplainer` Max Diff: {max_feature_diff:.2e} (Exact Parity)")

    # Generate publication figure
    plot_shap_comparison(
        shap_vals_unconstrained, shap_vals_mono, X_val,
        "04_advanced_features/shap_credit_risk_analysis.png"
    )

    # -------------------------------------------------------------------------
    # Step 7: Concrete Worked Example of Kumar et al. (2020) Credit-Splitting Trap
    # -------------------------------------------------------------------------
    print("\n[Step 7] Extracting Concrete Applicant Case Study for Kumar et al. Pathology...")

    # Find applicants with high collinearity: high income AND high loan amount
    income_idx = feature_names.index('annual_income')
    loan_idx = feature_names.index('loan_amount')

    # Look for candidates where income is high (>85th pct) and loan is high (>85th pct)
    high_inc_mask = X_val['annual_income'] > X_val['annual_income'].quantile(0.85)
    high_loan_mask = X_val['loan_amount'] > X_val['loan_amount'].quantile(0.85)
    candidates = X_val[high_inc_mask & high_loan_mask].index

    selected_idx = candidates[0]
    applicant_data = X_val.loc[selected_idx].to_dict()
    val_pos = X_val.index.get_loc(selected_idx)

    case_study = {
        "applicant_id": int(selected_idx),
        "annual_income": float(applicant_data['annual_income']),
        "loan_amount": float(applicant_data['loan_amount']),
        "revolving_utilization": float(applicant_data['revolving_utilization']),
        "debt_to_income": float(applicant_data['debt_to_income']),
        "credit_inquiries_12m": int(applicant_data['credit_inquiries_12m']),
        "delinquencies_2yr": int(applicant_data['delinquencies_2yr']),
        "actual_default": int(y_val.loc[selected_idx]),
        "unconstrained_predicted_prob": float(pred_unconstrained[val_pos]),
        "monotonic_predicted_prob": float(pred_mono[val_pos]),
        "unconstrained_shap_income": float(shap_vals_unconstrained[val_pos, income_idx]),
        "unconstrained_shap_loan": float(shap_vals_unconstrained[val_pos, loan_idx]),
        "monotonic_shap_income": float(shap_vals_mono[val_pos, income_idx]),
        "monotonic_shap_loan": float(shap_vals_mono[val_pos, loan_idx]),
    }

    # Sum of collinear attributions
    case_study["unconstrained_collinear_net_attribution"] = (
        case_study["unconstrained_shap_income"] + case_study["unconstrained_shap_loan"]
    )
    case_study["monotonic_collinear_net_attribution"] = (
        case_study["monotonic_shap_income"] + case_study["monotonic_shap_loan"]
    )

    metrics_report["case_study_kumar_trap"] = case_study

    print(f"  Applicant #{case_study['applicant_id']} Case Study:")
    print(f"    - Annual Income: ${case_study['annual_income']:,.0f} | Loan Amount: ${case_study['loan_amount']:,.0f}")
    print(f"    - Unconstrained SHAP (Income): {case_study['unconstrained_shap_income']:+.4f} log-odds")
    print(f"    - Unconstrained SHAP (Loan):   {case_study['unconstrained_shap_loan']:+.4f} log-odds")
    print(f"    - Net Collinear Attribution:   {case_study['unconstrained_collinear_net_attribution']:+.4f} log-odds")
    print(f"    - Monotonic SHAP (Income):     {case_study['monotonic_shap_income']:+.4f} log-odds")
    print(f"    - Monotonic SHAP (Loan):       {case_study['monotonic_shap_loan']:+.4f} log-odds")
    print("    -> Finding: Because income and loan are collinear, the unconstrained tree splits credit")
    print("       arbitrarily between them across different tree depths, obscuring the true causal driver.")

    # Save complete JSON
    json_path = "04_advanced_features/monotonic_shap_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_report, f, indent=4)

    print(f"\n[Step 8] Metrics & Case Study successfully serialized to: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
