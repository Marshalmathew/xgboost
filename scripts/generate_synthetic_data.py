"""
Centralized Synthetic Dataset Generator for Enterprise XGBoost Mastery
========================================================================

Generates deterministic, production-realistic synthetic datasets for all modules:
1. Credit Risk: Correlated financials, collinearity (Kumar et al.), monotonic risk factors.
2. AML Transaction Fraud: Extreme class imbalance (0.5%), asymmetric costs ($5,000 vs $500).
3. Quantile Sketch: Bimodal Hessian curvature dataset for approximate split benchmarking.
4. Survival AFT: Interval-censored multi-year loan cohorts for CECL/IFRS 9 Expected Credit Loss.

Usage:
------
    uv run python scripts/generate_synthetic_data.py --dataset all --samples 20000 --output-dir data/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_credit_dataset(n_samples: int = 15000, seed: int = 42) -> pd.DataFrame:
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
        + 1.8 * revolving_utilization
        + 2.2 * debt_to_income
        + 0.45 * credit_inquiries_12m
        + 0.75 * delinquencies_2yr
        - 0.55 * (np.log(annual_income) - 11.0)
        + 0.65 * (np.log(loan_amount) - 10.0)
    )

    prob = 1.0 / (1.0 + np.exp(-log_odds))
    target = np.random.binomial(1, prob)

    df = pd.DataFrame({
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "revolving_utilization": revolving_utilization,
        "debt_to_income": debt_to_income,
        "credit_inquiries_12m": credit_inquiries_12m,
        "delinquencies_2yr": delinquencies_2yr,
        "is_default": target,
    })
    return df


def generate_fraud_aml_dataset(n_samples: int = 20000, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic anti-money laundering (AML) transaction dataset with
    extreme class imbalance (~0.5% positive illicit activity) and business cost metadata.
    """
    np.random.seed(seed)

    tx_velocity_1h = np.random.exponential(1.5, n_samples)
    cash_spike_ratio = np.random.beta(1, 10, n_samples) * 5.0
    wire_out_amount = np.random.lognormal(8.0, 1.2, n_samples)
    account_age_months = np.random.uniform(1, 120, n_samples)
    kyc_risk_score = np.random.beta(2, 5, n_samples) * 100.0

    # Underlying log-odds of illicit AML activity
    log_odds = (
        -4.6
        + 0.55 * tx_velocity_1h
        + 0.85 * cash_spike_ratio
        + 0.45 * (np.log(wire_out_amount) - 8.0)
        - 0.02 * (account_age_months - 30.0)
        + 0.03 * (kyc_risk_score - 30.0)
    )

    prob = 1.0 / (1.0 + np.exp(-log_odds))
    target = np.random.binomial(1, prob)

    df = pd.DataFrame({
        "tx_velocity_1h": tx_velocity_1h,
        "cash_spike_ratio": cash_spike_ratio,
        "wire_out_amount": wire_out_amount,
        "account_age_months": account_age_months,
        "kyc_risk_score": kyc_risk_score,
        "is_fraud": target,
    })
    return df


def generate_sketch_dataset(n_samples: int = 4000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a 1D feature distribution with non-uniform, bimodal Hessian curvature.
    Designed to test the Weighted Quantile Sketch against uniform percentiles.
    """
    np.random.seed(seed)
    x = np.random.uniform(0.0, 100.0, n_samples)

    # Uncertainty / high-Hessian zone clustered in [40.0, 60.0]
    high_hessian_mask = (x >= 40.0) & (x <= 60.0)
    hessian = np.where(high_hessian_mask, 0.25, 0.01)

    # Target gradient
    y_true = np.where(x > 50.0, 1.0, 0.0)
    p = 1.0 / (1.0 + np.exp(-(x - 50.0) / 10.0))
    gradient = p - y_true

    df = pd.DataFrame({
        "feature_x": x,
        "gradient": gradient,
        "hessian": hessian,
        "target": y_true,
    })
    return df


def generate_survival_dataset(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generates interval-censored retail loan cohorts for Accelerated Failure Time (AFT)
    survival analysis and lifetime Expected Credit Loss (IFRS 9 / CECL).
    """
    np.random.seed(seed)

    loan_term_months = np.random.choice([36, 60], size=n_samples, p=[0.6, 0.4])
    fico_score = np.random.normal(700, 50, n_samples).clip(500, 850)
    dti = np.random.beta(2, 5, n_samples) * 0.6 + 0.1
    interest_rate = 0.05 + 0.15 * (850 - fico_score) / 350 + np.random.normal(0, 0.01, n_samples)

    # True latent log-survival time (in months)
    # Higher FICO -> longer survival; Higher DTI -> shorter survival
    mu = 4.2 + 0.003 * (fico_score - 700) - 1.2 * (dti - 0.25)
    sigma = 0.4
    latent_time = np.exp(mu + sigma * np.random.normal(0, 1, n_samples))

    # Observation window (up to 36 or 60 months)
    observed_window = loan_term_months.astype(float)
    event_observed = latent_time <= observed_window

    duration = np.where(event_observed, latent_time, observed_window)

    # Interval bounds for XGBoost survival:aft
    # For observed default: [t, t]
    # For right-censored (performing): [t_obs, +inf]
    lower_bound = duration
    upper_bound = np.where(event_observed, duration, np.inf)

    df = pd.DataFrame({
        "fico_score": fico_score,
        "dti": dti,
        "interest_rate": interest_rate,
        "loan_term_months": loan_term_months,
        "duration_months": duration,
        "event_observed": event_observed.astype(int),
        "label_lower_bound": lower_bound,
        "label_upper_bound": upper_bound,
    })
    return df


def generate_uplift_dataset(
    n_samples: int = 10000,
    seed: int = 42,
    observational: bool = False,
    include_hidden_confounder: bool = False,
) -> pd.DataFrame:
    """
    Generates synthetic retail banking uplift dataset for Causal Machine Learning (CATE).
    Features continuous true uplift tau*(x) to evaluate S/T/X-Learners, Qini curves,
    and budget-constrained ROI optimization.

    Supports:
    - RCT variant: Constant treatment assignment (e(x) = 0.15)
    - Observational / Confounded variant: Treatment assignment confounded by customer
      features and an optional hidden confounder ('relationship_manager_quality').
    """
    np.random.seed(seed)

    # 1. Feature Generation
    # account_balance: Log-normal $500 - $150,000
    account_balance = np.exp(np.random.normal(9.5, 0.8, n_samples))
    account_balance = np.clip(account_balance, 500.0, 150000.0)

    # customer_age: Uniform integer 21 - 78
    customer_age = np.random.randint(21, 79, size=n_samples)

    # tenure_months: Uniform integer 3 - 180
    tenure_months = np.random.randint(3, 181, size=n_samples)

    # credit_card_spend_30d: Log-normal $0 - $8,000
    credit_card_spend_30d = np.exp(np.random.normal(6.5, 1.0, n_samples))
    credit_card_spend_30d = np.clip(credit_card_spend_30d, 0.0, 8000.0)

    # web_logins_30d: Poisson(lambda=4), 0 - 30
    web_logins_30d = np.random.poisson(4.0, size=n_samples)
    web_logins_30d = np.clip(web_logins_30d, 0, 30)

    # has_direct_deposit: Bernoulli(0.55)
    has_direct_deposit = np.random.binomial(1, 0.55, size=n_samples)

    # Optional hidden confounder
    rm_quality = np.random.binomial(1, 0.40, size=n_samples)

    # Standardized features for latent response surfaces
    web_logins_z = (web_logins_30d - np.mean(web_logins_30d)) / (np.std(web_logins_30d) + 1e-8)
    balance_z = (account_balance - np.mean(account_balance)) / (np.std(account_balance) + 1e-8)
    tenure_z = (tenure_months - np.mean(tenure_months)) / (np.std(tenure_months) + 1e-8)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    # 2. Continuous Ground-Truth CATE: tau*(x)
    eps = np.random.normal(0, 0.02, n_samples)
    tau_star_raw = (
        0.40 * sigmoid(web_logins_z)
        - 0.30 * sigmoid(balance_z)
        + 0.15 * (has_direct_deposit == 0).astype(float)
        - 0.05
        + eps
    )

    # 3. Potential Outcomes Response Surfaces
    # Base control conversion probability mu_0(x)
    confounder_effect = 0.35 * (rm_quality - 0.40) if include_hidden_confounder else 0.0
    mu_0_latent = -1.5 + 0.60 * balance_z + 0.40 * tenure_z + confounder_effect
    mu_0 = np.clip(sigmoid(mu_0_latent), 0.02, 0.98)

    # Treated conversion probability mu_1(x)
    mu_1 = np.clip(mu_0 + tau_star_raw, 0.01, 0.99)

    # True treatment effect tau_star
    tau_star = mu_1 - mu_0

    # 4. Treatment Assignment Mechanism
    if observational:
        if include_hidden_confounder:
            propensity = sigmoid(-1.2 + 0.80 * balance_z + 1.50 * (rm_quality - 0.40))
        else:
            propensity = sigmoid(-1.2 + 0.80 * balance_z)
        propensity = np.clip(propensity, 0.05, 0.95)
    else:
        propensity = np.full(n_samples, 0.15)

    treatment = np.random.binomial(1, propensity)

    # 5. Observed Outcomes
    y_0 = np.random.binomial(1, mu_0)
    y_1 = np.random.binomial(1, mu_1)
    outcome = np.where(treatment == 1, y_1, y_0)

    data_dict = {
        "account_balance": account_balance,
        "customer_age": customer_age,
        "tenure_months": tenure_months,
        "credit_card_spend_30d": credit_card_spend_30d,
        "web_logins_30d": web_logins_30d,
        "has_direct_deposit": has_direct_deposit,
        "treatment": treatment,
        "outcome": outcome,
        "propensity_score": propensity,
        "tau_true": tau_star,
        "mu_0": mu_0,
        "mu_1": mu_1,
    }

    if include_hidden_confounder:
        data_dict["relationship_manager_quality"] = rm_quality

    df = pd.DataFrame(data_dict)
    return df


def generate_conformal_credit_dataset(
    n_samples: int = 15000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates synthetic retail credit underwriting and heteroskedastic loss dataset
    specifically engineered for Conformal Prediction, Tripartite Triage, CQR, and Mondrian Auditing.

    Key Features:
    - Highly non-linear default boundary with deliberate epistemic ambiguity near decision boundaries.
    - Extreme heteroskedastic loss variance: conditional loss variance scales with loan amount and FICO risk.
    - Demographically realistic FICO credit score tiers (Prime, NearPrime, Subprime) for Mondrian fairness audits.
    """
    np.random.seed(seed)

    # 1. Covariates
    income_log = np.random.normal(11.0, 0.45, n_samples)
    annual_income = np.clip(np.exp(income_log), 25000.0, 300000.0)

    loan_noise = np.random.normal(0, 0.35, n_samples)
    loan_amount_log = income_log * 0.75 + 1.5 + loan_noise
    loan_amount = np.clip(np.exp(loan_amount_log), 2000.0, 60000.0)

    fico_score = np.clip(np.random.normal(690.0, 55.0, n_samples), 450.0, 850.0).round()
    debt_to_income = np.random.beta(2, 5, n_samples) * 0.60 + 0.05
    revolving_utilization = np.clip(np.random.beta(2, 4, n_samples) * 1.2, 0.0, 1.2)
    inquiries_last_6m = np.clip(np.random.poisson(1.2, n_samples), 0, 6)

    # FICO demographic tiers for Mondrian auditing
    fico_tier = np.where(
        fico_score > 720.0,
        "Prime",
        np.where(fico_score >= 660.0, "NearPrime", "Subprime"),
    )

    # 2. Binary Default Target with high ambiguity near FICO 620-680, DTI 0.35-0.45
    logit = (
        -3.5
        + 2.2 * revolving_utilization
        + 1.8 * debt_to_income
        - 0.012 * (fico_score - 650.0)
        + 0.40 * inquiries_last_6m
    )
    prob_default = 1.0 / (1.0 + np.exp(-logit))
    is_default = np.random.binomial(1, prob_default)

    # 3. Continuous Heteroskedastic Loss Given Default (LGD in $)
    # Conditional mean loss depends on loan amount, FICO, and DTI
    loss_fraction_latent = -0.5 - 0.005 * (fico_score - 650.0) + 1.2 * debt_to_income
    loss_fraction_mean = 1.0 / (1.0 + np.exp(-loss_fraction_latent))
    mu_loss = loan_amount * loss_fraction_mean

    # Heteroskedastic conditional standard deviation: increases with lower FICO and higher loan amounts
    sigma_loss = 0.15 * mu_loss + 0.20 * ((850.0 - fico_score) / 200.0) * mu_loss
    loss_noise = np.random.normal(0, 1, n_samples)
    loss_given_default = np.maximum(0.0, mu_loss + sigma_loss * loss_noise)

    df = pd.DataFrame({
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "fico_score": fico_score,
        "debt_to_income": debt_to_income,
        "revolving_utilization": revolving_utilization,
        "inquiries_last_6m": inquiries_last_6m,
        "fico_tier": fico_tier,
        "prob_default_latent": prob_default,
        "is_default": is_default,
        "mu_loss_latent": mu_loss,
        "sigma_loss_latent": sigma_loss,
        "loss_given_default": loss_given_default,
    })
    return df


DATASET_GENERATORS = {
    "credit": generate_credit_dataset,
    "fraud": generate_fraud_aml_dataset,
    "sketch": generate_sketch_dataset,
    "survival": generate_survival_dataset,
    "uplift": generate_uplift_dataset,
    "conformal": generate_conformal_credit_dataset,
}


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic Centralized Synthetic Data Generator for XGBoost Mastery"
    )
    parser.add_argument(
        "--dataset",
        choices=["credit", "fraud", "sketch", "survival", "uplift", "conformal", "all"],
        default="all",
        help="Dataset to generate (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples (defaults: credit=15k, fraud=20k, sketch=4k, survival=10k, uplift=10k)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save generated datasets (default: data)",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = list(DATASET_GENERATORS.keys()) if args.dataset == "all" else [args.dataset]

    print("================================================================================")
    print("XGBOOST MASTERY: CENTRALIZED SYNTHETIC DATA GENERATOR")
    print(f"Target datasets: {targets} | Seed: {args.seed} | Output: {out_dir.resolve()}")
    print("================================================================================")

    for name in targets:
        gen_func = DATASET_GENERATORS[name]
        n_samples = args.samples if args.samples is not None else (
            15000 if name == "credit" else 20000 if name == "fraud" else 4000 if name == "sketch" else 10000
        )
        print(f"Generating '{name}' dataset ({n_samples:,} rows)...", end=" ")
        df = gen_func(n_samples=n_samples, seed=args.seed)
        out_csv = out_dir / f"synthetic_{name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved to {out_csv} ({out_csv.stat().st_size / 1024:.1f} KB)")

    print("Generation complete! All datasets deterministically generated.")


if __name__ == "__main__":
    main()

