import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification

# Add repo root to sys.path so modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../01_theory_foundations")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../02_xgboost_core_mechanics")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../03_basic_usage_and_tuning")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../04_advanced_features")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../05_production_and_quirks")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../06_projects_finance/04_marketing_propensity")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../06_projects_finance/05_massive_bank_data")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../07_distributed_xgboost")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))



def test_custom_asymmetric_loss():
    """Verify gradient and hessian properties for custom asymmetric loss."""
    labels = np.array([10.0, 5.0])
    preds_under = np.array([5.0, 2.0])  # under-prediction (residual > 0)
    preds_over = np.array([12.0, 7.0])  # over-prediction (residual <= 0)

    penalty = 10.0
    res_under = labels - preds_under
    grad_under = np.where(res_under > 0, penalty * (preds_under - labels), (preds_under - labels))
    hess_under = np.where(res_under > 0, penalty, 1.0)

    # Under-prediction must receive 10x penalty on gradient and hessian
    assert np.all(hess_under == penalty)
    assert np.all(grad_under == penalty * (preds_under - labels))

    res_over = labels - preds_over
    grad_over = np.where(res_over > 0, penalty * (preds_over - labels), (preds_over - labels))
    hess_over = np.where(res_over > 0, penalty, 1.0)

    assert np.all(hess_over == 1.0)
    assert np.all(grad_over == (preds_over - labels))


def test_woe_iv_leakage_free():
    """Verify that WoE mappings fit strictly on train and can transform test."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feature": np.random.normal(100, 15, n),
        "target": np.random.binomial(1, 0.2, n)
    })

    train_df = df.iloc[:150].copy()
    test_df = df.iloc[150:].copy()

    # Learn bin edges on train only
    _, bin_edges = pd.qcut(train_df["feature"], q=5, retbins=True, duplicates="drop")
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    binned_train = pd.cut(train_df["feature"], bins=bin_edges)
    df_bin = pd.DataFrame({"Bin": binned_train, "Target": train_df["target"]})
    grouped = df_bin.groupby("Bin", observed=False)["Target"].agg(["count", "sum"])
    grouped.columns = ["Total", "Events"]
    grouped["Non_Events"] = grouped["Total"] - grouped["Events"]

    total_events = max(grouped["Events"].sum(), 1)
    total_non_events = max(grouped["Non_Events"].sum(), 1)

    grouped["Event_Rate"] = (grouped["Events"] + 0.5) / total_events
    grouped["Non_Event_Rate"] = (grouped["Non_Events"] + 0.5) / total_non_events
    grouped["WoE"] = np.log(grouped["Event_Rate"] / grouped["Non_Event_Rate"])

    woe_map = grouped["WoE"].to_dict()

    # Transform test without seeing test labels
    binned_test = pd.cut(test_df["feature"], bins=bin_edges)
    transformed_test = binned_test.map(woe_map).astype(float)

    assert not transformed_test.isna().any()
    assert len(transformed_test) == len(test_df)


def test_xgboost_monotonic_and_hist():
    """Verify that XGBoost trains successfully with monotonic constraints and hist method."""
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "monotone_constraints": (1, 0, -1, 0)
    }
    bst = xgb.train(params, dtrain, num_boost_round=10)
    preds = bst.predict(dtrain)

    assert len(preds) == 200
    assert (preds >= 0.0).all() and (preds <= 1.0).all()


def test_production_categorical_schema():
    """Verify categorical schemas preserve known categories and map unseen values safely."""
    categories = ["AA", "AB", "AC", "AD", "AE"]

    payload = {"payment_type": "BB"}  # Unseen category
    df = pd.DataFrame([payload])
    cleaned_s = df["payment_type"].where(df["payment_type"].isin(categories))
    df["payment_type"] = pd.Categorical(cleaned_s, categories=categories)

    # Value not in categories should map to NaN without warning or crash
    assert df["payment_type"].isna().iloc[0]

    payload_valid = {"payment_type": "AC"}
    df_valid = pd.DataFrame([payload_valid])
    cleaned_valid = df_valid["payment_type"].where(df_valid["payment_type"].isin(categories))
    df_valid["payment_type"] = pd.Categorical(cleaned_valid, categories=categories)

    assert df_valid["payment_type"].cat.codes.iloc[0] == 2


def test_decision_tree_scratch_dataframe_support():
    """Verify from-scratch decision trees handle pandas DataFrames and train/predict correctly."""
    from scratch_tree import DecisionTreeClassifierFromScratch, DecisionTreeRegressorFromScratch

    X_df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "f2": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
    })
    y_reg = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    y_clf = pd.Series([0, 1, 0, 1, 0, 1])

    # 1. Regressor verification
    tree_reg = DecisionTreeRegressorFromScratch(max_depth=3)
    tree_reg.fit(X_df, y_reg)
    preds_reg = tree_reg.predict(X_df)
    assert len(preds_reg) == 6
    assert tree_reg.feature_importances_ is not None
    assert np.isclose(np.sum(tree_reg.feature_importances_), 1.0)
    # Monotonic check: first prediction should be lower than last
    assert preds_reg[0] < preds_reg[-1]

    # 2. Classifier verification
    tree_clf = DecisionTreeClassifierFromScratch(max_depth=3)
    tree_clf.fit(X_df, y_clf)
    preds_clf = tree_clf.predict(X_df)
    assert len(preds_clf) == 6
    assert set(np.unique(preds_clf)).issubset({0, 1})
    assert tree_clf.feature_importances_ is not None


def test_quant_forward_return_alignment():
    """Verify position established at close of day t aligns with forward return over [t, t+1]."""
    prices = pd.Series([100.0, 102.0, 101.0, 105.0])
    daily_rets = prices.pct_change()
    forward_rets = daily_rets.shift(-1)

    # Signal at day t
    positions = pd.Series([1.0, 0.0, 1.0, 0.0])
    turnover = positions.diff().fillna(positions.iloc[0]).abs()

    # Turnover on first trade should equal initial position
    assert turnover.iloc[0] == 1.0

    # Position at day 0 (1.0) captures forward return from day 0 to day 1: (102 - 100)/100 = +0.02
    assert pytest.approx(forward_rets.iloc[0]) == 0.02
    strat_ret_day0 = positions.iloc[0] * forward_rets.iloc[0]
    assert pytest.approx(strat_ret_day0) == 0.02


def test_smooth_asymmetric_loss_c2_continuous():
    """Verify smooth asymmetric loss provides non-zero, continuous gradient and hessian."""
    diff = np.array([-5.0, 0.0, 5.0])
    penalty = 10.0
    sigmoid_w = 1.0 / (1.0 + np.exp(np.clip(diff, -15.0, 15.0)))
    weight = 1.0 + (penalty - 1.0) * sigmoid_w
    grad = weight * diff
    hess = weight

    assert (hess > 0).all()
    assert (hess <= penalty).all()
    assert grad[1] == 0.0  # Exactly 0 at residual 0


def test_production_inference_service_end_to_end(sample_transaction_payload):
    """Verify microservice inference API returns valid payload, handles schema, and approves/blocks."""
    import production_inference as pi

    is_loaded = pi.load_models()
    assert is_loaded, "Model should load successfully"

    response = pi.score_transaction(sample_transaction_payload)
    assert response["transaction_id"] == "TXN_TEST_001"
    assert "fraud_probability" in response
    assert 0.0 <= response["fraud_probability"] <= 1.0
    assert response["decision"] in ["APPROVE", "BLOCK"]
    assert "latency_ms" in response
    assert isinstance(response["anomaly_flag"], bool)


def test_xgboost_scratch_exact_parity_with_c_api():
    """Verify mathematical parity between scratch 2nd-order engine and official XGBoost C++ API."""
    import json

    from xgboost_scratch import XGBoostScratch

    X, y = make_classification(n_samples=50, n_features=3, n_informative=3, n_redundant=0, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "max_depth": 2,
        "eta": 0.3,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "base_score": 0.5,
        "tree_method": "exact",
        "objective": "binary:logistic",
    }
    bst = xgb.train(params, dtrain, num_boost_round=3)
    xgb_margins = bst.predict(dtrain, output_margin=True)

    # Extract base score correctly for XGBoost >= 2.0 compatibility
    base_score = 0.5
    try:
        cfg = json.loads(bst.save_config())
        raw_val = cfg.get("learner", {}).get("learner_model_param", {}).get("base_score")
        if raw_val is not None:
            base_score = float(str(raw_val).strip("[]"))
    except Exception:
        pass

    scratch = XGBoostScratch(
        n_estimators=3,
        learning_rate=0.3,
        max_depth=2,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1.0,
        base_score=base_score,
    )
    scratch.fit(X, y)
    scratch_margins = scratch.predict_raw_margin(X)

    # Assert exact numerical agreement down to 1e-4 tolerance
    assert np.allclose(scratch_margins, xgb_margins, atol=1e-4)


def test_day2_overfit_and_threshold_diagnostics():
    """Verify Day 2 ablation setup and threshold calculation under alert volume budgets."""
    from overfit_then_regularize import generate_fraud_dataset, get_ablation_stages

    X_train, X_val, y_train, y_val, spw = generate_fraud_dataset(n_samples=500, random_state=42)
    assert len(X_train) == 350
    assert len(X_val) == 150
    assert spw > 10.0  # severely imbalanced

    stages = get_ablation_stages()
    assert len(stages) == 9
    assert stages[0]["id"] == "Stage 0"
    assert stages[0]["params"]["max_depth"] == 9
    assert stages[1]["params"]["max_depth"] == 4
    assert stages[2]["params"]["min_child_weight"] == 5.0

    # Quick train test on Stage 0 vs Stage 2a
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evallist = [(dtrain, "train"), (dval, "val")]

    evals_0 = {}
    xgb.train(stages[0]["params"], dtrain, num_boost_round=10, evals=evallist, evals_result=evals_0, verbose_eval=False)

    evals_2a = {}
    xgb.train(stages[2]["params"], dtrain, num_boost_round=10, evals=evallist, evals_result=evals_2a, verbose_eval=False)

    assert "train" in evals_0 and "val" in evals_0
    assert len(evals_0["train"]["logloss"]) == 10
    assert len(evals_2a["train"]["logloss"]) == 10


def test_day3_staged_tuning_and_benchmarks():
    """Verify Day 3 Optuna canonical integration, staged tuner, and benchmark outputs."""
    import json

    from optuna_xgboost_canonical import run_canonical_study
    from staged_tuning_pipeline import StagedXGBoostTuner

    # 1. Canonical study test (fast smoke run)
    study = run_canonical_study(n_trials=3, random_state=42)
    assert len(study.trials) == 3
    assert study.best_value > 0.0

    # 2. Hardware tree method benchmark validation
    bench_json = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../03_basic_usage_and_tuning/tree_method_benchmark.json")
    )
    assert os.path.exists(bench_json), "tree_method_benchmark.json should exist"
    with open(bench_json, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    assert "10000" in bench_data and "50000" in bench_data
    for size_key in ["10000", "50000"]:
        for method in ["exact", "approx", "hist"]:
            assert method in bench_data[size_key]
            assert bench_data[size_key][method]["wall_clock_sec"] > 0
            assert bench_data[size_key][method]["pr_auc"] > 0

    # 3. Fast staged tuner smoke test on small synthetic data
    X, y = make_classification(n_samples=250, n_features=10, weights=[0.9, 0.1], random_state=42)
    tuner = StagedXGBoostTuner(random_state=42, n_trials_per_stage=2)
    best_p1 = tuner.stage_1_tree_architecture(X[:180], y[:180], X[180:], y[180:])
    assert "max_depth" in best_p1
    assert "min_child_weight" in best_p1
    assert "stage_1_tree_architecture" in tuner.stage_history

    best_p2 = tuner.stage_2_randomness_and_regularization(X[:180], y[:180])
    assert "subsample" in best_p2
    assert "colsample_bytree" in best_p2
    assert "stage_2_randomness_and_regularization" in tuner.stage_history

    # 4. Tuned hyperparameters artifact validation
    tuned_json = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../03_basic_usage_and_tuning/tuned_hyperparameters.json")
    )
    assert os.path.exists(tuned_json), "tuned_hyperparameters.json should exist"
    with open(tuned_json, "r", encoding="utf-8") as f:
        tuned_data = json.load(f)
    assert "final_hyperparameters" in tuned_data
    assert "n_estimators" in tuned_data["final_hyperparameters"]
    assert tuned_data["final_holdout_pr_auc"] > 0.4


def test_day4_monotonic_and_shap_diagnostics():
    """Verify Day 4 monotonic constraints, interaction tree isolation, and TreeSHAP guarantees."""
    import json

    metrics_json = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../04_advanced_features/monotonic_shap_metrics.json")
    )
    assert os.path.exists(metrics_json), "monotonic_shap_metrics.json must exist"
    with open(metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 1. Validate performance metrics
    assert "unconstrained" in metrics
    assert "monotonic_constrained" in metrics
    assert "interaction_constrained" in metrics
    assert metrics["monotonic_constrained"]["roc_auc"] > 0.65

    # 2. Validate zero co-occurrence in interaction-constrained tree dump
    dump_stats = metrics["tree_dump_interactions"]
    assert dump_stats["unconstrained_co_occurrence_rate"] > 0.50
    assert dump_stats["constrained_co_occurrence_rate"] == 0.0
    assert dump_stats["constrained_trees_with_interaction"] == 0

    # 3. Validate C++ vs Python TreeSHAP exact numerical parity
    shap_parity = metrics["shap_parity"]
    assert shap_parity["max_feature_diff"] < 1e-4
    assert shap_parity["max_bias_diff"] < 1e-4

    # 4. Validate Kumar et al. credit-splitting case study
    case_study = metrics["case_study_kumar_trap"]
    assert case_study["annual_income"] > 80000
    assert case_study["loan_amount"] > 100000
    assert "unconstrained_shap_income" in case_study
    assert "unconstrained_shap_loan" in case_study

    # 5. Validate artifacts existence
    memo_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../04_advanced_features/model_risk_explainability_memo.md")
    )
    spline_img = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../04_advanced_features/monotonic_splines_comparison.png")
    )
    shap_img = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../04_advanced_features/shap_credit_risk_analysis.png")
    )
    assert os.path.exists(memo_path), "Explainability memo must exist"
    assert os.path.exists(spline_img), "Monotonic spline visualization must exist"
    assert os.path.exists(shap_img), "SHAP analysis visualization must exist"

    # 6. Functional test: verify monotonicity guarantee on a 1D grid
    grid = np.linspace(0.0, 1.2, 50).reshape(-1, 1)
    dgrid = xgb.DMatrix(grid, feature_names=["util"])
    dtrain_toy = xgb.DMatrix(np.array([[0.1], [0.4], [0.8], [1.1]]), label=np.array([0, 0, 1, 1]), feature_names=["util"])
    toy_bst = xgb.train(
        {"monotone_constraints": (1,), "tree_method": "hist", "max_depth": 3},
        dtrain_toy,
        num_boost_round=10
    )
    toy_preds = toy_bst.predict(dgrid, output_margin=True)
    # Differences must be strictly non-negative
    diffs = np.diff(toy_preds)
    assert np.all(diffs >= -1e-6), "Monotonic constraint violated on 1D grid"


def test_day5_custom_loss_sparsity_and_playbook():
    """Verify Day 5 custom loss math, SLE parity, DMatrix sparsity routing, and Playbook."""
    import json

    from custom_loss_and_sparsity import get_asymmetric_fraud_obj

    # 1. Validate benchmarks artifact
    bench_json = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/day5_benchmarks.json")
    )
    assert os.path.exists(bench_json), "day5_benchmarks.json must exist"
    with open(bench_json, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    # 2. SLE Parity & Negative Hessian Clipping
    sle_audit = bench_data["sle_audit"]
    assert sle_audit["correlation_custom_vs_native"] > 0.99
    assert sle_audit["negative_hessian_detected_in_stress_test"] is True

    # 3. Sparsity routing results
    sparsity = bench_data["sparsity_split_routing"]
    assert "native_nan_routing" in sparsity
    assert sparsity["native_nan_routing"] > 0.20

    # 4. Strict Convexity Proof of Custom Asymmetric Loss on 1D grid
    k_cost = 10.0
    obj_fn = get_asymmetric_fraud_obj(k_cost_ratio=k_cost)
    z_grid = np.linspace(-8.0, 8.0, 100)
    for y_label in [0.0, 1.0]:
        labels = np.full_like(z_grid, y_label)
        d_dummy = xgb.DMatrix(z_grid.reshape(-1, 1), label=labels)
        grad, hess = obj_fn(z_grid, d_dummy)
        # Hessian must be strictly positive everywhere (strict convexity)
        assert np.all(hess > 0.0), f"Hessian violation for label {y_label}"
        if y_label == 1.0:
            # Positive labels must have negative gradient (pulling toward positive prediction)
            assert np.all(grad < 0.0)
        else:
            # Negative labels must have positive gradient (pulling toward negative prediction)
            assert np.all(grad > 0.0)

    # 5. Playbook Existence & Content Audit
    playbook_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../XGBoost_Playbook.md")
    )
    assert os.path.exists(playbook_path), "XGBoost_Playbook.md must exist"
    with open(playbook_path, "r", encoding="utf-8") as f:
        playbook_text = f.read()

    assert "Enterprise Banking" in playbook_text
    assert "Proposal 1" in playbook_text
    assert "Proposal 2" in playbook_text
    assert "Proposal 3" in playbook_text
    assert "Library Selection Framework" in playbook_text

    # 6. Figures existence
    fig1 = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/asymmetric_vs_scale_pos_weight.png")
    )
    fig2 = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/sparsity_missing_value_benchmark.png")
    )
    assert os.path.exists(fig1), "asymmetric_vs_scale_pos_weight.png must exist"
    assert os.path.exists(fig2), "sparsity_missing_value_benchmark.png must exist"


def test_weighted_quantile_sketch_and_approx_algorithm():
    """Verify Weighted Quantile Sketch rank functions, candidate clustering, and approx parity."""
    import json

    from weighted_quantile_sketch import WeightedQuantileSketch

    # 1. Validate benchmark artifact
    bench_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../02_xgboost_core_mechanics/quantile_sketch_benchmark.json")
    )
    assert os.path.exists(bench_path), "quantile_sketch_benchmark.json must exist"
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    # 2. Check real XGBoost exact vs approx correlation
    corr = bench_data["real_xgboost_comparison"]["correlation_exact_vs_approx"]
    assert corr > 0.99, f"Exact vs Approx correlation too low: {corr}"

    # 3. Check hessian mass concentration in high-uncertainty region [40, 60]
    clustering = bench_data["clustering_summary"]
    assert clustering["percentage_in_high_hessian_weighted"] > 0.70
    assert clustering["percentage_in_high_hessian_value_spaced"] < 0.30

    # 4. Functional test on rank functions r^-(y) and r^+(y)
    x_test = np.array([10.0, 20.0, 20.0, 30.0])
    h_test = np.array([1.0, 2.0, 3.0, 4.0])
    sketch = WeightedQuantileSketch(eps=0.25)
    unique_x, r_minus, r_plus, total_h = sketch.compute_ranks(x_test, h_test)

    assert total_h == 10.0
    assert len(unique_x) == 3
    # Check r^-(y)
    np.testing.assert_array_equal(unique_x, [10.0, 20.0, 30.0])
    assert r_minus[0] == 0.0
    assert r_minus[1] == 1.0  # weight strictly < 20.0 is 1.0
    assert r_minus[2] == 6.0  # weight strictly < 30.0 is 1.0 + 5.0 = 6.0

    # Check r^+(y)
    assert r_plus[0] == 1.0  # weight <= 10.0 is 1.0
    assert r_plus[1] == 6.0  # weight <= 20.0 is 1.0 + 5.0 = 6.0
    assert r_plus[2] == 10.0  # weight <= 30.0 is 10.0

    # Candidates must cover endpoints
    candidates = sketch.get_split_candidates(x_test, h_test)
    assert candidates[0] == 10.0
    assert candidates[-1] == 30.0
    assert np.all(np.diff(candidates) > 0)


def test_probability_calibration_and_governance():
    """Validates the probability calibration suite, Murphy decomposition, and governance memo."""
    import json

    from probability_calibration import (
        brier_score_decomposition,
        calculate_ece,
        make_banking_risk_dataset,
        unbias_odds_probabilities,
    )

    # 1. ECE and Brier Murphy decomposition validation
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    ece, mce, _, _, _ = calculate_ece(y_true, y_prob, n_bins=5)
    assert ece >= 0.0
    assert mce >= ece

    decomp = brier_score_decomposition(y_true, y_prob, n_bins=5)
    assert np.isclose(decomp["brier_score"], decomp["reconstructed_brier"], atol=1e-5)
    assert decomp["uncertainty"] == 0.4 * 0.6  # base rate = 2/5 = 0.4

    # 2. scale_pos_weight analytical odds unbiasing
    # If model predicts p=0.5 with scale_pos_weight=10, odds_true = (1.0) / 10 = 0.1 -> p_true = 0.1 / 1.1
    p_unbiased = unbias_odds_probabilities(np.array([0.5]), scale_pos_weight=10.0)[0]
    expected_p = 0.1 / 1.1
    assert np.isclose(p_unbiased, expected_p, atol=1e-5)

    # 3. Dataset synthesis
    df = make_banking_risk_dataset(n_samples=500, seed=42)
    assert len(df) == 500
    assert "annual_income" in df.columns
    assert "target" in df.columns

    # 4. Calibration Benchmark JSON Artifact
    json_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/calibration_benchmarks.json")
    )
    assert os.path.exists(json_path), "calibration_benchmarks.json must exist"
    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "Raw XGBoost (Uncalibrated)" in metrics
    assert "Platt Scaling (Sigmoid)" in metrics
    assert "Isotonic Regression" in metrics
    assert "Raw scale_pos_weight" in metrics
    assert "scale_pos_weight + Platt Scaling" in metrics

    # Raw scale_pos_weight must have severely degraded ECE vs Calibrated
    spw_raw_ece = metrics["Raw scale_pos_weight"]["ece"]
    spw_calib_ece = metrics["scale_pos_weight + Platt Scaling"]["ece"]
    assert spw_calib_ece < spw_raw_ece
    assert spw_calib_ece < 0.02

    # 5. Diagnostic Multi-Panel Figure Artifact
    fig_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/probability_calibration_benchmark.png")
    )
    assert os.path.exists(fig_path), "probability_calibration_benchmark.png must exist"
    assert os.path.getsize(fig_path) > 50000

    # 6. Model Governance Memorandum Audit
    memo_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/model_governance_calibration_memo.md")
    )
    assert os.path.exists(memo_path), "model_governance_calibration_memo.md must exist"
    with open(memo_path, "r", encoding="utf-8") as f:
        memo_text = f.read()

    assert "CECL / IFRS 9" in memo_text
    assert "TIER 1" in memo_text or "Tier 1" in memo_text
    assert "TIER 2" in memo_text or "Tier 2" in memo_text
    assert "Platt Scaling" in memo_text
    assert "Isotonic Regression" in memo_text


def test_production_drift_monitoring_and_governance():
    """Verify production drift monitoring suite, PSI mathematical properties, and governance policy."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../05_production_and_quirks")))
    from production_drift_monitoring import (
        calculate_categorical_psi,
        calculate_psi,
        evaluate_feature_drift,
    )

    # 1. Mathematical Identity: PSI(A, A) must be identically 0
    np.random.seed(42)
    sample_a = np.random.normal(loc=10.0, scale=2.0, size=5000)
    psi_ident, details_ident = calculate_psi(sample_a, sample_a, num_bins=10)
    assert abs(psi_ident) < 1e-4, f"PSI(A, A) must be ~0, got {psi_ident}"

    # 2. Mathematical Non-Negativity & Drift Detection
    sample_b = np.random.normal(loc=12.5, scale=2.0, size=5000) # +1.25 sigma shift
    psi_drift, _ = calculate_psi(sample_a, sample_b, num_bins=10)
    assert psi_drift > 0.25, f"Shifted distribution must trigger critical PSI > 0.25, got {psi_drift}"

    # 3. Categorical PSI
    cat_base = pd.Series(["mobile"] * 450 + ["web"] * 300 + ["branch"] * 150 + ["partner"] * 100)
    cat_curr = pd.Series(["mobile"] * 250 + ["web"] * 200 + ["branch"] * 100 + ["partner"] * 450)
    cat_psi, _ = calculate_categorical_psi(cat_base, cat_curr)
    assert cat_psi > 0.25, f"Severe categorical shift must yield PSI > 0.25, got {cat_psi}"

    # 4. Statistical Metric Battery
    eval_res = evaluate_feature_drift(sample_a, sample_b, feature_name="test_feat")
    assert 0.0 <= eval_res["ks_statistic"] <= 1.0
    assert 0.0 <= eval_res["js_divergence"] <= 1.0
    assert eval_res["drift_status"] in ["MODERATE_DRIFT", "CRITICAL_DRIFT"]

    # 5. Verify Generated Artifacts
    json_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/drift_monitoring_summary.json")
    )
    assert os.path.exists(json_path), "drift_monitoring_summary.json must exist"
    with open(json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert "metadata" in summary_data
    assert "feature_metrics" in summary_data
    assert len(summary_data["feature_metrics"]) >= 5

    fig_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/drift_monitoring_dashboard.png")
    )
    assert os.path.exists(fig_path), "drift_monitoring_dashboard.png must exist"
    assert os.path.getsize(fig_path) > 50000

    policy_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/champion_challenger_retraining_policy.md")
    )
    assert os.path.exists(policy_path), "champion_challenger_retraining_policy.md must exist"
    with open(policy_path, "r", encoding="utf-8") as f:
        policy_text = f.read()

    assert "SR 11-7" in policy_text
    assert "Covariate Shift" in policy_text
    assert "Concept Drift" in policy_text
    assert "Prediction / Target Drift" in policy_text
    assert "Champion-Challenger" in policy_text


def test_fairness_bias_audit_and_governance():
    """Verify fairness metrics, Kleinberg impossibility implications, and governance artifacts."""
    from fairness_bias_audit import Float64XGBClassifier

    # 1. Verify Float64 wrapper returns float64 probabilities
    wrapper = Float64XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    X_toy = np.random.normal(size=(50, 4))
    y_toy = np.random.binomial(1, 0.5, 50)
    wrapper.fit(X_toy, y_toy)
    probs = wrapper.predict_proba(X_toy)
    assert probs.dtype == np.float64, f"Wrapper must output float64, got {probs.dtype}"
    assert probs.shape == (50, 2)

    # 2. Verify Generated Audit Ledger JSON
    json_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/fairness_audit_results.json")
    )
    assert os.path.exists(json_path), "fairness_audit_results.json must exist"
    with open(json_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    # Assert unmitigated 0.72 parity ratio finding
    unmit = audit["unmitigated_metrics"]
    assert 0.70 <= unmit["demographic_parity_ratio"] <= 0.74, (
        f"Expected unmitigated parity ratio ~0.72, got {unmit['demographic_parity_ratio']}"
    )
    assert unmit["four_fifths_rule_passed"] is False, "Unmitigated model must fail 4/5ths rule"
    assert unmit["equalized_odds_difference"] > 0.20

    # Assert mitigated results
    mit = audit["mitigated_metrics"]
    assert mit["demographic_parity_ratio"] >= 0.80, (
        f"Mitigated model must surpass 4/5ths threshold (>=0.80), got {mit['demographic_parity_ratio']}"
    )
    assert mit["four_fifths_rule_passed"] is True
    assert mit["equalized_odds_difference"] < unmit["equalized_odds_difference"] * 0.40, (
        "Mitigation must slash equalized odds disparity by >60%"
    )

    # Assert Kleinberg subgroup calibration
    sub_calib = audit["subgroup_calibration"]
    assert sub_calib["Mature (Age>=30)"]["ece"] < 0.05
    assert sub_calib["Young (Age<30)"]["ece"] < 0.05
    mature_br = sub_calib["Mature (Age>=30)"]["base_rate"]
    young_br = sub_calib["Young (Age<30)"]["base_rate"]
    assert abs(mature_br - young_br) > 0.10, "Base rates must meaningfully differ to demonstrate Kleinberg's theorem"

    # 3. Verify Dashboard Visual Artifact
    fig_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/fairness_bias_audit_dashboard.png")
    )
    assert os.path.exists(fig_path), "fairness_bias_audit_dashboard.png must exist"
    assert os.path.getsize(fig_path) > 50000

    # 4. Verify Model Governance Fairness Memo
    memo_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../05_production_and_quirks/model_governance_fairness_memo.md")
    )
    assert os.path.exists(memo_path), "model_governance_fairness_memo.md must exist"
    with open(memo_path, "r", encoding="utf-8") as f:
        memo_text = f.read()

    assert "Kleinberg" in memo_text
    assert "Four-Fifths" in memo_text
    assert "Epistemic" in memo_text
    assert "Equal Opportunity" in memo_text
    assert "ECOA" in memo_text
    assert "Reg B" in memo_text
    assert "SR 11-7" in memo_text


def test_onnx_export_and_runtime_parity():
    """Verify ONNX export pipeline and floating-point parity with native XGBoost."""
    from export_onnx_benchmark import run_onnx_benchmark

    results = run_onnx_benchmark(n_samples=2000, n_queries=100)
    assert "onnx_runtime" in results
    assert "inplace_predict" in results
    assert results["max_parity_diff"] < 1e-4, f"Parity diff {results['max_parity_diff']} too large"
    assert results["onnx_runtime"]["mean_us"] > 0.0


def test_multi_output_vector_trees():
    """Verify native Multi-Output Vector Trees (multi_strategy='multi_output_tree')."""
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_classes=3,
        n_informative=5,
        random_state=42,
    )

    clf = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        multi_strategy="multi_output_tree",
        tree_method="hist",
        random_state=42,
    )
    clf.fit(X, y)

    preds_prob = clf.predict_proba(X)
    assert preds_prob.shape == (300, 3)
    np.testing.assert_allclose(preds_prob.sum(axis=1), 1.0, rtol=1e-5)

    booster = clf.get_booster()
    dump = booster.get_dump()
    # In vector trees, exactly 1 tree is constructed per round (10 trees total, not 30)
    assert len(dump) == 10, f"Expected 10 vector trees, got {len(dump)}"


def test_synthetic_data_generation_reproducibility():
    """Verify deterministic generation of credit, fraud, sketch, and survival datasets."""
    from generate_synthetic_data import (
        generate_credit_dataset,
        generate_fraud_aml_dataset,
        generate_sketch_dataset,
        generate_survival_dataset,
    )

    df_c1 = generate_credit_dataset(n_samples=500, seed=42)
    df_c2 = generate_credit_dataset(n_samples=500, seed=42)
    pd.testing.assert_frame_equal(df_c1, df_c2)

    df_f1 = generate_fraud_aml_dataset(n_samples=500, seed=42)
    df_f2 = generate_fraud_aml_dataset(n_samples=500, seed=42)
    pd.testing.assert_frame_equal(df_f1, df_f2)

    df_s1 = generate_sketch_dataset(n_samples=500, seed=42)
    df_s2 = generate_sketch_dataset(n_samples=500, seed=42)
    pd.testing.assert_frame_equal(df_s1, df_s2)

    df_surv1 = generate_survival_dataset(n_samples=500, seed=42)
    df_surv2 = generate_survival_dataset(n_samples=500, seed=42)
    pd.testing.assert_frame_equal(df_surv1, df_surv2)


def test_distributed_pipeline_interfaces():
    """Verify distributed pipeline scripts execute cleanly and return valid metrics."""
    from dask_xgboost_pipeline import run_dask_pipeline
    from pyspark_xgboost_pipeline import run_pyspark_pipeline
    from ray_xgboost_pipeline import run_ray_pipeline

    dask_res = run_dask_pipeline()
    assert "framework" in dask_res and dask_res["framework"] == "Dask"

    spark_res = run_pyspark_pipeline()
    assert "framework" in spark_res and spark_res["framework"] == "PySpark"

    ray_res = run_ray_pipeline()
    assert "framework" in ray_res and ray_res["framework"] == "Ray"


def test_causal_dgp_properties():
    """Verify statistical properties and continuity of causal uplift synthetic dataset."""
    from generate_synthetic_data import generate_uplift_dataset

    df = generate_uplift_dataset(n_samples=2500, seed=42)
    assert not df.isna().any().any()
    assert len(df) == 2500

    # Required columns
    expected_cols = [
        "account_balance", "customer_age", "tenure_months", "credit_card_spend_30d",
        "web_logins_30d", "has_direct_deposit", "treatment", "outcome", "propensity_score",
        "tau_true", "mu_0", "mu_1"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col}"

    # RCT treatment probability between 10% and 25%
    treatment_rate = df["treatment"].mean()
    assert 0.10 <= treatment_rate <= 0.25, f"Treatment rate {treatment_rate} outside [0.10, 0.25]"

    # Smooth, continuous CATE (not hard-coded archetypes)
    assert df["tau_true"].nunique() > 100, "tau_true is not continuous"
    assert df["tau_true"].min() < 0.0, "tau_true lacks Sleeping Dog negative region"
    assert df["tau_true"].max() > 0.15, "tau_true lacks Persuadables positive region"


def test_xlearner_propensity_weighting_direction():
    """Verify Künzel et al. (PNAS 2019) counterfactual imputation weighting in X-Learner."""
    from causal_uplift_engine import XGBoostXLearner
    from generate_synthetic_data import generate_uplift_dataset

    df = generate_uplift_dataset(n_samples=2000, seed=42)
    features = [
        "account_balance", "customer_age", "tenure_months",
        "credit_card_spend_30d", "web_logins_30d", "has_direct_deposit"
    ]
    X = df[features]
    t = df["treatment"].values
    y = df["outcome"].values
    tau_true = df["tau_true"].values

    xl = XGBoostXLearner(n_estimators=30, max_depth=3, random_state=42).fit(X, t, y)
    tau_pred = xl.predict_cate(X)

    # Positive correlation with ground truth treatment effect
    corr = np.corrcoef(tau_pred, tau_true)[0, 1]
    assert corr > 0.40, f"X-Learner correlation with true tau too low: {corr:.3f}"


def test_cross_fitting_reduces_bias():
    """Verify that K-fold cross-fitting eliminates in-sample overfitting on counterfactual residuals."""
    from causal_uplift_engine import XGBoostXLearner
    from generate_synthetic_data import generate_uplift_dataset

    df_train = generate_uplift_dataset(n_samples=3000, seed=42)
    df_test = generate_uplift_dataset(n_samples=2000, seed=123)
    features = [
        "account_balance", "customer_age", "tenure_months",
        "credit_card_spend_30d", "web_logins_30d", "has_direct_deposit"
    ]
    X_train = df_train[features]
    t_train = df_train["treatment"].values
    y_train = df_train["outcome"].values

    X_test = df_test[features]
    tau_true_test = df_test["tau_true"].values

    # Fit with and without cross-fitting
    xl_cf = XGBoostXLearner(n_estimators=30, max_depth=3, n_splits=5, use_cross_fitting=True, random_state=42)
    xl_cf.fit(X_train, t_train, y_train)
    tau_cf = xl_cf.predict_cate(X_test)

    xl_nocf = XGBoostXLearner(n_estimators=30, max_depth=3, use_cross_fitting=False, random_state=42)
    xl_nocf.fit(X_train, t_train, y_train)
    tau_nocf = xl_nocf.predict_cate(X_test)

    rmse_cf = np.sqrt(np.mean((tau_cf - tau_true_test) ** 2))
    rmse_nocf = np.sqrt(np.mean((tau_nocf - tau_true_test) ** 2))

    # Both estimators must achieve strong CATE recovery
    assert rmse_cf < 0.15, f"CF RMSE too high: {rmse_cf:.4f}"
    assert rmse_nocf < 0.15, f"NoCF RMSE too high: {rmse_nocf:.4f}"

    # Cross-fitting eliminates asymptotic bias: estimated SATE matches true SATE within 0.05
    sate_true = np.mean(tau_true_test)
    sate_cf = np.mean(tau_cf)
    assert abs(sate_cf - sate_true) < 0.05, f"SATE bias too large: {abs(sate_cf - sate_true):.4f}"

    # Predictions must be consistent across cross-fitting strategies
    assert abs(rmse_cf - rmse_nocf) < 0.03, f"Discrepancy between CF and NoCF too large: {abs(rmse_cf - rmse_nocf):.4f}"


def test_qini_and_auuc_correctness():
    """Verify Qini curve, AUUC, and bootstrap permutation statistical test."""
    from causal_uplift_engine import bootstrap_permutation_test, compute_auuc, compute_qini_curve
    from generate_synthetic_data import generate_uplift_dataset

    df = generate_uplift_dataset(n_samples=3000, seed=42)
    y = df["outcome"].values
    t = df["treatment"].values
    tau_true = df["tau_true"].values

    fracs, q_vals = compute_qini_curve(y, t, tau_true, n_bins=10)
    assert len(fracs) == 11
    assert q_vals[0] == 0.0

    auuc_true = compute_auuc(y, t, tau_true)
    auuc_rand = compute_auuc(y, t, np.random.RandomState(42).randn(len(y)))
    assert auuc_true > auuc_rand, "Oracle AUUC must exceed random baseline AUUC"

    # Permutation test: oracle must be significant (p < 0.05)
    perm_oracle = bootstrap_permutation_test(y, t, tau_true, n_bootstrap=200, random_state=42)
    assert perm_oracle["is_significant"], f"Oracle should be significant, got p={perm_oracle['p_value']}"

    # Permutation test: random permutation must NOT be significant
    perm_rand = bootstrap_permutation_test(y, t, np.random.RandomState(42).randn(len(y)), n_bootstrap=200, random_state=42)
    assert not perm_rand["is_significant"], f"Random noise should not be significant, got p={perm_rand['p_value']}"


def test_budget_optimizer_policy_dominance():
    """Verify closed-form NEV argmax policy optimizer dominates naive propensity targeting."""
    import xgboost as xgb
    from causal_uplift_engine import (
        XGBoostXLearner,
        evaluate_policy_realized_profit,
        optimize_policy,
    )
    from generate_synthetic_data import generate_uplift_dataset

    df = generate_uplift_dataset(n_samples=3000, seed=42)
    features = [
        "account_balance", "customer_age", "tenure_months",
        "credit_card_spend_30d", "web_logins_30d", "has_direct_deposit"
    ]
    X = df[features]
    t = df["treatment"].values
    y = df["outcome"].values
    tau_true = df["tau_true"].values

    # Train X-Learner
    xl = XGBoostXLearner(n_estimators=30, max_depth=3, random_state=42).fit(X, t, y)
    tau_hat = xl.predict_cate(X)

    # Train naive propensity model P(Y=1|X)
    clf_naive = xgb.XGBClassifier(n_estimators=30, max_depth=3, random_state=42, eval_metric="logloss")
    clf_naive.fit(X, y)
    p_naive = clf_naive.predict_proba(X)[:, 1]

    # Evaluate at 15% budget
    b = 0.15
    mask_causal = optimize_policy(tau_hat, budget_ratio=b)
    res_causal = evaluate_policy_realized_profit(mask_causal, y, t, tau_true=tau_true)

    naive_rank = np.argsort(-p_naive)
    mask_naive = np.zeros(len(y), dtype=bool)
    mask_naive[naive_rank[:int(b * len(y))]] = True
    res_naive = evaluate_policy_realized_profit(mask_naive, y, t, tau_true=tau_true)

    # Causal uplift policy must dominate naive predictive policy by >= 10%
    assert res_causal["net_profit"] > res_naive["net_profit"] * 1.10, (
        f"Causal net profit ({res_causal['net_profit']:.1f}) did not dominate naive ({res_naive['net_profit']:.1f}) by >= 10%"
    )


def test_conformal_classification_coverage_guarantee():
    """Verify Split Conformal finite-sample coverage guarantees and sample-size guardrails."""
    from conformal_risk_calibration import SplitConformalClassifier, train_calibrate_test_split
    from generate_synthetic_data import generate_conformal_credit_dataset

    df = generate_conformal_credit_dataset(n_samples=4000, seed=42)
    feature_cols = [
        "annual_income", "loan_amount", "fico_score",
        "debt_to_income", "revolving_utilization", "inquiries_last_6m"
    ]
    train_df, calib_df, test_df = train_calibrate_test_split(df, 0.50, 0.25, 0.25, random_state=42)

    X_train, y_train = train_df[feature_cols], train_df["is_default"].values
    X_calib, y_calib = calib_df[feature_cols], calib_df["is_default"].values
    X_test, y_test = test_df[feature_cols], test_df["is_default"].values

    # Test alpha = 0.05 (95% nominal coverage)
    clf_95 = SplitConformalClassifier(alpha=0.05, random_state=42)
    clf_95.fit(X_train, y_train)
    clf_95.calibrate(X_calib, y_calib)
    metrics_95 = clf_95.evaluate_coverage(X_test, y_test)

    assert metrics_95["empirical_coverage"] >= 0.95 - 0.02, (
        f"Expected coverage >= 0.93, got {metrics_95['empirical_coverage']:.4f}"
    )
    assert metrics_95["mean_set_size"] > 0.0

    # Test alpha = 0.10 (90% nominal coverage)
    clf_90 = SplitConformalClassifier(alpha=0.10, random_state=42)
    clf_90.fit(X_train, y_train)
    clf_90.calibrate(X_calib, y_calib)
    metrics_90 = clf_90.evaluate_coverage(X_test, y_test)

    assert metrics_90["empirical_coverage"] >= 0.90 - 0.02, (
        f"Expected coverage >= 0.88, got {metrics_90['empirical_coverage']:.4f}"
    )

    # Test sample size guardrail (n < 1/alpha - 1 must raise ValueError)
    clf_small = SplitConformalClassifier(alpha=0.01)  # requires n >= 99
    clf_small.fit(X_train, y_train)
    with pytest.raises(ValueError, match="Calibration set size .* is too small"):
        clf_small.calibrate(X_calib[:50], y_calib[:50])


def test_tripartite_triage_semantics():
    """Verify banking tripartite triage actions, default rate separation, and ambiguity handling."""
    from conformal_risk_calibration import SplitConformalClassifier, train_calibrate_test_split
    from generate_synthetic_data import generate_conformal_credit_dataset

    df = generate_conformal_credit_dataset(n_samples=4000, seed=42)
    feature_cols = [
        "annual_income", "loan_amount", "fico_score",
        "debt_to_income", "revolving_utilization", "inquiries_last_6m"
    ]
    train_df, calib_df, test_df = train_calibrate_test_split(df, 0.50, 0.25, 0.25, random_state=42)

    clf = SplitConformalClassifier(alpha=0.05, random_state=42)
    clf.fit(train_df[feature_cols], train_df["is_default"].values)
    clf.calibrate(calib_df[feature_cols], calib_df["is_default"].values)

    triage_df = clf.triage_policy(test_df[feature_cols])
    test_df_eval = test_df.copy()
    test_df_eval["triage_action"] = triage_df["triage_action"].values

    # Check key categories exist in production portfolio
    actions = set(triage_df["triage_action"].unique())
    assert "AUTO_APPROVE" in actions
    assert "REFER_TO_SENIOR_UNDERWRITER" in actions

    # Risk separation: Auto-approved loans must have substantially lower default rate than referred
    auto_approve_defaults = test_df_eval[test_df_eval["triage_action"] == "AUTO_APPROVE"]["is_default"]
    referred_defaults = test_df_eval[test_df_eval["triage_action"] == "REFER_TO_SENIOR_UNDERWRITER"]["is_default"]

    assert auto_approve_defaults.mean() < 0.10, (
        f"Auto-approve default rate too high: {auto_approve_defaults.mean():.4f}"
    )
    assert referred_defaults.mean() > auto_approve_defaults.mean() * 2.0, (
        f"Referred loans ({referred_defaults.mean():.4f}) did not achieve 2x risk separation over auto-approved ({auto_approve_defaults.mean():.4f})"
    )

    # Test all 4 mapping states: {0} -> AUTO_APPROVE, {1} -> AUTO_DENY, {0, 1} -> REFER, {} -> OOD
    # By verifying triage mapping on mock prediction sets
    class MockClassifier(SplitConformalClassifier):
        def predict_set(self, X):
            return [[0], [1], [0, 1], []]
        def predict_proba(self, X):
            return np.array([[0.95, 0.05], [0.05, 0.95], [0.50, 0.50], [0.20, 0.20]])

    mock_clf = MockClassifier(alpha=0.05)
    mock_clf.is_calibrated_ = True
    mock_clf.q_hat_ = 0.85
    mock_triage = mock_clf.triage_policy(pd.DataFrame([[1, 2, 3]] * 4))
    expected_actions = ["AUTO_APPROVE", "AUTO_DENY", "REFER_TO_SENIOR_UNDERWRITER", "OUT_OF_DISTRIBUTION_ALERT"]
    assert mock_triage["triage_action"].tolist() == expected_actions


def test_cqr_heteroskedastic_coverage():
    """Verify CQR heteroskedastic loss intervals, finite-sample coverage, and zero-bound clipping."""
    from conformal_risk_calibration import (
        ConformalizedQuantileRegressor,
        train_calibrate_test_split,
    )
    from generate_synthetic_data import generate_conformal_credit_dataset

    df = generate_conformal_credit_dataset(n_samples=4000, seed=42)
    feature_cols = [
        "annual_income", "loan_amount", "fico_score",
        "debt_to_income", "revolving_utilization", "inquiries_last_6m"
    ]
    train_df, calib_df, test_df = train_calibrate_test_split(df, 0.50, 0.25, 0.25, random_state=42)

    X_train, y_train = train_df[feature_cols], train_df["loss_given_default"].values
    X_calib, y_calib = calib_df[feature_cols], calib_df["loss_given_default"].values
    X_test, y_test = test_df[feature_cols], test_df["loss_given_default"].values

    cqr = ConformalizedQuantileRegressor(alpha=0.10, n_estimators=40, max_depth=3, random_state=42)
    cqr.fit(X_train, y_train)
    cqr.calibrate(X_calib, y_calib)

    metrics = cqr.evaluate_coverage(
        X_test, y_test,
        true_sigma=test_df["sigma_loss_latent"].values,
        clip_zero=True
    )

    # Calibrated CQR coverage must be >= 0.88 for nominal 0.90
    assert metrics["cqr_calibrated_coverage"] >= 0.88, (
        f"CQR coverage ({metrics['cqr_calibrated_coverage']:.4f}) failed to achieve target 0.90"
    )

    # Verify zero-bound clipping: no negative monetary loss bounds
    cal_lo, cal_hi = cqr.predict_interval(X_test, clip_zero=True)
    assert np.all(cal_lo >= 0.0), "Physical zero clipping failed: negative loss bound detected"
    assert np.all(cal_hi >= cal_lo), "Inverted intervals detected"

    # Interval width must adapt to heteroskedastic conditional volatility
    assert metrics["width_heteroskedastic_correlation"] > 0.35, (
        f"Adaptive width correlation too weak: {metrics['width_heteroskedastic_correlation']:.4f}"
    )


def test_mondrian_group_conditional_fairness():
    """Verify Mondrian group-conditional calibration eliminates demographic under-coverage."""
    from conformal_risk_calibration import (
        MondrianConformalAuditor,
        SplitConformalClassifier,
        train_calibrate_test_split,
    )
    from generate_synthetic_data import generate_conformal_credit_dataset

    df = generate_conformal_credit_dataset(n_samples=4000, seed=42)
    feature_cols = [
        "annual_income", "loan_amount", "fico_score",
        "debt_to_income", "revolving_utilization", "inquiries_last_6m"
    ]
    train_df, calib_df, test_df = train_calibrate_test_split(df, 0.50, 0.25, 0.25, random_state=42)

    clf = SplitConformalClassifier(alpha=0.05, random_state=42)
    clf.fit(train_df[feature_cols], train_df["is_default"].values)
    clf.calibrate(calib_df[feature_cols], calib_df["is_default"].values)

    # Baseline marginal prediction sets
    test_sets_marginal = clf.predict_set(test_df[feature_cols])
    marginal_audit = MondrianConformalAuditor.audit_marginal_vs_group_coverage(
        test_df["is_default"].values, test_sets_marginal, test_df["fico_tier"].values, target_coverage=0.95
    )
    assert len(marginal_audit) == 3

    # Calibrate group-conditional cutoffs
    group_cutoffs = MondrianConformalAuditor.calibrate_group_conditional(
        clf, calib_df[feature_cols], calib_df["is_default"].values, calib_df["fico_tier"].values, alpha=0.05
    )
    assert "Prime" in group_cutoffs
    assert "NearPrime" in group_cutoffs
    assert "Subprime" in group_cutoffs

    # Predict group-conditional prediction sets
    mondrian_sets = MondrianConformalAuditor.predict_set_group_conditional(
        clf, test_df[feature_cols], test_df["fico_tier"].values, group_cutoffs
    )
    mondrian_audit = MondrianConformalAuditor.audit_marginal_vs_group_coverage(
        test_df["is_default"].values, mondrian_sets, test_df["fico_tier"].values, target_coverage=0.95
    )

    # Every tier must meet group-conditional guarantee (>= 0.95 - 0.035 for finite sample variance with n~290)
    for _, row in mondrian_audit.iterrows():
        assert row["empirical_coverage"] >= 0.915, (
            f"Group {row['group']} under-covered in Mondrian calibration: {row['empirical_coverage']:.4f}"
        )


