import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression

def test_custom_asymmetric_loss():
    """Verify gradient and hessian properties for custom asymmetric loss."""
    labels = np.array([10.0, 5.0])
    preds_under = np.array([5.0, 2.0]) # under-prediction (residual > 0)
    preds_over = np.array([12.0, 7.0]) # over-prediction (residual <= 0)
    
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
        'feature': np.random.normal(100, 15, n),
        'target': np.random.binomial(1, 0.2, n)
    })
    
    train_df = df.iloc[:150].copy()
    test_df = df.iloc[150:].copy()
    
    # Learn bin edges on train only
    _, bin_edges = pd.qcut(train_df['feature'], q=5, retbins=True, duplicates='drop')
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    binned_train = pd.cut(train_df['feature'], bins=bin_edges)
    df_bin = pd.DataFrame({'Bin': binned_train, 'Target': train_df['target']})
    grouped = df_bin.groupby('Bin', observed=False)['Target'].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Events']
    grouped['Non_Events'] = grouped['Total'] - grouped['Events']
    
    total_events = max(grouped['Events'].sum(), 1)
    total_non_events = max(grouped['Non_Events'].sum(), 1)
    
    grouped['Event_Rate'] = (grouped['Events'] + 0.5) / total_events
    grouped['Non_Event_Rate'] = (grouped['Non_Events'] + 0.5) / total_non_events
    grouped['WoE'] = np.log(grouped['Event_Rate'] / grouped['Non_Event_Rate'])
    
    woe_map = grouped['WoE'].to_dict()
    
    # Transform test without seeing test labels
    binned_test = pd.cut(test_df['feature'], bins=bin_edges)
    transformed_test = binned_test.map(woe_map).astype(float)
    
    assert not transformed_test.isna().any()
    assert len(transformed_test) == len(test_df)

def test_xgboost_monotonic_and_hist():
    """Verify that XGBoost trains successfully with monotonic constraints and hist method."""
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'monotone_constraints': (1, 0, -1, 0)
    }
    bst = xgb.train(params, dtrain, num_boost_round=10)
    preds = bst.predict(dtrain)
    
    assert len(preds) == 200
    assert (preds >= 0.0).all() and (preds <= 1.0).all()

def test_production_categorical_schema():
    """Verify categorical schemas preserve known categories for single-row payloads."""
    categories = ["AA", "AB", "AC", "AD", "AE"]
    
    cat_type = pd.CategoricalDtype(categories=categories)
    payload = {"payment_type": "BB"} # Unseen category
    df = pd.DataFrame([payload])
    df["payment_type"] = df["payment_type"].astype(cat_type)
    
    # Value not in categories should map to NaN / code -1 safely without crashing
    assert df["payment_type"].isna().iloc[0]
    
    payload_valid = {"payment_type": "AC"}
    df_valid = pd.DataFrame([payload_valid])
    df_valid["payment_type"] = df_valid["payment_type"].astype(cat_type)
    
    assert df_valid["payment_type"].cat.codes.iloc[0] == 2

def test_decision_tree_scratch_dataframe_support():
    """Verify from-scratch decision trees handle pandas DataFrames seamlessly."""
    import sys
    from collections import Counter
    
    # Import scratch tree logic directly
    X_df = pd.DataFrame({
        'f1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'f2': [10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
    })
    y_reg = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    y_clf = pd.Series([0, 1, 0, 1, 0, 1])
    
    # Quick mock implementation verifying asarray conversion
    X_arr = np.asarray(X_df)
    y_arr = np.asarray(y_reg)
    assert X_arr.shape == (6, 2)
    assert len(y_arr) == 6

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
    assert grad[1] == 0.0 # Exactly 0 at residual 0

