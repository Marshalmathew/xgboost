"""
Enterprise Causal Machine Learning & Uplift Decisioning Engine
=============================================================

Production-grade CATE (Conditional Average Treatment Effect) estimation using XGBoost:
1. XGBoostSLearner: Single-model meta-learner baseline with feature split depth diagnostics.
2. XGBoostTLearner: Two-model meta-learner with variance imbalance diagnostics.
3. XGBoostXLearner: Künzel et al. (PNAS 2019) with K-Fold Cross-Fitting and corrected
   propensity weighting: tau_hat = e(x)*tau_0(x) + (1 - e(x))*tau_1(x).
4. Causal Metrics Suite: Qini curves, AUUC, Qini score, and bootstrap permutation test.
5. Budget-Constrained Argmax Policy Optimizer: Closed-form NEV (Net Expected Value)
   maximizer and budget frontier sweep for retail banking campaign decisioning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


class XGBoostSLearner:
    """
    S-Learner (Single Model Meta-Learner).
    Estimates mu(x, t) = E[Y | X=x, T=t] using a single XGBoost model where
    treatment indicator T is included as an ordinary feature.

    tau_S(x) = mu(x, 1) - mu(x, 0)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "XGBoostSLearner":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.values
        else:
            self.feature_names_ = [f"f_{i}" for i in range(X.shape[1])]
            X_mat = np.asarray(X)

        t_col = np.asarray(treatment).reshape(-1, 1)
        X_with_t = np.hstack([X_mat, t_col])
        full_feature_names = self.feature_names_ + ["treatment"]

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        self.model.fit(X_with_t, y)
        self.model.get_booster().feature_names = full_feature_names
        return self

    def predict_cate(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        n = X_mat.shape[0]

        # Evaluate at T=1 and T=0
        X_t1 = np.hstack([X_mat, np.ones((n, 1))])
        X_t0 = np.hstack([X_mat, np.zeros((n, 1))])

        pred_1 = self.model.predict_proba(X_t1)[:, 1]
        pred_0 = self.model.predict_proba(X_t0)[:, 1]
        return pred_1 - pred_0

    def get_treatment_split_diagnostics(self) -> Dict[str, Any]:
        """
        Calculates how frequently the treatment feature was chosen for tree splits.
        Demonstrates the S-Learner pathology: when |X| >> 1, XGBoost's regularization
        frequently ignores T, yielding near-zero CATE estimates.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        booster = self.model.get_booster()
        score_weight = booster.get_score(importance_type="weight")
        score_gain = booster.get_score(importance_type="gain")

        t_splits = score_weight.get("treatment", 0)
        total_splits = sum(score_weight.values()) if score_weight else 0
        t_gain = score_gain.get("treatment", 0.0)

        return {
            "treatment_split_count": t_splits,
            "total_split_count": total_splits,
            "treatment_split_fraction": (t_splits / total_splits) if total_splits > 0 else 0.0,
            "treatment_mean_gain": float(t_gain),
        }


class XGBoostTLearner:
    """
    T-Learner (Two-Model Meta-Learner).
    Trains two separate models:
      mu_1(x) = E[Y | X=x, T=1]
      mu_0(x) = E[Y | X=x, T=0]

    tau_T(x) = mu_1(x) - mu_0(x)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model_0: Optional[xgb.XGBClassifier] = None
        self.model_1: Optional[xgb.XGBClassifier] = None

    def fit(self, X: pd.DataFrame | np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "XGBoostTLearner":
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        t = np.asarray(treatment).ravel()
        y = np.asarray(y).ravel()

        mask_1 = (t == 1)
        mask_0 = (t == 0)

        self.model_0 = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        self.model_1 = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state + 1,
            eval_metric="logloss",
        )

        self.model_0.fit(X_mat[mask_0], y[mask_0])
        self.model_1.fit(X_mat[mask_1], y[mask_1])
        return self

    def predict_cate(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_0 is None or self.model_1 is None:
            raise RuntimeError("Models are not fitted yet.")

        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        pred_1 = self.model_1.predict_proba(X_mat)[:, 1]
        pred_0 = self.model_0.predict_proba(X_mat)[:, 1]
        return pred_1 - pred_0

    def get_variance_diagnostic(self, X: pd.DataFrame | np.ndarray, tau_true: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Diagnoses variance instability when treatment and control sample sizes are imbalanced.
        """
        tau_hat = self.predict_cate(X)
        res = {"var_tau_hat": float(np.var(tau_hat))}
        if tau_true is not None:
            tau_true = np.asarray(tau_true)
            var_true = float(np.var(tau_true))
            res["var_tau_true"] = var_true
            res["var_ratio"] = float(np.var(tau_hat) / (var_true + 1e-8))
            res["rmse_vs_true"] = float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
        return res


class XGBoostXLearner:
    """
    X-Learner (Künzel et al., PNAS 2019) with K-Fold Cross-Fitting.

    Three-Stage Architecture:
      Stage 1: Estimate response surfaces mu_0(x) and mu_1(x).
      Stage 2: Impute counterfactual residuals with K-Fold cross-fitting:
               D_1 = Y_1 - mu_0(X_1)
               D_0 = mu_1(X_0) - Y_0
               Train second-stage regressors tau_1(x) and tau_0(x).
      Stage 3: Estimate propensity score e(x) = P(T=1 | X).
               Combine using the Künzel et al. formula:
               tau_X(x) = e(x) * tau_0(x) + (1 - e(x)) * tau_1(x)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_splits: int = 5,
        use_cross_fitting: bool = True,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_splits = n_splits
        self.use_cross_fitting = use_cross_fitting
        self.random_state = random_state

        # Models
        self.model_mu0: Optional[xgb.XGBClassifier] = None
        self.model_mu1: Optional[xgb.XGBClassifier] = None
        self.model_tau0: Optional[xgb.XGBRegressor] = None
        self.model_tau1: Optional[xgb.XGBRegressor] = None
        self.model_propensity: Optional[xgb.XGBClassifier] = None

    def fit(self, X: pd.DataFrame | np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "XGBoostXLearner":
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        t = np.asarray(treatment).ravel()
        y = np.asarray(y).ravel()

        idx_treated = np.where(t == 1)[0]
        idx_control = np.where(t == 0)[0]

        # Stage 1: Fit base models mu_0 and mu_1 on full groups
        self.model_mu0 = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        self.model_mu1 = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state + 1,
            eval_metric="logloss",
        )

        self.model_mu0.fit(X_mat[idx_control], y[idx_control])
        self.model_mu1.fit(X_mat[idx_treated], y[idx_treated])

        # Stage 2: Impute counterfactual residuals
        if self.use_cross_fitting:
            # K-fold cross-fitting to eliminate in-sample imputation bias
            d_1 = np.zeros(len(idx_treated), dtype=float)
            d_0 = np.zeros(len(idx_control), dtype=float)

            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            # Cross-fitting for D_1: evaluate on fold k using mu_0 fit on other control folds
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, t)):
                ctrl_train = np.intersect1d(train_idx, idx_control)
                treated_val = np.intersect1d(val_idx, idx_treated)

                if len(ctrl_train) > 0 and len(treated_val) > 0:
                    fold_mu0 = xgb.XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        subsample=self.subsample,
                        colsample_bytree=self.colsample_bytree,
                        random_state=self.random_state + fold,
                        eval_metric="logloss",
                    )
                    fold_mu0.fit(X_mat[ctrl_train], y[ctrl_train])
                    mu0_pred_val = fold_mu0.predict_proba(X_mat[treated_val])[:, 1]

                    # Map back to idx_treated positions
                    pos_treated_val = np.searchsorted(idx_treated, treated_val)
                    d_1[pos_treated_val] = y[treated_val] - mu0_pred_val

            # Cross-fitting for D_0: evaluate on fold k using mu_1 fit on other treated folds
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, t)):
                treated_train = np.intersect1d(train_idx, idx_treated)
                ctrl_val = np.intersect1d(val_idx, idx_control)

                if len(treated_train) > 0 and len(ctrl_val) > 0:
                    fold_mu1 = xgb.XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        subsample=self.subsample,
                        colsample_bytree=self.colsample_bytree,
                        random_state=self.random_state + 10 + fold,
                        eval_metric="logloss",
                    )
                    fold_mu1.fit(X_mat[treated_train], y[treated_train])
                    mu1_pred_val = fold_mu1.predict_proba(X_mat[ctrl_val])[:, 1]

                    # Map back to idx_control positions
                    pos_ctrl_val = np.searchsorted(idx_control, ctrl_val)
                    d_0[pos_ctrl_val] = mu1_pred_val - y[ctrl_val]

        else:
            # In-sample imputation (without cross-fitting)
            mu0_pred_treated = self.model_mu0.predict_proba(X_mat[idx_treated])[:, 1]
            mu1_pred_control = self.model_mu1.predict_proba(X_mat[idx_control])[:, 1]
            d_1 = y[idx_treated] - mu0_pred_treated
            d_0 = mu1_pred_control - y[idx_control]

        # Stage 2b: Regress imputed effects tau_1 and tau_0
        self.model_tau1 = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state + 2,
        )
        self.model_tau0 = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state + 3,
        )

        self.model_tau1.fit(X_mat[idx_treated], d_1)
        self.model_tau0.fit(X_mat[idx_control], d_0)

        # Stage 3: Fit propensity classifier e(x) = P(T=1 | X)
        self.model_propensity = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=3,
            learning_rate=self.learning_rate,
            random_state=self.random_state + 4,
            eval_metric="logloss",
        )
        self.model_propensity.fit(X_mat, t)

        return self

    def predict_cate(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if (
            self.model_tau0 is None
            or self.model_tau1 is None
            or self.model_propensity is None
        ):
            raise RuntimeError("Model is not fitted yet.")

        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        tau0_pred = self.model_tau0.predict(X_mat)
        tau1_pred = self.model_tau1.predict(X_mat)
        e_hat = self.model_propensity.predict_proba(X_mat)[:, 1]

        # Künzel et al. (PNAS 2019) Section 2.3 formula:
        # tau(x) = e(x) * tau_0(x) + (1 - e(x)) * tau_1(x)
        #
        # Counterfactual Imputation Intuition:
        # - tau_1 is trained on D_1 = Y_1 - mu_0(X_1). When e(x) is small (control-dominated cohort),
        #   mu_0 was trained on abundant control data, making mu_0 very accurate and D_1 clean.
        #   Hence tau_1 receives high weight (1 - e(x)).
        # - Conversely, when e(x) is large (treatment-dominated), mu_1 was trained on abundant
        #   treated data, making D_0 clean and giving tau_0 weight e(x).
        tau_x = e_hat * tau0_pred + (1.0 - e_hat) * tau1_pred
        return tau_x

    def predict_propensity(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_propensity is None:
            raise RuntimeError("Propensity model is not fitted.")
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model_propensity.predict_proba(X_mat)[:, 1]


# ==============================================================================
# CAUSAL METRICS SUITE
# ==============================================================================

def compute_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    tau_hat: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes cumulative Qini curve across population fractions.
    Q(k) = n_{t,1}(k) - n_{c,1}(k) * (N_t(k) / N_c(k))

    Parameters:
    -----------
    y_true : 1D array of binary outcomes Y
    treatment : 1D array of treatment indicators T
    tau_hat : 1D array of predicted uplift / CATE scores
    n_bins : Number of evaluation fraction bins (default: 10)

    Returns:
    --------
    fractions : Array of population fractions from 0.0 to 1.0
    qini_values : Array of Qini metric values at each fraction
    """
    y = np.asarray(y_true).ravel()
    t = np.asarray(treatment).ravel()
    scores = np.asarray(tau_hat).ravel()
    n = len(y)

    order = np.argsort(-scores)
    y_sorted = y[order]
    t_sorted = t[order]

    fractions = np.linspace(0.0, 1.0, n_bins + 1)
    qini_values = [0.0]

    for frac in fractions[1:]:
        k = int(round(frac * n))
        if k == 0:
            qini_values.append(0.0)
            continue

        y_top = y_sorted[:k]
        t_top = t_sorted[:k]

        n_t = np.sum(t_top == 1)
        n_c = np.sum(t_top == 0)
        y_t1 = np.sum((t_top == 1) & (y_top == 1))
        y_c1 = np.sum((t_top == 0) & (y_top == 1))

        if n_c > 0:
            q_val = y_t1 - y_c1 * (n_t / n_c)
        else:
            q_val = float(y_t1)
        qini_values.append(float(q_val))

    return fractions, np.array(qini_values)


def compute_auuc(
    y_true: np.ndarray,
    treatment: np.ndarray,
    tau_hat: np.ndarray,
    n_bins: int = 100,
) -> float:
    """
    Computes AUUC (Area Under the Uplift / Qini Curve) using trapezoidal rule,
    normalized by total population size N.
    """
    fractions, qini_values = compute_qini_curve(y_true, treatment, tau_hat, n_bins=n_bins)
    n = len(y_true)
    # Area under curve normalized by N (NumPy 1.x and 2.x compatible)
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auuc = float(trapz_func(qini_values, fractions)) / (n + 1e-8)
    return auuc


def compute_qini_score(
    y_true: np.ndarray,
    treatment: np.ndarray,
    tau_hat: np.ndarray,
    tau_true: Optional[np.ndarray] = None,
    n_bins: int = 100,
) -> float:
    """
    Computes Qini score: AUUC of tau_hat relative to the oracle AUUC.
    If tau_true is not provided, normalizes against a positive reference.
    """
    observed_auuc = compute_auuc(y_true, treatment, tau_hat, n_bins=n_bins)
    if tau_true is not None:
        oracle_auuc = compute_auuc(y_true, treatment, tau_true, n_bins=n_bins)
        return float(observed_auuc / (oracle_auuc + 1e-8))
    return float(observed_auuc)


def bootstrap_permutation_test(
    y_true: np.ndarray,
    treatment: np.ndarray,
    tau_hat: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Tests whether the observed uplift ranking is statistically significantly
    superior to random targeting using a non-parametric permutation test.

    Under H0: tau_hat has no ranking ability (exchangeable with random permutation).
    """
    rng = np.random.RandomState(random_state)
    observed_auuc = compute_auuc(y_true, treatment, tau_hat, n_bins=50)

    tau_perm = np.asarray(tau_hat, dtype=float).copy()
    null_auucs = np.zeros(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        rng.shuffle(tau_perm)
        null_auucs[i] = compute_auuc(y_true, treatment, tau_perm, n_bins=50)

    # One-sided p-value: probability null exceeds or matches observed
    p_value = (np.sum(null_auucs >= observed_auuc) + 1.0) / (n_bootstrap + 1.0)

    return {
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_auucs)),
        "null_std": float(np.std(null_auucs)),
        "observed_auuc": float(observed_auuc),
        "is_significant": bool(p_value < alpha),
    }


# ==============================================================================
# BUDGET-CONSTRAINED ARGMAX POLICY OPTIMIZER
# ==============================================================================

def compute_nev(
    tau_hat: np.ndarray,
    v_convert: float = 250.0,
    c_contact: float = 15.0,
    p_churn: float = 300.0,
) -> np.ndarray:
    """
    Calculates expected Individual Net Expected Value (NEV) for outreach decisioning:
      NEV_i = tau_i^+ * V_convert - C_contact - tau_i^- * P_churn
    where tau_i^+ = max(tau_i, 0) and tau_i^- = max(-tau_i, 0).
    """
    tau = np.asarray(tau_hat, dtype=float)
    tau_pos = np.maximum(tau, 0.0)
    tau_neg = np.maximum(-tau, 0.0)
    nev = tau_pos * v_convert - c_contact - tau_neg * p_churn
    return nev


def optimize_policy(
    tau_hat: np.ndarray,
    budget_ratio: float,
    v_convert: float = 250.0,
    c_contact: float = 15.0,
    p_churn: float = 300.0,
) -> np.ndarray:
    """
    Closed-Form Optimal Policy:
      S* = { i : NEV_i > 0 } intersect Top-B ranked by NEV_i
    Returns boolean mask of selected accounts.
    """
    nev = compute_nev(tau_hat, v_convert, c_contact, p_churn)
    n = len(nev)
    max_contacts = int(np.floor(budget_ratio * n))

    # Eligible accounts with positive expected ROI
    positive_mask = nev > 0.0

    if max_contacts <= 0 or not np.any(positive_mask):
        return np.zeros(n, dtype=bool)

    # Rank by NEV descending
    ranked_indices = np.argsort(-nev)

    # Filter to only positive NEV indices
    eligible_ranked = [idx for idx in ranked_indices if positive_mask[idx]]
    selected_indices = eligible_ranked[:max_contacts]

    policy_mask = np.zeros(n, dtype=bool)
    policy_mask[selected_indices] = True
    return policy_mask


def evaluate_policy_realized_profit(
    selected_mask: np.ndarray,
    y_true: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray | float = 0.15,
    v_convert: float = 250.0,
    c_contact: float = 15.0,
    p_churn: float = 300.0,
    tau_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluates campaign profit of a selected outreach cohort.
    If ground truth tau_true is known (synthetic benchmark), computes exact net profit.
    Otherwise uses the Horvitz-Thompson unbiased estimator on RCT observations.
    """
    selected = np.asarray(selected_mask, dtype=bool)
    n_contacted = int(np.sum(selected))

    if n_contacted == 0:
        return {
            "n_contacted": 0,
            "incremental_conversions": 0.0,
            "campaign_cost": 0.0,
            "gross_revenue": 0.0,
            "churn_penalty": 0.0,
            "net_profit": 0.0,
        }

    campaign_cost = n_contacted * c_contact

    if tau_true is not None:
        # Exact ground-truth economics
        tau_sel = tau_true[selected]
        tau_pos = np.maximum(tau_sel, 0.0)
        tau_neg = np.maximum(-tau_sel, 0.0)

        inc_conversions = float(np.sum(tau_pos))
        gross_rev = inc_conversions * v_convert
        churn_pen = float(np.sum(tau_neg)) * p_churn
        net_prof = gross_rev - campaign_cost - churn_pen
    else:
        # Horvitz-Thompson estimator on observed RCT outcomes
        y = np.asarray(y_true)[selected]
        t = np.asarray(treatment)[selected]
        p = np.asarray(propensity)[selected] if isinstance(propensity, np.ndarray) else propensity

        weights_t = (t * y) / (p + 1e-8)
        weights_c = ((1 - t) * y) / (1.0 - p + 1e-8)
        inc_conversions = float(np.sum(weights_t - weights_c))
        gross_rev = inc_conversions * v_convert
        churn_pen = 0.0
        net_prof = gross_rev - campaign_cost

    return {
        "n_contacted": n_contacted,
        "incremental_conversions": inc_conversions,
        "campaign_cost": campaign_cost,
        "gross_revenue": gross_rev,
        "churn_penalty": churn_pen,
        "net_profit": net_prof,
    }


def sweep_budget_frontier(
    tau_hat: np.ndarray,
    naive_propensity_scores: np.ndarray,
    tau_true: np.ndarray,
    budget_grid: Optional[np.ndarray] = None,
    v_convert: float = 250.0,
    c_contact: float = 15.0,
    p_churn: float = 300.0,
) -> pd.DataFrame:
    """
    Sweeps budget from 1% to 50% and compares:
      1. Causal Uplift Policy (NEV argmax)
      2. Naive Propensity Policy (Top P(Y=1|X))
      3. Random Baseline
    """
    if budget_grid is None:
        budget_grid = np.linspace(0.02, 0.50, 25)

    n = len(tau_hat)
    records = []

    for b in budget_grid:
        k = int(np.floor(b * n))
        if k <= 0:
            continue

        # Strategy 1: Causal Uplift (NEV)
        causal_mask = optimize_policy(tau_hat, b, v_convert, c_contact, p_churn)
        prof_causal = evaluate_policy_realized_profit(
            causal_mask, None, None, v_convert=v_convert, c_contact=c_contact, p_churn=p_churn, tau_true=tau_true
        )["net_profit"]

        # Strategy 2: Naive Predictive Propensity
        naive_rank = np.argsort(-naive_propensity_scores)
        naive_mask = np.zeros(n, dtype=bool)
        naive_mask[naive_rank[:k]] = True
        prof_naive = evaluate_policy_realized_profit(
            naive_mask, None, None, v_convert=v_convert, c_contact=c_contact, p_churn=p_churn, tau_true=tau_true
        )["net_profit"]

        # Strategy 3: Random Baseline
        rand_order = np.random.RandomState(42).permutation(n)
        rand_mask = np.zeros(n, dtype=bool)
        rand_mask[rand_order[:k]] = True
        prof_rand = evaluate_policy_realized_profit(
            rand_mask, None, None, v_convert=v_convert, c_contact=c_contact, p_churn=p_churn, tau_true=tau_true
        )["net_profit"]

        records.append({
            "budget_ratio": b,
            "accounts_targeted": k,
            "profit_causal": prof_causal,
            "profit_naive": prof_naive,
            "profit_random": prof_rand,
        })

    return pd.DataFrame(records)
