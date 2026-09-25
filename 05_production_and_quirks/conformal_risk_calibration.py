"""
Conformal Prediction & Distribution-Free Uncertainty Guarantees for XGBoost
============================================================================

Implements rigorous, finite-sample, distribution-free conformal inference under exchangeability
(Vovk et al., 2005; Romano et al., 2019) for enterprise quantitative credit risk and algorithmic underwriting.

Architectural Components:
-------------------------
1. SplitConformalClassifier:
   - Finite-sample coverage guarantee: P(Y in C(X)) >= 1 - alpha
   - Uniform infinitesimal jitter to break XGBoost leaf probability ties
   - Institutional Tripartite Underwriting Triage:
     * {0}    -> AUTO_APPROVE (Zero/near-zero default rate)
     * {1}    -> AUTO_DENY (High-probability default)
     * {0, 1} -> REFER_TO_SENIOR_UNDERWRITER (Epistemic boundary ambiguity)
     * {}     -> OUT_OF_DISTRIBUTION_ALERT (Anomalous applicant)

2. ConformalizedQuantileRegressor (CQR):
   - Heteroskedastic loss forecasting for Loss Given Default (LGD) and Value-at-Risk (VaR)
   - Dual XGBoost pinball loss regressors (reg:quantileerror) at alpha/2 and 1 - alpha/2
   - Non-conformity score E_i = max(q_lo - y, y - q_hi) with exact finite-sample inflation
   - Financial zero-bound clipping: max(0.0, q_lo - q_hat)

3. MondrianConformalAuditor:
   - Group-conditional coverage audit: P(Y in C(X) | Group = g) >= 1 - alpha
   - Demographically stratified cutoffs across FICO risk tiers (Prime, NearPrime, Subprime)
   - Small-sample empirical Bayesian shrinkage to prevent threshold explosion on rare groups

Regulatory & Governance Alignment:
----------------------------------
Satisfies Federal Reserve SR 11-7 and OCC Bulletin 2011-12 expectations for mathematically
verifiable model risk bounds, automated straight-through processing (STP) guardrails, and
model epistemic uncertainty transparency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb


def train_calibrate_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.50,
    calib_frac: float = 0.25,
    test_frac: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions a DataFrame into three strictly disjoint partitions:
    Train (for model fitting), Calibration (for non-conformity scores), and Test (for evaluation).

    Strict disjointness is mathematically mandatory to preserve the exchangeability theorem.
    """
    total = train_frac + calib_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Fractions must sum to 1.0, got {total:.4f}")

    rng = np.random.RandomState(random_state)
    n = len(df)
    indices = rng.permutation(n)

    n_train = int(n * train_frac)
    n_calib = int(n * calib_frac)

    train_idx = indices[:n_train]
    calib_idx = indices[n_train : n_train + n_calib]
    test_idx = indices[n_train + n_calib :]

    if hasattr(df, "iloc"):
        return (
            df.iloc[train_idx].copy().reset_index(drop=True),
            df.iloc[calib_idx].copy().reset_index(drop=True),
            df.iloc[test_idx].copy().reset_index(drop=True),
        )
    else:
        arr = np.asarray(df)
        return (
            arr[train_idx],
            arr[calib_idx],
            arr[test_idx],
        )


class SplitConformalClassifier:
    """
    Inductive Split Conformal Prediction for XGBoost binary classification.
    Guarantees marginal finite-sample coverage under exchangeability:

        P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha

    Parameters
    ----------
    base_classifier : Optional[xgb.XGBClassifier]
        Base probability estimator. If None, default tuned XGBClassifier is instantiated.
    alpha : float, default=0.05
        User-specified maximum error rate (e.g. alpha=0.05 => 95% confidence).
    random_state : int, default=42
        Seed for jitter generation and tie-breaking reproducibility.
    """

    def __init__(
        self,
        base_classifier: Optional[xgb.XGBClassifier] = None,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.random_state = random_state
        self.base_classifier = base_classifier
        self.q_hat_: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_calib_: Optional[int] = None
        self.is_calibrated_: bool = False

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        **xgb_fit_params,
    ) -> "SplitConformalClassifier":
        """
        Fits the underlying XGBoost classifier on the training partition.
        """
        y_arr = np.asarray(y_train)
        self.classes_ = np.sort(np.unique(y_arr))

        if self.base_classifier is None:
            self.base_classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=self.random_state,
            )

        self.base_classifier.fit(X_train, y_arr, **xgb_fit_params)
        return self

    def calibrate(
        self,
        X_calib: Union[pd.DataFrame, np.ndarray],
        y_calib: Union[pd.Series, np.ndarray],
        apply_jitter: bool = True,
    ) -> "SplitConformalClassifier":
        """
        Calibrates the non-conformity threshold q_hat on held-out calibration data.

        Theoretical Finite-Sample Quantile Index:
            k = ceil((n_calib + 1) * (1 - alpha)) / n_calib
        """
        if self.base_classifier is None or self.classes_ is None:
            raise RuntimeError("Model must be fitted via .fit() before calling .calibrate().")

        y_calib_arr = np.asarray(y_calib)
        n_calib = len(y_calib_arr)

        # Enterprise Guardrail 1: Calibration sample size check
        min_required = int(np.ceil(1.0 / self.alpha) - 1)
        if n_calib < min_required:
            raise ValueError(
                f"Calibration set size (n={n_calib}) is too small for alpha={self.alpha}. "
                f"A minimum of {min_required} samples is required to satisfy finite-sample coverage guarantees."
            )

        # Predict probability distribution
        probas = self.base_classifier.predict_proba(X_calib)

        # Map true label to class column index
        class_to_idx = {c: idx for idx, c in enumerate(self.classes_)}
        true_indices = np.array([class_to_idx[y] for y in y_calib_arr])

        # True-class predicted probability: pi_{y_i}(x_i)
        prob_true = probas[np.arange(n_calib), true_indices]

        # Non-conformity score: s_i = 1 - pi_{y_i}(x_i)
        scores = 1.0 - prob_true

        # Enterprise Guardrail 2: Infinitesimal Uniform Jitter for tree probability tie-breaking
        if apply_jitter:
            rng = np.random.RandomState(self.random_state)
            jitter = rng.uniform(0.0, 1e-6, size=n_calib)
            scores = scores + jitter

        # Exact finite-sample quantile calculation
        # To cover rank ceil((n + 1) * (1 - alpha)), we take the k-th order statistic (1-indexed)
        k = int(np.ceil((n_calib + 1) * (1.0 - self.alpha)))
        k = min(k, n_calib)
        sorted_scores = np.sort(scores)
        self.q_hat_ = float(sorted_scores[k - 1])
        self.n_calib_ = n_calib
        self.is_calibrated_ = True

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns raw point-probability estimates from the underlying classifier."""
        if self.base_classifier is None:
            raise RuntimeError("Classifier must be fitted via .fit() before generating predictions.")
        return self.base_classifier.predict_proba(X)

    def predict_set(self, X: Union[pd.DataFrame, np.ndarray]) -> List[List[int]]:
        """
        Constructs conformal prediction sets C(x) for each query instance:
            C(x) = {k in Y : pi_k(x) >= 1 - q_hat}

        Guarantees that the true label is in C(x) with probability >= 1 - alpha.
        """
        if not self.is_calibrated_ or self.q_hat_ is None:
            raise RuntimeError("Classifier must be calibrated before generating prediction sets.")

        probas = self.base_classifier.predict_proba(X)
        threshold = 1.0 - self.q_hat_

        prediction_sets: List[List[int]] = []
        for p_row in probas:
            included = [int(self.classes_[idx]) for idx, p in enumerate(p_row) if p >= threshold]
            prediction_sets.append(included)

        return prediction_sets

    def triage_policy(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Maps conformal prediction sets into institutional banking underwriting actions:

        1. {0}     -> 'AUTO_APPROVE'
           High confidence good borrower; zero/negligible default probability.
        2. {1}     -> 'AUTO_DENY'
           High confidence risky borrower; default set is unambiguous.
        3. {0, 1}  -> 'REFER_TO_SENIOR_UNDERWRITER'
           Epistemic ambiguity on decision boundary; model admits uncertainty.
        4. {}      -> 'OUT_OF_DISTRIBUTION_ALERT'
           Model's highest probability falls below 1 - q_hat; abnormal applicant profile.
        """
        pred_sets = self.predict_set(X)
        probas = self.predict_proba(X)
        threshold = 1.0 - self.q_hat_

        actions: List[str] = []
        set_strs: List[str] = []
        cardinalities: List[int] = []

        for p_set in pred_sets:
            cardinalities.append(len(p_set))
            s = set(p_set)
            set_strs.append(str(p_set))

            if s == {0}:
                actions.append("AUTO_APPROVE")
            elif s == {1}:
                actions.append("AUTO_DENY")
            elif s == {0, 1}:
                actions.append("REFER_TO_SENIOR_UNDERWRITER")
            elif len(s) == 0:
                actions.append("OUT_OF_DISTRIBUTION_ALERT")
            else:
                actions.append("REFER_TO_SENIOR_UNDERWRITER")

        # Class 1 is default
        p_default = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]

        return pd.DataFrame({
            "prediction_set": set_strs,
            "set_cardinality": cardinalities,
            "triage_action": actions,
            "prob_default": p_default,
            "conformal_threshold": threshold,
            "q_hat": self.q_hat_,
        })

    def evaluate_coverage(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """
        Calculates empirical coverage and set size diagnostics on holdout evaluation data.
        """
        y_test_arr = np.asarray(y_test)
        pred_sets = self.predict_set(X_test)

        n_samples = len(y_test_arr)
        covered = sum(y_test_arr[i] in pred_sets[i] for i in range(n_samples))
        cardinalities = [len(s) for s in pred_sets]

        empty_count = sum(c == 0 for c in cardinalities)
        singleton_count = sum(c == 1 for c in cardinalities)
        ambiguous_count = sum(c > 1 for c in cardinalities)

        return {
            "empirical_coverage": float(covered / n_samples),
            "target_coverage": float(1.0 - self.alpha),
            "coverage_gap": float(covered / n_samples - (1.0 - self.alpha)),
            "mean_set_size": float(np.mean(cardinalities)),
            "empty_set_rate": float(empty_count / n_samples),
            "singleton_rate": float(singleton_count / n_samples),
            "ambiguous_rate": float(ambiguous_count / n_samples),
            "n_eval": n_samples,
        }


class ConformalizedQuantileRegressor:
    """
    Conformalized Quantile Regression (CQR) for heteroskedastic loss forecasting (Romano et al., 2019).
    Guarantees distribution-free coverage of continuous response targets (e.g. Loss Given Default / VaR):

        P(Y_{n+1} in [q_lo(X_{n+1}) - q_hat, q_hi(X_{n+1}) + q_hat]) >= 1 - alpha

    Parameters
    ----------
    alpha : float, default=0.10
        Desired miscoverage rate (alpha=0.10 guarantees >= 90% coverage).
    n_estimators : int, default=100
        Number of boosting trees per quantile regressor.
    max_depth : int, default=4
        Tree depth for capturing non-linear conditional heteroskedasticity.
    learning_rate : float, default=0.05
        Shrinkage parameter.
    random_state : int, default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.alpha_lo = alpha / 2.0
        self.alpha_hi = 1.0 - (alpha / 2.0)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        # Native XGBoost quantile loss models
        self.model_lo: Optional[xgb.XGBRegressor] = None
        self.model_hi: Optional[xgb.XGBRegressor] = None
        self.q_hat_: Optional[float] = None
        self.is_calibrated_: bool = False

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> "ConformalizedQuantileRegressor":
        """
        Fits two independent XGBoost quantile regressors for the lower and upper bounds.
        """
        y_arr = np.asarray(y_train, dtype=float)

        self.model_lo = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=self.alpha_lo,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

        self.model_hi = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=self.alpha_hi,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state + 1,
        )

        self.model_lo.fit(X_train, y_arr)
        self.model_hi.fit(X_train, y_arr)
        return self

    def calibrate(
        self,
        X_calib: Union[pd.DataFrame, np.ndarray],
        y_calib: Union[pd.Series, np.ndarray],
    ) -> "ConformalizedQuantileRegressor":
        """
        Calibrates the CQR interval expansion factor q_hat on the calibration partition.

        Non-conformity score (Romano et al., 2019):
            E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
        """
        if self.model_lo is None or self.model_hi is None:
            raise RuntimeError("Model must be fitted via .fit() before calling .calibrate().")

        y_calib_arr = np.asarray(y_calib, dtype=float)
        n_calib = len(y_calib_arr)

        # Enterprise Guardrail: Sample size validation
        min_required = int(np.ceil(1.0 / self.alpha) - 1)
        if n_calib < min_required:
            raise ValueError(
                f"Calibration set size (n={n_calib}) is too small for alpha={self.alpha}. "
                f"Minimum required is {min_required} samples."
            )

        pred_lo = self.model_lo.predict(X_calib)
        pred_hi = self.model_hi.predict(X_calib)

        # Compute symmetric conformity error: distance outside the raw quantile envelope
        scores = np.maximum(pred_lo - y_calib_arr, y_calib_arr - pred_hi)

        # Exact finite-sample quantile index: ceil((n + 1) * (1 - alpha)) / n
        k = int(np.ceil((n_calib + 1) * (1.0 - self.alpha)))
        k = min(k, n_calib)
        sorted_scores = np.sort(scores)
        self.q_hat_ = float(sorted_scores[k - 1])
        self.is_calibrated_ = True

        return self

    def predict_interval(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        clip_zero: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates calibrated continuous prediction intervals [lo, hi].

        Parameters
        ----------
        X : query covariates
        clip_zero : bool, default=True
            Enforces physical zero lower-bound for monetary loss quantities (LGD/VaR >= 0).
        """
        if not self.is_calibrated_ or self.q_hat_ is None:
            raise RuntimeError("Regressor must be calibrated before generating intervals.")

        raw_lo = self.model_lo.predict(X)
        raw_hi = self.model_hi.predict(X)

        calibrated_lo = raw_lo - self.q_hat_
        calibrated_hi = raw_hi + self.q_hat_

        if clip_zero:
            calibrated_lo = np.maximum(0.0, calibrated_lo)
            calibrated_hi = np.maximum(calibrated_lo, calibrated_hi)
        else:
            calibrated_hi = np.maximum(calibrated_lo, calibrated_hi)

        return calibrated_lo, calibrated_hi

    def evaluate_coverage(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        true_sigma: Optional[np.ndarray] = None,
        clip_zero: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluates empirical coverage, interval efficiency, and heteroskedastic correlation.
        """
        y_arr = np.asarray(y_test, dtype=float)
        n = len(y_arr)

        # Raw pinball quantiles (before CQR calibration)
        raw_lo = self.model_lo.predict(X_test)
        raw_hi = self.model_hi.predict(X_test)
        raw_covered = np.sum((y_arr >= raw_lo) & (y_arr <= raw_hi))
        raw_coverage = float(raw_covered / n)

        # Calibrated intervals
        cal_lo, cal_hi = self.predict_interval(X_test, clip_zero=clip_zero)
        cal_covered = np.sum((y_arr >= cal_lo) & (y_arr <= cal_hi))
        cal_coverage = float(cal_covered / n)

        interval_widths = cal_hi - cal_lo

        results = {
            "raw_quantile_coverage": raw_coverage,
            "cqr_calibrated_coverage": cal_coverage,
            "target_coverage": float(1.0 - self.alpha),
            "mean_interval_width": float(np.mean(interval_widths)),
            "median_interval_width": float(np.median(interval_widths)),
            "q_hat_adjustment": float(self.q_hat_),
            "n_eval": n,
        }

        # If ground-truth conditional volatility is provided, verify adaptive width scaling
        if true_sigma is not None:
            sig = np.asarray(true_sigma)
            if np.std(sig) > 1e-9 and np.std(interval_widths) > 1e-9:
                corr = np.corrcoef(interval_widths, sig)[0, 1]
                results["width_heteroskedastic_correlation"] = float(corr)
            else:
                results["width_heteroskedastic_correlation"] = 0.0

        return results


class MondrianConformalAuditor:
    """
    Mondrian (Group-Conditional) Conformal Prediction and Fair Lending Auditor.
    Audits and enforces group-level coverage parity:

        P(Y in C(X) | Group = g) >= 1 - alpha, for all g in G

    Prevents demographic under-coverage pathology where global marginal coverage is 95%
    but protected groups or high-risk FICO tiers suffer severe miscoverage.
    """

    @staticmethod
    def audit_marginal_vs_group_coverage(
        y_true: Union[pd.Series, np.ndarray],
        prediction_sets: List[List[int]],
        groups: Union[pd.Series, np.ndarray],
        target_coverage: float = 0.95,
    ) -> pd.DataFrame:
        """
        Computes per-group empirical coverage, set cardinality, and parity metrics.
        """
        y_arr = np.asarray(y_true)
        grp_arr = np.asarray(groups)

        df_records = []
        unique_groups = np.unique(grp_arr)

        for g in unique_groups:
            mask = grp_arr == g
            n_g = np.sum(mask)
            y_g = y_arr[mask]
            sets_g = [prediction_sets[i] for i in np.where(mask)[0]]

            covered_g = sum(y_g[i] in sets_g[i] for i in range(n_g))
            cov_rate = covered_g / n_g if n_g > 0 else 0.0
            mean_size = np.mean([len(s) for s in sets_g]) if n_g > 0 else 0.0

            df_records.append({
                "group": g,
                "n_samples": int(n_g),
                "empirical_coverage": float(cov_rate),
                "target_coverage": float(target_coverage),
                "coverage_gap": float(cov_rate - target_coverage),
                "is_undercovered": bool(cov_rate < target_coverage - 0.01),
                "mean_set_size": float(mean_size),
            })

        return pd.DataFrame(df_records).sort_values("group").reset_index(drop=True)

    @staticmethod
    def calibrate_group_conditional(
        classifier: SplitConformalClassifier,
        X_calib: Union[pd.DataFrame, np.ndarray],
        y_calib: Union[pd.Series, np.ndarray],
        groups_calib: Union[pd.Series, np.ndarray],
        alpha: float = 0.05,
        min_group_size: int = 30,
    ) -> Dict[Any, float]:
        """
        Computes individual conformal cutoffs q_hat_g for each demographic group.

        Enterprise Guardrail (Shrinkage):
        If a group contains fewer than `min_group_size` samples in calibration,
        shrinks the threshold towards the global marginal q_hat to prevent high-variance breakdown.
        """
        y_calib_arr = np.asarray(y_calib)
        grp_arr = np.asarray(groups_calib)
        probas = classifier.predict_proba(X_calib)

        class_to_idx = {c: idx for idx, c in enumerate(classifier.classes_)}
        true_indices = np.array([class_to_idx[y] for y in y_calib_arr])
        scores_all = 1.0 - probas[np.arange(len(y_calib_arr)), true_indices]

        # Marginal cutoff for shrinkage fallback
        k_global = int(np.ceil((len(scores_all) + 1) * (1.0 - alpha)))
        k_global = min(k_global, len(scores_all))
        q_marginal = float(np.sort(scores_all)[k_global - 1])

        group_cutoffs: Dict[Any, float] = {}
        unique_groups = np.unique(grp_arr)

        for g in unique_groups:
            mask = grp_arr == g
            scores_g = scores_all[mask]
            n_g = len(scores_g)

            if n_g < min_group_size:
                # Shrinkage towards global marginal cutoff
                weight = float(n_g) / float(min_group_size)
                # Compute raw group cutoff if feasible
                if n_g >= int(np.ceil(1.0 / alpha) - 1):
                    k_g = int(np.ceil((n_g + 1) * (1.0 - alpha)))
                    k_g = min(k_g, n_g)
                    raw_q_g = float(np.sort(scores_g)[k_g - 1])
                    q_g = weight * raw_q_g + (1.0 - weight) * q_marginal
                else:
                    q_g = q_marginal
            else:
                k_g = int(np.ceil((n_g + 1) * (1.0 - alpha)))
                k_g = min(k_g, n_g)
                q_g = float(np.sort(scores_g)[k_g - 1])

            group_cutoffs[g] = float(q_g)

        return group_cutoffs

    @staticmethod
    def predict_set_group_conditional(
        classifier: SplitConformalClassifier,
        X: Union[pd.DataFrame, np.ndarray],
        groups: Union[pd.Series, np.ndarray],
        group_cutoffs: Dict[Any, float],
        fallback_cutoff: Optional[float] = None,
    ) -> List[List[int]]:
        """
        Generates prediction sets using group-specific cutoffs to guarantee Mondrian fairness.
        """
        probas = classifier.predict_proba(X)
        grp_arr = np.asarray(groups)

        default_q = fallback_cutoff if fallback_cutoff is not None else (
            classifier.q_hat_ if classifier.q_hat_ is not None else (
                float(np.mean(list(group_cutoffs.values()))) if group_cutoffs else 0.5
            )
        )
        prediction_sets: List[List[int]] = []

        for i, p_row in enumerate(probas):
            g = grp_arr[i]
            if pd.isna(g):
                q_g = default_q
            else:
                q_g = group_cutoffs.get(g, default_q)
            threshold = 1.0 - q_g

            included = [int(classifier.classes_[idx]) for idx, p in enumerate(p_row) if p >= threshold]
            prediction_sets.append(included)

        return prediction_sets
