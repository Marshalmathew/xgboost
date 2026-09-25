"""
XGBoost From Scratch: True 2nd-Order Gradient Boosting Tree Engine
==================================================================

Pure Python and NumPy implementation of the XGBoost exact greedy algorithm,
derived directly from Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System"
(arXiv:1603.02754).

Core Equations Implemented:
---------------------------
1. Binary LogLoss Gradient & Hessian (Sec 2.2):
   p_i = 1 / (1 + exp(-y_hat_i))
   g_i = p_i - y_i
   h_i = p_i * (1 - p_i)

2. Optimal Leaf Weight (Sec 2.2, Eq. 5):
   w* = - G / (H + lambda)
   where G = sum(g_i), H = sum(h_i)

3. Exact Split Gain Formula (Sec 2.2, Eq. 7):
   Gain = 0.5 * [ G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda) - G^2 / (H + lambda) ] - gamma

4. Split Pruning and Min Child Weight (Sec 2.2 & 2.1):
   - Reject split if H_L < min_child_weight or H_R < min_child_weight
   - Reject split if Gain <= 0 (gamma acts as complexity cost)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    # Clip x to avoid overflow in exp(-x)
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class LogLossObjective:
    """Binary Logistic Loss objective matching XGBoost's 'binary:logistic'."""

    @staticmethod
    def predict_proba(raw_margin: np.ndarray) -> np.ndarray:
        """Convert raw margins (logits) to probabilities."""
        return sigmoid(raw_margin)

    @staticmethod
    def gradient(y_true: np.ndarray, raw_margin: np.ndarray) -> np.ndarray:
        """First derivative of logloss w.r.t. raw_margin (logit): g_i = p_i - y_i."""
        p = sigmoid(raw_margin)
        return p - y_true

    @staticmethod
    def hessian(y_true: np.ndarray, raw_margin: np.ndarray) -> np.ndarray:
        """Second derivative of logloss w.r.t. raw_margin (logit): h_i = p_i * (1 - p_i)."""
        p = sigmoid(raw_margin)
        return p * (1.0 - p)


def compute_split_gain(
    g_l: float,
    h_l: float,
    g_r: float,
    h_r: float,
    reg_lambda: float,
    gamma: float,
) -> float:
    """Compute split gain according to Chen & Guestrin (2016) Eq. 7.

    Gain = 0.5 * [ G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda) - G^2 / (H + lambda) ] - gamma
    """
    g_total = g_l + g_r
    h_total = h_l + h_r

    score_left = (g_l**2) / (h_l + reg_lambda)
    score_right = (g_r**2) / (h_r + reg_lambda)
    score_total = (g_total**2) / (h_total + reg_lambda)

    gain = 0.5 * (score_left + score_right - score_total) - gamma
    return gain


def compute_leaf_weight(g_sum: float, h_sum: float, reg_lambda: float) -> float:
    """Compute optimal leaf weight w* according to Chen & Guestrin (2016) Eq. 5.

    w* = - G / (H + lambda)
    """
    return -g_sum / (h_sum + reg_lambda)


class XGBoostNode:
    """Represents a single node (decision branch or leaf) in an XGBoost tree."""

    def __init__(
        self,
        is_leaf: bool = False,
        weight: float = 0.0,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional[XGBoostNode] = None,
        right: Optional[XGBoostNode] = None,
        gain: float = 0.0,
    ) -> None:
        self.is_leaf = is_leaf
        self.weight = weight
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain

    def predict_one(self, x: np.ndarray) -> float:
        """Route a single 1D feature vector down the tree to return leaf weight."""
        if self.is_leaf:
            return self.weight
        assert self.feature_idx is not None and self.threshold is not None
        # In XGBoost, exact split routes x[feature_idx] <= threshold to left
        if x[self.feature_idx] <= self.threshold:
            assert self.left is not None
            return self.left.predict_one(x)
        else:
            assert self.right is not None
            return self.right.predict_one(x)


class XGBoostTreeScratch:
    """Single decision tree optimizing 2nd-order Taylor objective."""

    def __init__(
        self,
        max_depth: int = 3,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
    ) -> None:
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.root: Optional[XGBoostNode] = None

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> XGBoostTreeScratch:
        """Build tree recursively using exact greedy split finding."""
        indices = np.arange(X.shape[0])
        self.root = self._build_node(X, g, h, indices, depth=0)
        return self

    def _build_node(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        indices: np.ndarray,
        depth: int,
    ) -> XGBoostNode:
        g_sum = float(np.sum(g[indices]))
        h_sum = float(np.sum(h[indices]))
        default_weight = compute_leaf_weight(g_sum, h_sum, self.reg_lambda)

        # Base conditions: reached max depth or insufficient samples
        if depth >= self.max_depth or len(indices) <= 1:
            return XGBoostNode(is_leaf=True, weight=default_weight)

        best_gain = 0.0
        best_feat: Optional[int] = None
        best_threshold: Optional[float] = None
        best_left_idx: Optional[np.ndarray] = None
        best_right_idx: Optional[np.ndarray] = None

        n_features = X.shape[1]

        # Scan each continuous feature
        for feat in range(n_features):
            feat_vals = X[indices, feat]
            sort_order = np.argsort(feat_vals)
            sorted_indices = indices[sort_order]
            sorted_feat = feat_vals[sort_order]
            sorted_g = g[sorted_indices]
            sorted_h = h[sorted_indices]

            # Running prefix sums for Left and Right children
            g_left = 0.0
            h_left = 0.0

            n_samples = len(sorted_indices)
            for i in range(n_samples - 1):
                g_left += float(sorted_g[i])
                h_left += float(sorted_h[i])

                # Skip split if adjacent values are identical (can't separate)
                if sorted_feat[i] == sorted_feat[i + 1]:
                    continue

                g_right = g_sum - g_left
                h_right = h_sum - h_left

                # min_child_weight constraint on child hessians
                if h_left < self.min_child_weight or h_right < self.min_child_weight:
                    continue

                gain = compute_split_gain(
                    g_l=g_left,
                    h_l=h_left,
                    g_r=g_right,
                    h_r=h_right,
                    reg_lambda=self.reg_lambda,
                    gamma=self.gamma,
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    # Midpoint split threshold
                    best_threshold = (float(sorted_feat[i]) + float(sorted_feat[i + 1])) / 2.0
                    best_left_idx = sorted_indices[: i + 1]
                    best_right_idx = sorted_indices[i + 1 :]

        # If no split with positive gain satisfying min_child_weight was found, create leaf
        if best_gain <= 0.0 or best_feat is None or best_threshold is None:
            return XGBoostNode(is_leaf=True, weight=default_weight)

        assert best_left_idx is not None and best_right_idx is not None
        left_child = self._build_node(X, g, h, best_left_idx, depth=depth + 1)
        right_child = self._build_node(X, g, h, best_right_idx, depth=depth + 1)

        return XGBoostNode(
            is_leaf=False,
            feature_idx=best_feat,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            gain=best_gain,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict tree leaf weights for a matrix of features."""
        assert self.root is not None, "Tree must be fitted before predict."
        return np.array([self.root.predict_one(x) for x in X], dtype=np.float64)


class XGBoostScratch:
    """XGBoost Classifier from scratch with full 2nd-order Taylor boosting loop.

    Matches the exact training mechanism of XGBoost with 'binary:logistic'.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        learning_rate: float = 0.3,
        max_depth: int = 3,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        base_score: float = 0.5,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.base_score = base_score

        self.trees: list[XGBoostTreeScratch] = []
        # Initial margin on logit scale: F_0 = log(base_score / (1 - base_score))
        self.base_margin_: float = float(np.log(base_score / (1.0 - base_score)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostScratch:
        """Fit an ensemble of gradient-boosted trees."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples = X.shape[0]
        self.trees = []

        # Current raw margin predictions initialized to F_0
        raw_margin = np.full(n_samples, self.base_margin_, dtype=np.float64)

        for _ in range(self.n_estimators):
            # Compute gradients and hessians at current predictions
            g = LogLossObjective.gradient(y, raw_margin)
            h = LogLossObjective.hessian(y, raw_margin)

            # Fit a new tree on (X, g, h)
            tree = XGBoostTreeScratch(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
            )
            tree.fit(X, g, h)
            self.trees.append(tree)

            # Update margins with shrinkage: F_{m} = F_{m-1} + eta * f_m(X)
            tree_preds = tree.predict(X)
            raw_margin += self.learning_rate * tree_preds

        return self

    def predict_raw_margin(self, X: np.ndarray) -> np.ndarray:
        """Compute accumulated raw logit margins: F_0 + eta * sum(f_m(X))."""
        X = np.asarray(X, dtype=np.float64)
        margin = np.full(X.shape[0], self.base_margin_, dtype=np.float64)
        for tree in self.trees:
            margin += self.learning_rate * tree.predict(X)
        return margin

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute predicted probabilities: p = 1 / (1 + exp(-margin))."""
        margin = self.predict_raw_margin(X)
        return LogLossObjective.predict_proba(margin)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary classification predictions based on threshold."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(np.int32)
