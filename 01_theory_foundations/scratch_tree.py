"""
Decision Tree implementations from scratch for educational and testing verification.
Supports pure Python / NumPy operations and pandas DataFrame/Series inputs.
"""
from collections import Counter

import numpy as np


class Node:
    """A single node within a decision tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeRegressorFromScratch:
    """Mean Squared Error (MSE) minimizing Decision Tree Regressor."""
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.feature_importances_ = None
        self.n_features_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root = self._grow_tree(X, y, 0)

        # Normalize feature importances
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance
        return self

    def _grow_tree(self, X, y, depth):
        n_samples, _ = X.shape

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(value=float(np.mean(y)))

        # Find the best split
        best_feat, best_thresh, best_mse = self._best_split(X, y)

        if best_feat is None:
            return Node(value=float(np.mean(y)))

        # Feature Importance: Decrease in variance weighted by sample size
        parent_variance = np.var(y)
        impurity_decrease = (parent_variance - best_mse) * n_samples
        self.feature_importances_[best_feat] += impurity_decrease

        # Create children
        left_idxs = np.where(X[:, best_feat] <= best_thresh)[0]
        right_idxs = np.where(X[:, best_feat] > best_thresh)[0]

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y):
        best_mse = float("inf")
        best_feat, best_thresh = None, None
        n_features = X.shape[1]

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thresh in thresholds:
                left_idxs = np.where(X_column <= thresh)[0]
                right_idxs = np.where(X_column > thresh)[0]

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                mse = self._calculate_split_mse(y, left_idxs, right_idxs)

                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh, best_mse

    def _calculate_split_mse(self, y, left_idxs, right_idxs):
        y_l, y_r = y[left_idxs], y[right_idxs]
        var_l = np.var(y_l) if len(y_l) > 0 else 0.0
        var_r = np.var(y_r) if len(y_r) > 0 else 0.0
        return (len(y_l) / len(y)) * var_l + (len(y_r) / len(y)) * var_r

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class DecisionTreeClassifierFromScratch:
    """Gini Impurity minimizing Decision Tree Classifier."""
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.feature_importances_ = None
        self.n_features_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root = self._grow_tree(X, y, 0)

        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance
        return self

    def _grow_tree(self, X, y, depth):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            return Node(value=self._most_common_label(y))

        best_feat, best_thresh, best_gini = self._best_split(X, y)

        if best_feat is None:
            return Node(value=self._most_common_label(y))

        parent_gini = self._gini(y)
        impurity_decrease = (parent_gini - best_gini) * n_samples
        self.feature_importances_[best_feat] += impurity_decrease

        left_idxs = np.where(X[:, best_feat] <= best_thresh)[0]
        right_idxs = np.where(X[:, best_feat] > best_thresh)[0]

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y):
        best_gini = float("inf")
        best_feat, best_thresh = None, None
        n_features = X.shape[1]

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thresh in thresholds:
                left_idxs = np.where(X_column <= thresh)[0]
                right_idxs = np.where(X_column > thresh)[0]

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                gini = self._calculate_gini_split(y, left_idxs, right_idxs)

                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh, best_gini

    def _gini(self, y_subset):
        _, counts = np.unique(y_subset, return_counts=True)
        probabilities = counts / len(y_subset)
        return 1.0 - float(np.sum(probabilities ** 2))

    def _calculate_gini_split(self, y, left_idxs, right_idxs):
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini_l = self._gini(y[left_idxs])
        gini_r = self._gini(y[right_idxs])
        return (n_l / n) * gini_l + (n_r / n) * gini_r

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
