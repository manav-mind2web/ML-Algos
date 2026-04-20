import numpy as np
from collections import Counter


class DecisionTree:
    """Decision Tree classifier using ID3 algorithm."""

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _information_gain(self, X_column, y, threshold):
        parent_entropy = self._entropy(y)

        left_mask = X_column <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0

        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        e_left, e_right = self._entropy(y[left_mask]), self._entropy(y[right_mask])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        return parent_entropy - child_entropy

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(X[:, feature_idx], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]

        left_mask = X[:, feature] <= threshold
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return {'feature': feature, 'threshold': threshold, 'left': left, 'right': right}

    def fit(self, X, y):
        self.root = self._build_tree(X, y.astype(int))
        return self

    def _predict_single(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        return self._predict_single(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
