import numpy as np
from collections import Counter
from .decision_tree import DecisionTree


class RandomForest:
    """Random Forest classifier using bootstrap aggregation."""

    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        self.feature_indices = []
        n_features = X.shape[1]
        max_features = self.max_features or int(np.sqrt(n_features))

        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)

            feature_idx = np.random.choice(n_features, max_features, replace=False)
            self.feature_indices.append(feature_idx)

            tree.fit(X_sample[:, feature_idx], y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X):
        predictions = np.array([
            tree.predict(X[:, self.feature_indices[i]])
            for i, tree in enumerate(self.trees)
        ])
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
