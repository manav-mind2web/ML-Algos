import numpy as np
from collections import Counter


class KNearestNeighbors:
    """K-Nearest Neighbors classifier."""

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
