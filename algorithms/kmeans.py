import numpy as np


class KMeans:
    """K-Means clustering algorithm."""

    def __init__(self, n_clusters=3, max_iterations=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol
        self.centroids = None
        self.labels_ = None

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def _init_centroids(self, X):
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X):
        distances = np.array([self._euclidean_distance(X, centroid) for centroid in self.centroids])
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
        return new_centroids

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iterations):
            old_centroids = self.centroids.copy()
            self.labels_ = self._assign_clusters(X)
            self.centroids = self._update_centroids(X, self.labels_)

            if np.all(np.abs(self.centroids - old_centroids) < self.tol):
                break

        return self

    def predict(self, X):
        return self._assign_clusters(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def inertia(self, X):
        labels = self._assign_clusters(X)
        return sum(np.sum((X[labels == k] - self.centroids[k]) ** 2) for k in range(self.n_clusters))
