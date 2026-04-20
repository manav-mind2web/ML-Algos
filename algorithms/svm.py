import numpy as np


class SVM:
    """Support Vector Machine classifier using gradient descent."""

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * y_[idx]

        return self

    def predict(self, X):
        linear = np.dot(X, self.weights) - self.bias
        return np.sign(linear).astype(int)

    def score(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        return np.mean(self.predict(X) == y_)
