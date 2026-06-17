import numpy as np


class CSP:
    def fit(self, X, y):
        # placeholder CSP implementation
        self.W = np.eye(X.shape[1])
        return self

    def transform(self, X):
        return np.dot(X, self.W)
