import numpy as np


class LeastSquaresClassifier:
    def __init__(self, add_intercept=True):
        self.add_intercept = add_intercept
        self.W = None

    def train(self, X, Y) -> None:
        N, self.p = X.shape
        N, self.C = Y.shape
        if self.add_intercept:
            X = np.concatenate(
                (np.ones((N, 1)), X),
                axis=1
            )
        self.W = np.linalg.pinv(X.T @ X) @ X.T @ Y

        assert self.W.shape == (self.p + 1, self.C), f"W.shape was expected to be ({self.p + 1} x {self.C}) but is {self.W.shape}"

    def predict(self, X) -> np.ndarray:
        if self.add_intercept:
            _X = self._add_intercept(X)
        preds = _X @ self.W
        return np.argmax(preds, axis=1)

    def _add_intercept(self, X) -> np.ndarray:
        N, _ = X.shape
        return np.concatenate(
            (np.ones((N, 1)), X),
            axis=1
        )
