import numpy as np


class LeastSquaresClassifier:
    def __init__(self):
        self.W = None

    def train(self, X, Y) -> None:
        N, self.p = X.shape
        N, self.C = Y.shape
        self.W = np.linalg.pinv(X.T @ X) @ X.T @ Y

    def predict(self, X) -> np.ndarray:
        preds = X @ self.W
        return np.argmax(preds, axis=1)
