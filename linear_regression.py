from enum import Enum
import numpy as np


class RegressionMethods(Enum):
    MEAN = 0
    LEAST_SQUARES = 1
    TIKHONOV = 2


class LinearRegression:
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X: np.float32 = X
        self.Y: np.float32 = Y
        self.N, self.p = X.shape
        self.beta: np.float32 | None = None

    def train(self, method: RegressionMethods = RegressionMethods.LEAST_SQUARES, k: float = 0.0) -> None:
        match method:
            case RegressionMethods.MEAN:
                mi = np.array((np.mean(self.Y),))
                mi = mi.reshape((1, 1))
                beta = np.concatenate(
                    (mi, np.zeros((self.p, 1))),
                )
                self.beta = beta
            case RegressionMethods.LEAST_SQUARES:
                X = np.concatenate(
                    (np.ones((self.N, 1)), self.X),
                    axis=1
                )
                self.beta = np.linalg.pinv(X.T @ X) @ X.T @ self.Y
            case RegressionMethods.TIKHONOV:
                X = np.concatenate(
                    (np.ones((self.N, 1)), self.X),
                    axis=1
                )
                self.beta = np.linalg.pinv((X.T @ X) + (k * np.eye(self.p + 1))) @ X.T @ self.Y

    def predict(self, X: np.float32) -> np.float32:
        return X @ self.beta

