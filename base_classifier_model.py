import numpy as np


class BaseClassifierModel:
    def train(self, X, Y) -> None:
        self.p, self.N = X.shape
        self.C = Y.shape[0]
        self.Ni = np.sum(Y == 1, axis=1)  # shape: (C,)
        self.priori = self.Ni / self.N  # shape: (C,)
        self.M = np.array([
            np.mean(X[:, Y[i] == 1], axis=1) for i in range(self.C)
        ])  # shape: (C, p)
        self.cov = None

        assert np.isclose(np.sum(self.priori), 1), f"Sum of priori probabilities is not 1: {np.sum(self.priori)}"
        assert self.M.shape == (self.C, self.p), f"Shape of M is not (C, p): {self.M.shape}"
        assert self.Ni.shape == (self.C,), f"Shape of Ni is not (C,): {self.Ni.shape}"
        assert self.priori.shape == (self.C,), f"Shape of priori is not (C,): {self.priori.shape}"

    def predict(self, Xi: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def _x_of_class_i(X, Y, i) -> np.ndarray:
        return X[:, Y[i, :] == 1]
