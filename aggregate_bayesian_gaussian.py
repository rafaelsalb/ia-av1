import numpy as np

from base_classifier_model import BaseClassifierModel


class AggregateBayesianGaussianClassifier(BaseClassifierModel):
    def train(self, X, Y) -> None:
        super().train(X, Y)

        self.cov = np.zeros((self.p, self.p))

        for i in range(self.C):
            xi = self._x_of_class_i(X, Y, i)
            self.cov += self.priori[i] * np.cov(xi.T, rowvar=False)

        self.cov_inv = np.linalg.pinv(self.cov)

        assert self.cov.shape == (self.p, self.p), f"Shape of cov is not (p, p): {self.cov.shape}"
        assert self.cov_inv.shape == (self.p, self.p), f"Shape of cov_inv is not (p, p): {self.cov_inv.shape}"

    def predict(self, Xi: np.ndarray) -> np.ndarray:
        p, N = Xi.shape
        predictions = np.zeros((N, 1))

        for i in range(N):
            posteriors = np.zeros((N, self.C))
            for c in range(self.C):
                posteriors[:, c] = self.likelihood(self.M[c], self.cov, Xi[:, i])
            prediction = np.argmin(posteriors[i])
            predictions[i] = prediction

        return predictions

    @staticmethod
    def likelihood(mi: float, cov_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        _x = x

        diff = _x - mi
        diff = np.reshape(diff, (diff.shape[0], 1))
        mahalanobis = diff.T @ cov_inv @ diff

        return mahalanobis
