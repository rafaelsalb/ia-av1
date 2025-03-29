import numpy as np

from base_classifier_model import BaseClassifierModel


class BayesianGaussianClassifier(BaseClassifierModel):
    def train(self, X, Y) -> None:
        super().train(X, Y)

        self.cov = np.zeros((self.C, self.p, self.p))

        for i in range(self.C):
            xi = self._x_of_class_i(X, Y, i)
            self.cov[i] = np.cov(xi.T, rowvar=False) + 1e-8 * np.eye(self.p)

        self.cov_det = np.zeros((self.C,))
        self.cov_inv = np.zeros((self.C, self.p, self.p))

        for i in range(self.C):
            self.cov_det[i] = np.linalg.det(self.cov[i]) + 1e-8
            self.cov_inv[i] = np.linalg.inv(self.cov[i])

        assert self.cov.shape == (self.C, self.p, self.p), f"Shape of cov is not (C, p, p): {self.cov.shape}"
        assert self.cov_det.shape == (self.C,), f"Shape of cov_det is not (C,): {self.cov_det.shape}"
        assert self.cov_inv.shape == (self.C, self.p, self.p), f"Shape of cov_inv is not (C, p, p): {self.cov_inv.shape}"

    def predict(self, Xi: np.ndarray) -> np.ndarray:
        p, N = Xi.shape
        predictions = np.zeros((N, 1))

        for i in range(N):
            posteriors = np.zeros((1, self.C))
            for c in range(self.C):
                posteriors[:, c] = np.log(self.priori[c]) + self.likelihood(self.M[c], self.cov_inv[c], self.cov_det[c], Xi[:, i])
            prediction = np.argmax(posteriors)
            predictions[i] = prediction

        return predictions

    @staticmethod
    def likelihood(mi: float, cov_inv: np.ndarray, cov_det: np.ndarray, x: np.ndarray) -> np.ndarray:

        diff = x.T - mi
        diff = np.reshape(diff, (diff.shape[0], 1))
        mahalanobis = diff.T @ cov_inv @ diff

        return - 0.5 * np.log(cov_det) - 0.5 * mahalanobis
