import numpy as np

from base_classifier_model import BaseClassifierModel


class NaiveBayesianGaussianClassifier(BaseClassifierModel):
    def train(self, X, Y) -> None:
        super().train(X, Y)
        self.cov = np.diag(np.diag(np.cov(X, rowvar=True)))
        self.cov_det = np.linalg.det(self.cov) + 1e-8
        self.cov_inv = np.linalg.pinv(self.cov)
        assert self.cov.shape == (self.p, self.p), f"Expected shape {(self.p, self.p)}, got {self.cov.shape}"

    def predict(self, Xi: np.ndarray) -> np.ndarray:
        p, N = Xi.shape
        predictions = np.zeros((N, 1))

        for i in range(N):
            posteriors = np.zeros((N, self.C))
            for c in range(self.C):
                posteriors[:, c] = np.log(self.priori[c]) + self.likelihood(self.M[c], self.cov_inv, self.cov_det, Xi[:, i])
            prediction = np.argmax(posteriors[i])
            predictions[i] = prediction

        return predictions

    @staticmethod
    def likelihood(mi: float, cov_inv: np.ndarray, cov_det: np.ndarray, x: np.ndarray) -> np.ndarray:

        diff = x - mi
        diff = np.reshape(diff, (diff.shape[0], 1))
        mahalanobis = diff.T @ cov_inv @ diff

        return - 0.5 * np.log(cov_det) - 0.5 * mahalanobis
