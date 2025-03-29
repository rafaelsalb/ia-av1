import numpy as np

from base_classifier_model import BaseClassifierModel


class SingleCovarianceBayesianGaussianClassifier(BaseClassifierModel):
    def train(self, X, Y) -> None:
        super().train(X, Y)
        self.cov = np.cov(X)
        self.cov_inv = np.linalg.pinv(self.cov)

    def predict(self, Xi: np.ndarray) -> np.ndarray:
        p, N = Xi.shape
        predictions = np.zeros((N, 1))

        for i in range(N):
            posteriors = np.zeros((N, self.C))
            for c in range(self.C):
                posteriors[:, c] = self.likelihood(self.M[c], self.cov_inv, Xi[:, i])
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
