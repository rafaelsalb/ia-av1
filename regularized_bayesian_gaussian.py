import numpy as np

from base_classifier_model import BaseClassifierModel


class RegularizedBayesianGaussianClassifier(BaseClassifierModel):
    def __init__(self, lambda_reg: float):
        self.lambda_reg = lambda_reg

    def train(self, X, Y) -> None:
        super().train(X, Y)
        self.cov = np.zeros((self.C, self.p, self.p))
        agg_cov = np.zeros((self.p, self.p))
        for i in range(self.C):
            xi = self._x_of_class_i(X, Y, i)
            cov = np.cov(xi.T, rowvar=False) + 1e-8 * np.eye(self.p)
            self.cov[i] = cov
            agg_cov += self.priori[i] * cov

        self.reg_cov = np.zeros((self.C, self.p, self.p))
        for i in range(self.C):
            self.reg_cov[i] = ((1 - self.lambda_reg) * (self.Ni[i] * self.cov[i]) + self.lambda_reg * (self.N * agg_cov)) / ((1 - self.lambda_reg) * self.Ni[i] + self.lambda_reg * self.N)
        self.cov_det = np.zeros((self.C,))
        self.cov_inv = np.zeros((self.C, self.p, self.p))
        for i in range(self.C):
            self.cov_det[i] = np.linalg.det(self.reg_cov[i]) + 1e-8
            self.cov_inv[i] = np.linalg.pinv(self.reg_cov[i])

        assert self.cov.shape == (self.C, self.p, self.p), f"Shape of cov is not (C, p, p): {self.cov.shape}"
        assert agg_cov.shape == (self.p, self.p), f"Shape of agg_cov is not (p, p): {self.agg_cov.shape}"
        assert self.reg_cov.shape == (self.C, self.p, self.p), f"Shape of reg_cov is not (C, p, p): {self.reg_cov.shape}"
        assert self.cov_det.shape == (self.C,), f"Shape of cov_det is not (C,): {self.cov_det.shape}"
        assert self.cov_inv.shape == (self.C, self.p, self.p), f"Shape of cov_inv is not (C, p, p): {self.cov_inv.shape}"

    def predict(self, Xi: np.ndarray) -> np.ndarray:
        p, N = Xi.shape
        predictions = np.zeros((N, 1))

        for i in range(N):
            posteriors = np.zeros((1, self.C))
            for c in range(self.C):
                posteriors[:, c] = self.likelihood(self.M[c], self.cov_inv[c], self.cov_det[c], Xi[:, i])
            prediction = np.argmax(posteriors, axis=1)
            predictions[i] = prediction

        return predictions

    @staticmethod
    def likelihood(mi: float, cov_inv: np.ndarray, cov_det: np.ndarray, x: np.ndarray) -> np.ndarray:
        _x = x

        diff = x.T - mi
        diff = np.reshape(diff, (diff.shape[0], 1))
        mahalanobis = diff.T @ cov_inv @ diff

        return - 0.5 * np.log(cov_det) - 0.5 * mahalanobis

