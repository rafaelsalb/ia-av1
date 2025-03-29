import numpy as np

from linear_regression import LinearRegression, RegressionMethods


class MonteCarlo:
    def __init__(self, X: np.ndarray, Y: np.ndarray, split: float, R: int):
        self.R = R
        self.split = split
        self.X = X
        self.Y = Y

    def run(self, regression_method: RegressionMethods, k: float | None = None) -> np.ndarray:
        results = []
        N, _ = self.X.shape
        for _ in range(self.R):
            perm = np.random.permutation(N)
            Xr = self.X[perm, :]
            Yr = self.Y[perm, :]

            X_train = Xr[0: int(N * self.split), :]
            Y_train = Yr[0: int(N * self.split), :]

            X_test = Xr[int(N * self.split):, :]
            N_test, _ = X_test.shape

            X_test = np.concatenate(
                (np.ones((N_test, 1)), X_test),
                axis=1
            )
            Y_test = Yr[int(N * self.split):, :]
            Y_test = np.concatenate(
                (np.ones((N_test, 1)), Y_test),
                axis=1
            )

            model = LinearRegression(X_train, Y_train)
            model.train(regression_method, k)
            pred = model.predict(X_test)
            score = np.sum((Y_test - pred) ** 2)
            results.append(score)
        diffs = np.array(results)
        results = np.array(
            [
                np.mean(diffs),
                np.std(diffs),
                np.max(diffs),
                np.min(diffs)
            ]
        )
        return results

