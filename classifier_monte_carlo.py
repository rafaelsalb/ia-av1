import numpy as np

from base_classifier_model import BaseClassifierModel


class ClassifierMonteCarlo:
    @staticmethod
    def run(rounds: int, model: BaseClassifierModel, ratio: float, X: np.ndarray, Y: np.ndarray, least_squares: bool = False):
        _X = X
        _Y = Y
        results = []
        p, N = X.shape
        for i in range(rounds):
            if i % 100 == 0:
                print(f"Round {i}")
            perm = np.random.permutation(N)
            Xr = _X[:, perm]
            Yr = _Y[:, perm]

            X_train = Xr[:, 0: int(N * ratio)]
            Y_train = Yr[:, 0: int(N * ratio)]

            X_test = Xr[:, int(N * ratio):]
            Y_test = Yr[:, int(N * ratio):]

            model.train(X_train, Y_train)

            pred = model.predict(X_test)
            pred = np.reshape(pred, (pred.shape[0], 1))

            correct_predictions = np.sum(pred[:] == np.argmax(Y_test, axis=1))
            total_predictions = Y_test.shape[1]

            score = correct_predictions / total_predictions

            results.append(score)
        else:
            print(f"Round {rounds}")

        diffs = np.array(results)
        result = {
            "mean": np.mean(diffs),
            "std": np.std(diffs),
            "max": np.max(diffs),
            "min": np.min(diffs)
        }
        return result
