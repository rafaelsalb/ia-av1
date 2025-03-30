from enum import Enum

import numpy as np


class MonteCarloCriteria(Enum):
    REGRESSION = 0
    CLASSIFICATION = 1


class MonteCarloEvaluator:
    @staticmethod
    def evaluate(preds, Y, criteria):
        match criteria:
            case MonteCarloCriteria.REGRESSION:
                return np.sum((Y - preds) ** 2)
            case MonteCarloCriteria.CLASSIFICATION:
                N = Y.shape[0]
                score = np.sum(np.argmax(preds[:], axis=1) == np.argmax(Y, axis=1)) / N
                return score
