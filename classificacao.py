import json
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np

from aggregate_bayesian_gaussian import AggregateBayesianGaussianClassifier
from base_classifier_model import BaseClassifierModel
from bayesian_gaussian import BayesianGaussianClassifier

import numpy as np
import matplotlib.pyplot as plt

from classifier_monte_carlo import ClassifierMonteCarlo
from least_squares_classifier import LeastSquaresClassifier
from linear_regression import RegressionMethods
from monte_carlo import MonteCarlo
from monte_carlo_criteria import MonteCarloCriteria
from naive_bayesian_gaussian import NaiveBayesianGaussianClassifier
from regularized_bayesian_gaussian import RegularizedBayesianGaussianClassifier
from single_covariance_bayesian_gaussian import SingleCovarianceBayesianGaussianClassifier


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='%(asctime)s %(message)s')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logger.info("starting...")


def plot_decision_boundaries(classifier, X, Y, title, class_labels, is_least_squares=False):
    logger.info("Plotting")
    plt.figure(title, figsize=(8, 6))

    logger.info("Defining grid")
    # Define the grid range
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    logger.info("Grid Points")
    # Predict class for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    logger.info("Predicting")
    grid_points = grid_points if is_least_squares else grid_points.T
    predictions = classifier.predict(grid_points)
    logger.info("Reshaping")
    predictions = predictions.reshape(xx.shape)

    logger.info("Painting areas")
    # Plot decision regions
    plt.contourf(xx, yy, predictions, alpha=0.3, cmap='inferno')

    logger.info("Plotting scatter")
    # Plot training points with color mapping
    scatter = plt.scatter(X[0, :], X[1, :], c=np.argmax(Y.T, axis=1), cmap='inferno', edgecolor='k', s=20)

    logger.info("Graph props")
    # Labels and title
    plt.xlabel("Corrugador do Supercílio")
    plt.ylabel("Zigomático Maior")
    plt.title(title)

    logger.info("Legend")
    # Create a legend with custom labels and colors
    handles = []
    for i, label in enumerate(class_labels):
        color = plt.cm.inferno(i / len(class_labels))  # Get color from the colormap
        handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        handles.append(handle)

    logger.info("Saving...")
    plt.legend(handles=handles, title="Class", loc="upper right")
    plt.savefig(f"images\\{title}-{n}.png")
    logger.info("Finished plotting.")


n = 50_000
data = np.loadtxt("EMGsDataset.csv", delimiter=",")[:, 0:n]

X = data[0:-1, :][:]  # shape: (p, N)
p, N = X.shape

colors = ['red', 'green', 'blue', 'magenta', 'yellow']
classes = data[-1, :]
C = np.unique(classes).shape[0]
Y = -1 * np.ones((C, N))
for i in range(N):
    j = int(classes[i] - 1)
    Y[j, i] = 1

# plt.figure("Visualização do Conjunto de Dados")
# plt.title("Visualização do Conjunto de Dados")
# plt.scatter(
#     X[:, 0],
#     X[:, 1],
#     c=np.argmax(Y, axis=1),
#     cmap='inferno',
#     s=20
# )

plot = lambda model, title, is_least_squares=False: (
    plot_decision_boundaries(model, X, Y, title,
                             ["Neutro", "Sorriso", "Sobrancelhas Levantadas", "Surpreso", "Rabugento"],
                             is_least_squares
    )
)

def monte_carlo_and_save(model: BaseClassifierModel, X, Y, title):
    monte_carlo = ClassifierMonteCarlo.run(500, model, 0.8, X, Y)
    logger.info(f"{title}\t", monte_carlo)
    with open(f"monte_carlo\\{title}-{n}.json", "w") as f:
        json.dump(monte_carlo, f)
    model.train(X, Y)
    plot(model, title)

# trad_model = BayesianGaussianClassifier()
# monte_carlo_and_save(trad_model, X, Y, "Bayesian Gaussian Classifier")

# agg_model = AggregateBayesianGaussianClassifier()
# monte_carlo_and_save(agg_model, X, Y, "Aggregate Bayesian Gaussian Classifier")

# single_model = SingleCovarianceBayesianGaussianClassifier()
# monte_carlo_and_save(single_model, X, Y, "Single Covariance Bayesian Gaussian Classifier")

# naive_model = NaiveBayesianGaussianClassifier()
# monte_carlo_and_save(naive_model, X, Y, "Naive Bayesian Gaussian Classifier")

ls_model = LeastSquaresClassifier()
ls_monte_carlo = MonteCarlo(X.T, Y.T, 0.8, 500)
stats = ls_monte_carlo.run(RegressionMethods.LEAST_SQUARES, MonteCarloCriteria.CLASSIFICATION)
with open(f"monte_carlo\\Least Squares-{n}.json", "w") as f:
    json.dump(stats, f)
print(stats)
ls_model.train(X.T, Y.T)
plot(ls_model, "MQO", True)

# for lbd in range(0, 101, 25):
#     reg_model = RegularizedBayesianGaussianClassifier(lbd / 100)
#     monte_carlo_and_save(reg_model, X, Y, f"Regularized Bayesian Gaussian Classifier (λ = {lbd / 100})")
