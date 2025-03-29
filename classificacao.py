import json
import sys
import matplotlib.pyplot as plt
import numpy as np

from aggregate_bayesian_gaussian import AggregateBayesianGaussianClassifier
from bayesian_gaussian import BayesianGaussianClassifier

import numpy as np
import matplotlib.pyplot as plt

from classifier_monte_carlo import ClassifierMonteCarlo
from least_squares_classifier import LeastSquaresClassifier
from naive_bayesian_gaussian import NaiveBayesianGaussianClassifier
from regularized_bayesian_gaussian import RegularizedBayesianGaussianClassifier
from single_covariance_bayesian_gaussian import SingleCovarianceBayesianGaussianClassifier


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_decision_boundaries(classifier, X, Y, title, class_labels):
    plt.figure(title, figsize=(8, 6))

    # Define the grid range
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Predict class for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = classifier.predict(grid_points.T)
    predictions = predictions.reshape(xx.shape)

    # Plot decision regions
    plt.contourf(xx, yy, predictions, alpha=0.3, cmap='inferno')

    # Plot training points with color mapping
    scatter = plt.scatter(X[0, :], X[1, :], c=np.argmax(Y.T, axis=1), cmap='inferno', edgecolor='k', s=20)

    # Labels and title
    plt.xlabel("Corrugador do Supercílio")
    plt.ylabel("Zigomático Maior")
    plt.title(title)

    # Create a legend with custom labels and colors
    handles = []
    for i, label in enumerate(class_labels):
        color = plt.cm.inferno(i / len(class_labels))  # Get color from the colormap
        handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        handles.append(handle)

    plt.legend(handles=handles, title="Class", loc="upper right")
    plt.savefig(f"images/{title}-{n}.png")


n = 50000
data = np.loadtxt("EMGsDataset.csv", delimiter=",")[:, 0:n]

X = data[0:-1, :][:]  # shape: (p, N)
p, N = X.shape
print(p, N)

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

plot = lambda model, title: (
    plot_decision_boundaries(model, X, Y, title,
                             ["Neutro", "Sorriso", "Sobrancelhas Levantadas", "Surpreso", "Rabugento"]
    )
)

def monte_carlo_and_save(model, X, Y, title):
    print("Model\tMean\tStd\tMax\tMin")
    monte_carlo = ClassifierMonteCarlo.run(500, model, 0.8, X, Y)
    print(f"{title}\t", monte_carlo)
    with open(f"monte_carlo/{title}-{n}.json", "w") as f:
        json.dump(monte_carlo, f)
    plot(model, title)

print("Model\tMean\tStd\tMax\tMin")
trad_model = BayesianGaussianClassifier()
monte_carlo_and_save(trad_model, X, Y, "Bayesian Gaussian Classifier")

print("Model\tMean\tStd\tMax\tMin")
agg_model = AggregateBayesianGaussianClassifier()
monte_carlo_and_save(agg_model, X, Y, "Aggregate Bayesian Gaussian Classifier")

print("Model\tMean\tStd\tMax\tMin")
single_model = SingleCovarianceBayesianGaussianClassifier()
monte_carlo_and_save(single_model, X, Y, "Single Covariance Bayesian Gaussian Classifier")

print("Model\tMean\tStd\tMax\tMin")
naive_model = NaiveBayesianGaussianClassifier()
monte_carlo_and_save(naive_model, X, Y, "Naive Bayesian Gaussian Classifier")


for lbd in range(0, 101, 25):
    reg_model = RegularizedBayesianGaussianClassifier(lbd / 100)
    monte_carlo_and_save(reg_model, X, Y, f"Regularized Bayesian Gaussian Classifier (λ = {lbd / 100})")
