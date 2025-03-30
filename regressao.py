import json
import numpy as np
import matplotlib.pyplot as plt

from linear_regression import LinearRegression, RegressionMethods
from monte_carlo import MonteCarlo


data = np.loadtxt("atividade_enzimatica.csv", delimiter=",")


def predict_and_plot(data: np.ndarray, X: np.ndarray, sample_size: int, model: LinearRegression, axes: plt.Axes, **kwargs):
    x_axis = np.linspace(np.min(X[:,0]), np.max(X[:,0]), sample_size)
    y_axis = np.linspace(np.min(X[:,1]), np.max(X[:,1]), sample_size)

    X3D, Y3D = np.meshgrid(x_axis, y_axis)

    X_plot = np.concatenate(
        (
            np.ones((sample_size, sample_size, 1)),
            np.reshape(X3D, (sample_size, sample_size, 1)),
            np.reshape(Y3D, (sample_size, sample_size, 1)),
        ),
        axis=2
    )
    Z = model.predict(X_plot)

    axes.scatter(data[:, 0], data[:, 1], data[:, 2], color="magenta", edgecolor="k")
    axes.plot_surface(X3D, Y3D, Z[:, :, 0], **kwargs)

graph_settings = {
    "color": "cyan",
    "alpha": 0.1,
    "edgecolors": 'k',
    "rstride": 20,
    "cstride": 20
}


X = data[:, 0:2]
Y = data[:, 2:]

N, p = X.shape

plt.figure(1)
ax = plt.subplot(projection="3d")

ax.set_title("Dados")
ax.scatter(data[:,0], data[:,1], data[:,2], color="magenta", edgecolor="k")

model = LinearRegression(X, Y)

plt.figure(2)

ax = plt.subplot(1, 2, 1, projection="3d")
model.train(RegressionMethods.MEAN)
ax.set_title("Média das variáveis independentes")
predict_and_plot(data, X, 100, model, ax, **graph_settings)

ax = plt.subplot(1, 2, 2, projection="3d")
model.train(RegressionMethods.LEAST_SQUARES)
ax.set_title("MQO")
predict_and_plot(data, X, 100, model, ax, **graph_settings)


plt.figure(3)
hyper = [0.0, 0.25, 0.5, 0.75, 1.0]

for i, h in enumerate(hyper):
    ax = plt.subplot(2, 3, i + 1, projection="3d")
    model.train(RegressionMethods.TIKHONOV, h)
    ax.set_title(f"MQO Regularizado, $\\lambda$ = {h}")
    predict_and_plot(data, X, 100, model, ax, **graph_settings)

monte_carlo = MonteCarlo(X, Y, 0.8, 500)
mean = monte_carlo.run(RegressionMethods.MEAN)
with open(f"monte_carlo\\regressao-mean.json", "w") as f:
    json.dump(mean, f)
mqo = monte_carlo.run(RegressionMethods.LEAST_SQUARES)
with open(f"monte_carlo\\regressao-mqo.json", "w") as f:
    json.dump(mqo, f)
print("mean\t", monte_carlo.run(RegressionMethods.MEAN))
print("mqo\t", monte_carlo.run(RegressionMethods.LEAST_SQUARES))
for h in hyper:
    mqo_r = monte_carlo.run(RegressionMethods.TIKHONOV, k=h)
    with open(f"monte_carlo\\regressao-mqo_r{h}.json", "w") as f:
        json.dump(mqo_r, f)
    print(f"{h:.2f}\t", mqo_r)

plt.show()