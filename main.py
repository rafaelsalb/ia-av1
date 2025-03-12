import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("av1/atividade_enzimatica.csv", delimiter=",")


X = data[:, 0:2]
Y = data[:, 2:]

N, p = X.shape

print(N, p)

plt.figure(1)
ax = plt.subplot(projection="3d")

ax.set_title("Dados")
ax.scatter(data[:,0], data[:,1], data[:,2], color="magenta", edgecolor="k")

plt.figure(2)
ax = plt.subplot(1, 2, 1, projection="3d")

B_media = np.array([
    [np.mean(Y)],
    [0],
    [0]
])

x_axis = np.linspace(np.min(X[:,0]), np.max(X[:,0]),100)
y_axis = np.linspace(np.min(X[:,1]), np.max(X[:,1]),100)

X3D, Y3D = np.meshgrid(x_axis, y_axis)

X_plot = np.concatenate(
    (
        np.ones((100, 100, 1)),
        np.reshape(X3D, (100, 100, 1)),
        np.reshape(Y3D, (100, 100, 1)),
    ),
    axis=2
)

Z = X_plot @ B_media

ax.set_title("Média das variáveis independentes")
ax.scatter(data[:, 0], data[:, 1], data[:, 2], color="magenta", edgecolor="k")
ax.plot_surface(X3D, Y3D, Z[:, :, 0], color="cyan", alpha=0.1, edgecolors='k', rstride=20, cstride=20)

ax = plt.subplot(1, 2, 2, projection="3d")

X = np.concatenate(
    (
        np.ones((N, 1)),
        X
    ),
    axis=1
)

B_MQO = np.linalg.pinv(X.T @ X) @ X.T @ Y
Z = X_plot @ B_MQO

ax.set_title("MQO")
ax.scatter(data[:, 0], data[:, 1], data[:, 2], color="magenta", edgecolor="k")
ax.plot_surface(X3D, Y3D, Z[:, :, 0], color="cyan", alpha=0.1, edgecolors='k', rstride=20, cstride=20)


plt.figure(3)
hyper = [0.0, 0.25, 0.5, 0.75, 1.0]

for i, h in enumerate(hyper):
    ax = plt.subplot(2, 3, i + 1, projection="3d")

    B_MQO_tk = np.linalg.inv((X.T @ X) + (h * np.eye(p + 1))) @ X.T @ Y

    Z = X_plot @ B_MQO_tk

    ax.set_title(f"MQO Regularizado, $\lambda$ = {h}")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color="magenta", edgecolor="k")
    ax.plot_surface(X3D, Y3D, Z[:, :, 0], color="cyan", alpha=0.1, edgecolors='k', rstride=20, cstride=20)

plt.show()
