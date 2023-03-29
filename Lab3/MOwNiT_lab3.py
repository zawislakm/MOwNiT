from math import pi, cos, sin
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

k = 1
m = 2

min_x = 0
max_x = 3 * pi


def Hermit(X, nodes, n, derivative):
    hermitParameters = getHermitParametrs(nodes, n, derivative)
    ans = [0 for _ in range(len(X))]
    p = [1 for _ in range(len(X))]

    for i in range(1, n * 2 + 1):
        for j in range(len(X)):
            ans[j] += p[j] * hermitParameters[i][i]
            p[j] *= (X[j] - hermitParameters[i][0])
    return ans


def getHermitParametrs(nodes, n, derivative):
    matrix = [[0 for _ in range(n * 2 + 1)] for _ in range(n * 2 + 1)]
    for i in range(1, n * 2 + 1, 1):
        index = (i - 1) // 2
        matrix[i][0] = nodes[index]
        matrix[i][1] = f(nodes[(i - 1) // 2])
        if i % 2 == 0:
            matrix[i][2] = derivative(nodes[index])

    for i in range(1, n * 2 + 1):
        now_node = nodes[(i - 1) // 2]
        for j in range(2, i + 1):
            if i % 2 == 1 or j > 2:
                matrix[i][j] = (matrix[i][j - 1] - matrix[i - 1][j - 1]) / (
                        now_node - nodes[(i - j) // 2])


    return matrix


def f(x):
    if not isinstance(x, float):
        return [sin(m * i) * sin(k * i ** 2 / pi) for i in x]
    return sin(m * x) * sin(k * x ** 2 / pi)


def df(x):  # pochodna
    if not isinstance(x, float):
        return [m * cos(m * i) * sin(k * i ** 2 / pi) + sin(m * i) * (2 * k * i / pi) * cos(k * i ** 2 / pi) for i in x]
    return m * cos(m * x) * sin(k * x ** 2 / pi) + sin(m * x) * (2 * k * x / pi) * cos(k * x ** 2 / pi)


def chebyshev_disrtibution(n, a, b):  # b > a
    # https://pl.wikipedia.org/wiki/W%C4%99z%C5%82y_Czebyszewa
    cheby_points = []
    for k in range(1, n + 1):
        xk = 0.5 * (b + a) + 0.5 * (b - a) * cos(((2 * k - 1) * np.pi) / (2 * n))
        cheby_points.append(xk)
    return cheby_points[::-1]


def drawFunction():
    plot.plot(X, f(X), label="Funckja")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


def drawHermite(nodes, n, derivative, type):
    plot.suptitle("Interpolacja Hermit'a na " + type + " na " + str(n) + " węzłach")
    plot.plot(X, f(X), label="Funckja")
    plot.plot(X, Hermit(X, nodes, n, derivative), label="Interpolacja")

    plot.xlabel("X")
    plot.ylabel("Y")
    plot.scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    plot.legend()
    plot.show()


def drawHermiteBothNodes(parallel_nodes, cheby_nodes, n, derivative):
    fig, axs = plot.subplots(1, 2)
    plot.suptitle("Interpolacaja Hermit'a na obu rodzajach węzłów z " + str(n) + " węzłami")
    axs[0].plot(X, f(X), label="Funkcja")
    axs[0].plot(X, Hermit(X, parallel_nodes, n, derivative), label="Interpolacja")
    axs[0].scatter(parallel_nodes, f(parallel_nodes), color="red", label="Punkty wspólne")
    axs[0].set_title("Interpolacja Hermit'a na węzłach równoległych")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    axs[1].plot(X, f(X), label="Funkcja")
    axs[1].plot(X, Hermit(X, cheby_nodes, n, derivative), label="Interpolacja")
    axs[1].scatter(cheby_nodes, f(cheby_nodes), color="red", label="Punkty wspólne")
    axs[1].set_title("Interpolacja Hermit'a na węzłach Czebyszewa")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].legend()

    fig.set_size_inches(14, 7)
    plot.show()


def diffrecne(X, interpoleted, n):
    ans = 0
    for i in range(0, len(X)):
        tmp = abs(f(X[i]) - interpoleted[i])
        ans = max(ans, tmp)
    return ans


def sqrdiffrence(X, interpoleted, n):
    ans = 0
    for i in range(0, len(X)):
        tmp = f(X[i]) - interpoleted[i]
        ans += (tmp ** 2)
    ans = (ans ** 0.5) / n
    return ans


def tableHermite(ens, X, derivative):
    outcome = []
    for n in ens:
        parallel_nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
        cheby_nodes = chebyshev_disrtibution(n, min_x, max_x)
        HP = Hermit(X, parallel_nodes, n, derivative)
        HC = Hermit(X, cheby_nodes, n, derivative)
        outcome.append([n, diffrecne(X, HP, n), sqrdiffrence(X, HP, n), diffrecne(X, HC, n), sqrdiffrence(X, HC, n)])

    df = pd.DataFrame(outcome,
                      columns=["n", "Max error Parallel Nodes", "Sum square error Parallel Nodes",
                               "Max error Chebyshev Nodes", "Sum square error Chebyshev Nodes"])
    print(df)
    return df


ens = [3, 4, 5, 7, 9, 10, 11, 12, 15, 20, 30, 40, 50, 60, 75]

# Interpolate using Hermite method
n = 14

cheby_nodes = chebyshev_disrtibution(n, min_x, max_x)
parallel_nodess = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
X = np.arange(min_x, max_x + 0.01, 0.01)
print(parallel_nodess, "nody")
type_p = "węzłach równoległych"
type_c = "węzłach Czebyszewa"
drawHermite(parallel_nodess, n, df, type_p)

drawHermiteBothNodes(parallel_nodess, cheby_nodes, n, df)

tableHermite(ens, X, df)
