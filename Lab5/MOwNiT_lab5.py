import numpy as np
import pandas as pd
from math import pi, cos, sin
import matplotlib.pyplot as plot

k = 1
m = 2

min_x = 0
max_x = 3 * pi


def f(x):
    if not isinstance(x, float):
        return [sin(m * i) * sin(k * i ** 2 / pi) for i in x]
    return sin(m * x) * sin(k * x ** 2 / pi)


def aprox(nodes: list, n: int, X: list, m: int, w: list = None) -> list:
    if w is None:
        w = [1 for _ in range(n + 1)]  # wages

    vector = [0 for _ in range(m + 1)]  # B

    for k in range(m + 1):
        for i in range(n):
            vector[k] += w[i] * f(nodes[i]) * (nodes[i] ** k)

    matrix = [[0 for _ in range(m + 1)] for _ in range(m + 1)]  # G

    for k in range(m + 1):
        for j in range(m + 1):
            for i in range(n):
                matrix[k][j] += w[i] * (nodes[i] ** (k + j))

    A = np.linalg.solve(matrix, vector)

    ans = [0 for _ in range(len(X))]

    for i in range(len(X)):
        elem = 0
        for k in range(m + 1):
            elem += A[k] * X[i] ** k
        ans[i] = elem
    print(len(ans))
    return ans


def drawFunction():
    plot.plot(X, f(X), label="Funckja")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


def drawAprox(nodes: list, X: list, m: int, w: list = None):
    n = len(nodes)
    plot.suptitle("Aproksymacja stopnia " + str(m) + " na " + str(n) + " węzłach równoległych")
    plot.plot(X, f(X), label="Funckja")
    plot.plot(X, aprox(nodes, n, X, m), label="Aproksymacja")
    plot.scatter(nodes, f(nodes), color="red", label="Węzły")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


n = 20
X = np.arange(min_x, max_x + 0.01, 0.01)
nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))



drawAprox(nodes, X,6)
drawAprox(nodes, X,20)
