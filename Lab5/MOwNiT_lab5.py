import numpy as np
import pandas as pd
from math import pi, cos, sin
import matplotlib.pyplot as plot
import csv

k = 1
m = 2

min_x = 0
max_x = 3 * pi


def f(x):
    if not isinstance(x, float):
        return [sin(m * i) * sin(k * i ** 2 / pi) for i in x]
    return sin(m * x) * sin(k * x ** 2 / pi)


def drawFunction():
    plot.plot(X, f(X), label="Funkcja aproksymowana")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


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
    return ans


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


def tableAprox(X):
    ens = [4, 5, 7, 10, 15, 20, 30, 40, 70]
    ems = [2, 5, 8, 10, 15]
    outcome = []

    for n in ens:
        parallel_nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
        for em in ems:
            ans = aprox(parallel_nodes, n, X, em)
            outcome.append(
                [n, em, diffrecne(X, ans, n), sqrdiffrence(X, ans, n)])

    df = pd.DataFrame(outcome,
                      columns=["n", "m", "Natural max error", "Natural square error"])

    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["n", "m", "Natural max error", "Natural square error"])
        for row in outcome:
            writer.writerow(row)
    return df


n = 20
mn = 10
X = np.arange(min_x, max_x + 0.01, 0.01)
nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))

# drawFunction()
drawAprox(nodes, X, mn)


