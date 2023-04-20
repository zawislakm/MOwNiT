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
    plot.suptitle("Aproksymacja na " + str(n) + " węzłach równoległych")
    plot.plot(X, f(X), label="Funckja")
    plot.plot(X, aprox(nodes, n, X, m), label="Aproksymacja m=" + str(m))
    plot.scatter(nodes, f(nodes), color="red", label="Węzły")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


def drawAproxBetween(X: list):
    n1 = 40
    n2 = 200
    ems = [2, 7, 9]
    fig, axs = plot.subplots(3, 2)
    fig.suptitle("Porównanie aproksymacji")
    n1nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n1 - 1))
    n2nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n2 - 1))

    for i in range(len(ems)):
        axs[i][0].plot(X, f(X), label="Funckja")
        axs[i][0].plot(X, aprox(n1nodes, n1, X, ems[i]), label="Aproksymacja m= " + str(ems[i]))
        axs[i][0].scatter(n1nodes, f(n1nodes), color="red", label="Węzły")
        axs[i][0].set_xlabel("X")
        axs[i][0].set_ylabel("Y")
        axs[i][0].set_title("Aproksymacja na " + str(n1) + " węzłach")
        axs[i][0].legend()

        axs[i][1].plot(X, f(X), label="Funckja")
        axs[i][1].plot(X, aprox(n2nodes, n2, X, ems[i]), label="Aproksymacja m= " + str(ems[i]))
        axs[i][1].scatter(n2nodes, f(n2nodes), color="red", label="Węzły")
        axs[i][1].set_xlabel("X")
        axs[i][1].set_ylabel("Y")
        axs[i][1].set_title("Aproksymacja na " + str(n2) + " węzłach")
        axs[i][1].legend()
    fig.set_size_inches(15, 10)
    fig.tight_layout()
    plot.show()


def drawAproxAll(nodes: list, X: list):
    ems = [2, 3, 5, 7, 8, 9]
    n = len(nodes)
    fig, axs = plot.subplots(2, 3)
    fig.suptitle("Aproksymacje na " + str(n) + " węzłach równoległych")
    for i in range(len(ems)):
        row = i // 3
        col = i % 3
        axs[row][col].plot(X, f(X), label="Funckja")
        axs[row][col].plot(X, aprox(nodes, n, X, ems[i]), label="Aproksymacja m= " + str(ems[i]))
        axs[row][col].scatter(nodes, f(nodes), color="red", label="Węzły")
        axs[row][col].set_xlabel("X")
        axs[row][col].set_ylabel("Y")
        # axs[row][col].set_title("ems[" + str(i) + "] = " + str(ems[i]))
        axs[row][col].legend()

    tableAproxGiven(X, n, ems)
    fig.set_size_inches(15, 10)
    plot.show()


def drawAproxAllN(X: list, m: int):
    # ems = [2, 3, 5, 7, 8, 9]
    ens = [10, 15, 20, 30, 40, 60, 80, 100, 200]

    fig, axs = plot.subplots(3, 3)
    fig.suptitle("Aproksymacja z " + str(m) + " funkcjami bazowymi")
    for i in range(len(ens)):
        nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (ens[i] - 1))
        row = i // 3
        col = i % 3
        axs[row][col].plot(X, f(X), label="Funckja")
        axs[row][col].plot(X, aprox(nodes, ens[i], X, m), label="Aproksymacja n=" + str(ens[i]))
        axs[row][col].scatter(nodes, f(nodes), color="red", label="Węzły")
        axs[row][col].set_xlabel("X")
        axs[row][col].set_ylabel("Y")
        # axs[row][col].set_title("Liczba węzłów " + str(ens[i]))
        axs[row][col].legend()

    fig.set_size_inches(15, 10)
    fig.tight_layout()
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


def tableAproxGiven(X, n, ems):
    outcome = []

    parallel_nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
    for em in ems:
        if em > n:
            break
        ans = aprox(parallel_nodes, n, X, em)
        outcome.append(
            [em, diffrecne(X, ans, n), sqrdiffrence(X, ans, n)])

    df = pd.DataFrame(outcome,
                      columns=["m", "Natural max error", "Natural square error"])

    filename = 'results' + str(n) + '.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["m", "Natural max error", "Natural square error"])
        for row in outcome:
            writer.writerow(row)
    return df


def tableAprox(X):
    ens = [10, 20, 30, 40, 60, 80, 100, 200]
    ems = [2, 3, 5, 7, 8, 9]

    outcome = []
    dif_outcome = [[0 for _ in range(len(ems) + 1)] for _ in range(len(ens) + 1)]
    sqdif_outcome = [[0 for _ in range(len(ems) + 1)] for _ in range(len(ens) + 1)]

    for i in range(1, len(ens) + 1):
        dif_outcome[i][0] = ens[i - 1]
        sqdif_outcome[i][0] = ens[i - 1]

    for j in range(1, len(ems) + 1):
        dif_outcome[0][j] = ems[j - 1]
        sqdif_outcome[0][j] = ems[j - 1]

    for i in range(1, len(ens) + 1):
        n = ens[i - 1]
        parallel_nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
        for j in range(1, len(ems) + 1):
            em = ems[j - 1]
            ans = aprox(parallel_nodes, n, X, em)
            dif = round(diffrecne(X, ans, n), 5)
            sqdif = round(sqrdiffrence(X, ans, n), 5)

            outcome.append([n, em, dif, sqdif])

            dif_outcome[i][j] = dif
            sqdif_outcome[i][j] = sqdif

    df = pd.DataFrame(outcome, columns=["n", "m", "Natural max error", "Natural square error"])

    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["n", "m", "Natural max error", "Natural square error"])
        for row in outcome:
            writer.writerow(row)

    with open('difresult.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in dif_outcome:
            writer.writerow(row)

    with open("sqdifresult.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        for row in sqdif_outcome:
            writer.writerow(row)
    return df


n = 10
mn = 8
X = np.arange(min_x, max_x + 0.01, 0.01)

nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))

drawFunction()

# tableAprox(X)
drawAproxAll(nodes, X)
drawAprox(nodes, X, mn)
drawAproxAllN(X, mn)
drawAprox(nodes, X, mn)
drawAproxBetween(X)
