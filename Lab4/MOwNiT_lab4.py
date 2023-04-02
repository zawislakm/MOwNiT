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


def chebyshev_disrtibution(n, a, b):  # b > a
    # https://pl.wikipedia.org/wiki/W%C4%99z%C5%82y_Czebyszewa
    cheby_points = []
    for k in range(1, n + 1):
        xk = 0.5 * (b + a) + 0.5 * (b - a) * cos(((2 * k - 1) * np.pi) / (2 * n))
        cheby_points.append(xk)
    return cheby_points[::-1]


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


def hi(i, nodes):
    h = nodes[i] - nodes[i - 1]  # dif between nodes
    return h


def delta(i, nodes):
    d = (f(nodes[i]) - f(nodes[i - 1])) / hi(i, nodes)
    return d


def cubic_spline(X, nodes, n, type=0):
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    vector = [0 for _ in range(n)]

    def delta2(i, nodes):
        return (delta(i + 1, nodes) - delta(i, nodes)) / (nodes[i + 1] - nodes[i - 1])

    def delta3(i, nodes):
        return (delta2(i + 1, nodes) - delta2(i, nodes)) / (nodes[i + 2] - nodes[i - 1])

    for i in range(1, n - 1):  # 1 brzegi3
        matrix[i][i - 1] = hi(i, nodes)
        matrix[i][i] = 2 * (hi(i, nodes) + hi(i + 1, nodes))
        matrix[i][i + 1] = hi(i + 1, nodes)
        vector[i] = delta(i + 1, nodes) - delta(i, nodes)

    if type == 0:
        # natural
        matrix[0][0] = 1
        matrix[0][1] = 0
        matrix[n - 1][n - 2] = 0
        matrix[n - 1][n - 1] = 1
        vector[0] = 0
        vector[n - 1] = 0
    else:
        # clamped
        matrix[0][0] = -1 * hi(1, nodes)
        matrix[0][1] = hi(1, nodes)
        vector[0] = hi(1, nodes) ** 2 * delta3(1, nodes)

        matrix[n - 1][n - 2] = hi(n - 1, nodes)
        matrix[n - 1][n - 1] = -1 * hi(n - 1, nodes)
        vector[n - 1] = -1 * hi(n - 1, nodes) ** 2 * delta3(n - 3, nodes)

    ROs = np.linalg.solve(matrix, vector)

    def bi(i):
        h = hi(i, nodes)
        y_next = f(nodes[i])
        y = f(nodes[i - 1])
        ro_next = ROs[i]
        ro = ROs[i - 1]
        b = (y_next - y) / h - h * (ro_next + 2 * ro)
        return b

    def ci(i):
        ro = ROs[i - 1]
        c = 3 * ro
        return c

    def di(i):
        h = hi(i, nodes)
        ro_next = ROs[i]
        ro = ROs[i - 1]
        d = (ro_next - ro) / h
        return d

    ans = [0 for _ in range(len(X))]
    for j in range(len(X)):
        i = 0
        while i < len(nodes) - 1 and nodes[i] < X[j]:
            i += 1
        y = f(nodes[i - 1])
        b = bi(i)
        c = ci(i)
        d = di(i)
        dif = (X[j - 1] - nodes[i - 1])
        ans[j] = (y + b * dif + c * dif ** 2 + d * dif ** 3)

    return ans


def quadratic_spline(X, nodes, n, type=0):
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    vector = [0 for _ in range(n)]

    for i in range(1, n):
        matrix[i][i - 1] = 1
        matrix[i][i] = 1
        vector[i] = 2 * delta(i, nodes)

    # change types
    if type == 0:
        # natural
        matrix[0][0] = 1
        matrix[0][1] = 0
        vector[0] = 0
    else:
        # clamped
        matrix[0][0] = 1
        matrix[0][1] = 0
        vector[0] = delta(2, nodes)

    matrix[n - 1][n - 2] = 1
    matrix[n - 1][n - 1] = 1

    ROs = np.linalg.solve(matrix, vector)

    def bi(i):
        b = ROs[i - 1]
        return b

    def ci(i):
        b2_next = bi(i + 1)
        b2 = bi(i)
        h = hi(i, nodes)
        c = (b2_next - b2) / (2 * h)
        return c

    ans = [0 for _ in range(len(X))]

    for j in range(len(X)):
        i = 0
        while i < len(nodes) - 1 and nodes[i] < X[j]:
            i += 1
        y = f(nodes[i - 1])
        b = bi(i)
        c = ci(i)
        dif = (X[j - 1] - nodes[i - 1])
        ans[j] = y + b * dif + c * dif ** 2
    return ans


def drawFunction():
    plot.plot(X, f(X), label="Funckja")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


def drawCubicBothNodes(X, paraller, cheby, n, type=0):
    fig, axs = plot.subplots(1, 2)

    if type == 0:
        mode = "naturalnym"
    else:
        mode = "z zaciśniętymi granicami"

    fig.suptitle("Interpolacja sześcienna  na " + str(n) + " węzłach z warunkiem " + mode)
    axs[0].plot(X, f(X), label="Funkcja")
    axs[0].plot(X, cubic_spline(X, paraller, n, type), label="Interpolacja")
    axs[0].scatter(paraller, f(paraller), color="red", label="Punkty wspólne")
    axs[0].set_title("Interpolacja na węzłach równoległych")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    axs[1].plot(X, f(X), label="Funkcja")
    axs[1].plot(X, cubic_spline(X, cheby, n, type), label="Interpolacja")
    axs[1].scatter(cheby, f(cheby), color="red", label="Punkty wspólne")
    axs[1].set_title("Interpolacja na węzłach Czebyszewa")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].legend()

    fig.set_size_inches(14, 7)
    plot.show()


def drawCubic(X, nodes, n, type=0):
    if type == 0:
        mode = "naturalnym"
    else:
        mode = "z zaciśniętymi granicami"

    if nodes[0] == 0:
        nodes_mode = "równoległych"
    else:
        nodes_mode = "Czebyszewa"
    plot.suptitle("Interpolacja sześcienna na " + str(n) + "na węzłach" + nodes_mode + " z warunkiem granicznym" + mode)
    plot.plot(X, f(X), label="Funckja")
    plot.plot(X, cubic_spline(X, nodes, n, type), label="Interpolacja")
    plot.scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


def drawQuad(X, nodes, n, type=0):
    if type == 0:
        mode = "naturalnym"
    else:
        mode = "z zaciśniętymi granicami"

    if nodes[0] == 0:
        nodes_mode = "równoległych"
    else:
        nodes_mode = "Czebyszewa"
    plot.suptitle("Interpolacja kwadratowa na " + str(n) + "na węzłach" + nodes_mode + " z warunkiem granicznym" + mode)
    plot.plot(X, f(X), label="Funckja")
    plot.plot(X, cubic_spline(X, nodes, n, type), label="Interpolacja")
    plot.scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


def drawQuadBothNodes(X, paraller, cheby, n, type=0):
    fig, axs = plot.subplots(1, 2)

    if type == 0:
        mode = "naturalnym"
    else:
        mode = "z zaciśniętymi granicami"

    fig.suptitle("Interpolacja kwadratowa na " + str(n) + " węzłach z warunkiem " + mode)
    axs[0].plot(X, f(X), label="Funkcja")
    axs[0].plot(X, quadratic_spline(X, paraller, n, type), label="Interpolacja")
    axs[0].scatter(paraller, f(paraller), color="red", label="Punkty wspólne")
    axs[0].set_title("Interpolacja na węzłach równoległych")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    axs[1].plot(X, f(X), label="Funkcja")
    axs[1].plot(X, quadratic_spline(X, cheby, n, type), label="Interpolacja")
    axs[1].scatter(cheby, f(cheby), color="red", label="Punkty wspólne")
    axs[1].set_title("Interpolacja na węzłach Czebyszewa")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].legend()

    fig.set_size_inches(14, 7)
    plot.show()


n = 500
cheby_nodes = chebyshev_disrtibution(n, min_x, max_x)
parallel_nodess = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
X = np.arange(min_x, max_x + 0.01, 0.01)

# cubic_spline(X, cheby_nodes, n, 1)
# quadratic_spline(X, cheby_nodes, n, 1)

# drawCubicBothNodes(X, parallel_nodess, cheby_nodes, n, 0)
# drawCubicBothNodes(X, parallel_nodess, cheby_nodes, n, 1)
#
# drawQuadBothNodes(X, parallel_nodess, cheby_nodes, n, 0)
# drawQuadBothNodes(X, parallel_nodess, cheby_nodes, n, 1)

drawCubic(X, parallel_nodess, n, 0)
drawQuad(X, parallel_nodess, n, 1)
