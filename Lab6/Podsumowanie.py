import numpy as np
import pandas as pd
from math import pi, cos, sin
import matplotlib.pyplot as plot

k = 1
m = 2

min_x = 0
max_x = 3 * pi

X = np.arange(min_x, max_x + 0.01, 0.01)


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


def L(j, x, n, nodes):
    p = 1
    for i in range(0, n):
        if i != j:
            tmp = (x - nodes[i]) / (nodes[j] - nodes[i])
            p *= tmp
    return p


def Lagrange(x, nodes, n):
    ans = 0
    for j in range(0, n, 1):
        ans += (f(nodes[j]) * L(j, x, n, nodes))
    return ans


def a(k, nodes):
    suma = 0
    for i in range(0, k + 1):
        p = 1
        for j in range(0, k + 1):
            if j != i:
                p *= (nodes[i] - nodes[j])
        suma += (f(nodes[i]) / p)
    return suma


def Newton(x, nodes, n):
    ans = 0
    for k in range(0, n):
        p = a(k, nodes)
        for i in range(0, k):
            p *= (x - nodes[i])
        ans += p
    return ans


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
    for j in range(1, len(X)):
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

    for j in range(1, len(X)):
        i = 0
        while i < len(nodes) - 1 and nodes[i] < X[j]:
            i += 1
        y = f(nodes[i - 1])
        b = bi(i)
        c = ci(i)
        dif = (X[j - 1] - nodes[i - 1])
        ans[j] = y + b * dif + c * dif ** 2
    return ans


def drawInterpolations(n, nodes):
    fig, axs = plot.subplots(1, 3)
    if nodes[0] == 0:
        type_node = " węzłach równoległych"
    else:
        type_node = " węzłach Czebyszewa"
    fig.suptitle("Interpolacje na " + str(n) + type_node)

    axs[0].plot(X, f(X), label="F. interpolowana")
    axs[0].plot(X, Newton(X, nodes, n), label="F. interpolujaca")
    axs[0].scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    axs[0].set_title("Interpolacja Newtona")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    axs[1].plot(X, f(X), label="F. interpolowana")
    axs[1].plot(X, Lagrange(X, nodes, n), label="F. interpolujaca")
    axs[1].scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    axs[1].set_title("Interpolacja Lagrange'a")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].legend()

    axs[2].plot(X, f(X), label="F. interpolowana")
    axs[2].plot(X, Hermit(X, nodes, n, df), label="F. interpolujaca")
    axs[2].scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    axs[2].set_title("Interpolacja Hermit'a")
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")
    axs[2].legend()

    fig.set_size_inches(14, 7)
    plot.show()


def drawSplines(n, nodes):
    fig, axs = plot.subplots(2, 2)
    fig.suptitle("Interpolacje na " + str(n) + " węzłach równoległych")
    axs[0][0].plot(X, f(X), label="F. interpolowana")
    axs[1][0].plot(X, f(X), label="F. interpolowana")
    axs[0][1].plot(X, f(X), label="F. interpolowana")
    axs[1][1].plot(X, f(X), label="F. interpolowana")
    axs[0][0].set_xlabel("X")
    axs[1][0].set_xlabel("X")
    axs[0][1].set_xlabel("X")
    axs[1][1].set_xlabel("X")
    axs[0][0].set_ylabel("Y")
    axs[1][0].set_ylabel("Y")
    axs[0][1].set_ylabel("Y")
    axs[1][1].set_ylabel("Y")

    axs[0][0].scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    axs[1][0].scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    axs[0][1].scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    axs[1][1].scatter(nodes, f(nodes), color="red", label="Punkty wspólne")

    axs[0][0].legend()
    axs[1][0].legend()
    axs[0][1].legend()
    axs[1][1].legend()

    axs[0][0].plot(X, cubic_spline(X, nodes, n, 0), label="F. interpolująca")
    axs[1][0].plot(X, quadratic_spline(X, nodes, n, 0), label="F. interpolująca")
    axs[0][1].plot(X, cubic_spline(X, nodes, n, 1), label="F. interpolująca")
    axs[1][1].plot(X, quadratic_spline(X, nodes, n, 1), label="F. interpolująca")

    axs[0][0].set_title("Interpolacja sześcienna z warunkiem naturalnych granic")
    axs[1][0].set_title("Interpolacja sześcienna z warunkiem zaciśniętych granic")
    axs[0][1].set_title("Interpolacja kwadratowa z warunkiem naturalnych granic")
    axs[1][1].set_title("Interpolacja kwadratowa z warunkiem zaciśniętych granic")

    axs[0][0].legend()
    axs[1][0].legend()
    axs[0][1].legend()
    axs[1][1].legend()

    fig.set_size_inches(15, 11)

    plot.show()


def aproxNormal(nodes: list, n: int, X: list, m: int, w: list = None) -> list:
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


def aproxTry(nodes: list, n: int, X: list, m: int):
    def transform_x(x: int):
        return ((x - nodes[0]) / (nodes[-1] - nodes[0])) * (pi - (-pi)) + (-pi)

    def calc_ak(k: int):
        return 2 / n * sum(f(nodes[i]) * cos(k * transform_x(nodes[i])) for i in range(n))

    def calc_bk(k: int):
        return 2 / n * sum(f(nodes[i]) * sin(k * transform_x(nodes[i])) for i in range(n))

    ak = list(map(calc_ak, range(m + 1)))
    bk = list(map(calc_bk, range(m + 1)))

    def fa(x):
        x = transform_x(x)
        return .5 * ak[0] + sum(ak[k] * cos(k * x) + bk[k] * sin(k * x) for k in range(1, m + 1))

    ans = [0 for _ in range(len(X))]

    for j in range(len(X)):
        ans[j] = fa(X[j])

    return ans


def drawAprox(n, nodes, em):
    fig, axs = plot.subplots(1, 2)
    if nodes[0] == 0:
        type_node = " węzłach równoległych"
    else:
        type_node = " węzłach Czebyszewa"
    fig.suptitle("Arpokasymacja stopnia " + str(em) + " na " + str(n) + " węzłach równoległych")

    axs[0].plot(X, f(X), label="F. aproksymowana")
    axs[0].plot(X, aproxNormal(nodes, n, X, em - 1), label="F. aproksymująca")
    axs[0].scatter(nodes, f(nodes), color="red", label="Węzły")
    axs[0].set_title("A. wielomianami algebraicznymi")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    axs[1].plot(X, f(X), label="F. aproksymowana")
    axs[1].plot(X, aproxTry(nodes,n,X,em), label="F. aproksymująca")
    axs[1].scatter(nodes, f(nodes), color="red", label="Węzły")
    axs[1].set_title("A. trygonometryczna")
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



n = 10
em = 3

pnodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
cnodes = chebyshev_disrtibution(n, min_x, max_x)

# drawInterpolations(n, pnodes)
# drawSplines(n, pnodes)
drawAprox(n,pnodes,em)

# ans = aproxTry(pnodes,n,X,em)
#
# print(round(diffrecne(X,ans,n),5))
# print(round(sqrdiffrence(X,ans,n),5))
