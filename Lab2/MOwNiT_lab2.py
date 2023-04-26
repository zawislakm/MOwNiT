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


def chebyshev_disrtibution(n, a, b):  # b > a
    # https://pl.wikipedia.org/wiki/W%C4%99z%C5%82y_Czebyszewa
    cheby_points = []
    for k in range(1, n + 1):
        xk = 0.5 * (b + a) + 0.5 * (b - a) * cos(((2 * k - 1) * np.pi) / (2 * n))
        cheby_points.append(xk)
    return cheby_points[::-1]


def drawLagrange(nodes, n):
    plot.plot(X, f(X), label="Funkcja")
    plot.plot(X, Lagrange(X, nodes, n), label="Interpolacja")
    plot.suptitle("Interpolacja Lagrange'a na  " + str(n) + " węzłach")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    plot.legend()
    plot.show()

    return Lagrange(X, nodes, n)


def drawFunction():  # draw given function
    plot.plot(X, f(X), label="Funckja")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


def drawNewton(nodes, n):
    plot.plot(X, f(X), label="Funkcja")
    plot.plot(X, Newton(X, nodes, n), label="Interpolacja")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.suptitle("Interpolacja Newtona na " + str(n) + " węzłach")
    plot.scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    plot.legend()
    plot.show()

    return Newton(X, nodes, n)


def drawNewton2(parallel_nodes, cheby_nodes, n):
    fig, axs = plot.subplots(1, 2)
    fig.suptitle("Interpolacaja Newtona na obu rodzajach węzłów z " + str(n) + " węzłami")
    axs[0].plot(X, f(X), label="Funkcja")
    axs[0].plot(X, Newton(X, parallel_nodes, n), label="Interpolacja")
    axs[0].scatter(parallel_nodes, f(parallel_nodes), color="red", label="Punkty wspólne")
    axs[0].set_title("Interpolacja Newtona na węzłach równoległych")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    axs[1].plot(X, f(X), label="Funkcja")
    axs[1].plot(X, Newton(X, cheby_nodes, n), label="Interpolacja")
    axs[1].scatter(cheby_nodes, f(cheby_nodes), color="red", label="Punkty wspólne")
    axs[1].set_title("Interpolacja Newtona na węzłach Czebyszewa")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].legend()

    fig.set_size_inches(14, 7)
    plot.show()


def drawLagrange2(parallel_nodes, cheby_nodes, n):
    fig, axs = plot.subplots(1, 2)
    fig.suptitle("Interpolacaja Lagrange'a na obu rodzajach węzłów z " + str(n) + " węzłami")
    axs[0].plot(X, f(X), label="Funkcja")
    axs[0].plot(X, Lagrange(X, parallel_nodes, n), label="Interpolacja")
    axs[0].scatter(parallel_nodes, f(parallel_nodes), color="red", label="Punkty wspólne")
    axs[0].set_title("Interpolacja Lagrange'a na węzłach równoległych")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    axs[1].plot(X, f(X), label="Funkcja")
    axs[1].plot(X, Lagrange(X, cheby_nodes, n), label="Interpolacja")
    axs[1].scatter(cheby_nodes, f(cheby_nodes), color="red", label="Punkty wspólne")
    axs[1].set_title("Interpolacja Lagrange'a na węzłach Czebyszewa")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].legend()

    fig.set_size_inches(14, 7)
    plot.show()


def drawTogetherLagrangeNewton(nodes, n, type):
    fig, axs = plot.subplots(1, 2)
    fig.suptitle("Interpolacje na " + str(n) +" " +type)
    axs[0].plot(X, f(X), label="F. interpolowana")
    axs[0].plot(X, Lagrange(X, nodes, n), label="F. interpolująca")
    axs[0].scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    axs[0].set_title("Interpolacja Lagrange'a")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    axs[1].plot(X, f(X), label="F. interpolowana")
    axs[1].plot(X, Newton(X, nodes, n), label="F. interpolująca")
    axs[1].scatter(nodes, f(nodes), color="red", label="Punkty wspólne")
    axs[1].set_title("Interpolacja Newtona")
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


n = 12 # number of nodes
X = np.arange(min_x, max_x + 0.01, 0.01)
parallel_nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
cheby_nodes = chebyshev_disrtibution(n, min_x, max_x)
type_p = "węzłach równoległych"
type_c = "węzłach Czebyszewa"

# drawFunction()
# drawTogetherLagrangeNewton(parallel_nodes, n, type_p)
# drawTogetherLagrangeNewton(cheby_nodes, n, type_c)
#drawNewton2(parallel_nodes, cheby_nodes, n)
# drawLagrange2(parallel_nodes, cheby_nodes, n)
# drawTogetherLagrangeNewton(parallel_nodes, n, type_p)
# drawTogetherLagrangeNewton(parallel_nodes, n, type_p)
# drawLagrange2(parallel_nodes, cheby_nodes, n)
# drawNewton2(parallel_nodes, cheby_nodes, n)

drawTogetherLagrangeNewton(parallel_nodes,n,type_p)
def tableLagrange(ens, X):
    outcome = []
    for n in ens:
        parallel_nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
        cheby_nodes = chebyshev_disrtibution(n, min_x, max_x)
        LP = Lagrange(X, parallel_nodes, n)
        LC = Lagrange(X, cheby_nodes, n)
        outcome.append([n, diffrecne(X, LP, n), sqrdiffrence(X, LP, n), diffrecne(X, LC, n), sqrdiffrence(X, LC, n)])

    df = pd.DataFrame(outcome,
                      columns=["n", "Max error Parallel Nodes", "Sum square error Parallel Nodes",
                               "Max error Chebyshev Nodes", "Sum square error Chebyshev Nodes"])
    print(df)
    return df


def tableNewton(ens, X):
    outcome = []
    for n in ens:
        parallel_nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
        cheby_nodes = chebyshev_disrtibution(n, min_x, max_x)
        NP = Newton(X, parallel_nodes, n)
        NC = Newton(X, cheby_nodes, n)
        outcome.append([n, diffrecne(X, NP, n), sqrdiffrence(X, NP, n), diffrecne(X, NC, n), sqrdiffrence(X, NC, n)])

    df = pd.DataFrame(outcome,
                      columns=["n", "Max error Parallel Nodes", "Sum square error Parallel Nodes",
                               "Max error Chebyshev Nodes", "Sum square error Chebyshev Nodes"])
    print(df)
    return df


ens = [3, 4, 5, 7, 9, 10, 11, 12, 15, 20, 30, 40, 50, 60, 75]
# tableNewton(ens, X)

# tableLagrange(ens, X)


