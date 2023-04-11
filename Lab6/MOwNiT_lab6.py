import numpy as np
import pandas as pd
from math import pi, cos, sin
import matplotlib.pyplot as plot

# mine
k = 1
m = 2

min_x = 0
max_x = 3 * pi


def f(x):
    if not isinstance(x, float):
        return [sin(m * i) * sin(k * i ** 2 / pi) for i in x]
    return sin(m * x) * sin(k * x ** 2 / pi)


def aproxtryLol(nodes, values, degree):
    # global a0
    # degree = len(nodes) // 2
    nodes_n = len(nodes)
    # nodes_n2 = len(nodes)//2
    a0 = 0
    for i in values:
        a0 += i
    a0 /= nodes_n
    A = []
    for j in range(degree):
        tmp = 0
        for i in range(nodes_n):
            # tmp += values[i] * cos(2 * pi * i * j / nodes_n)
            tmp += values[i] * cos((j + 1) * nodes[i])
        tmp *= (2 / nodes_n)
        A.append(tmp)
    B = []
    for j in range(degree):
        tmp = 0
        for i in range(nodes_n):
            # tmp += values[i] * sin(2 * pi * i * j / nodes_n)
            tmp += values[i] * sin((j + 1) * nodes[i])
        tmp *= (2 / nodes_n)
        B.append(tmp)
    ANS = []
    for s in X:
        tmp = a0
        for i in range(degree):
            tmp += A[i] * cos((i + 1) * s)  # (s-min_x)*pi*(2-1/nodes_n)/(max_x-min_x)-pi
            # tmp += A[i] * cos((i + 1) * ((s-min_x)*pi*(2-2/nodes_n)/(max_x-min_x)-pi)) #(s-min_x)*pi*(2-1/nodes_n)/(max_x-min_x)-pi
            # tmp += A[i] * cos(s)**(i + 1)
        for i in range(degree):
            tmp += B[i] * sin((i + 1) * s)
            # tmp += B[i] * sin((i + 1) * (s-min_x)*pi*(2-2/nodes_n)/(max_x-min_x)-pi)
            # tmp += B[i] * sin(s)**(i + 1)
        ANS.append(tmp)
    return ANS


def aproxtry(nodes: list, n: int, X: list, m: int, ) -> list:
    A = [0 for _ in range(m + 1)]
    for k in range(m + 1):
        for i in range(n):
            A[k] += f(nodes[i]) * cos((k + 1) * nodes[i])
        A[k] *= (2 / n)

    B = [0 for _ in range(m + 1)]
    for k in range(m + 1):
        for i in range(n):
            B[k] += f(nodes[i]) * sin((k + 1) * nodes[i])
        B[k] *= (2 / n)

    def transformEX(x) -> int:
        x_prim = ((x - min_x) / (max_x - min_x)) * (pi - (-1*pi)) + (-1*pi)
        return x_prim

    ans = [0 for _ in range(len(X))]
    for j in range(len(X)):
        x_p = transformEX(X[j])
        ans[j] = A[0] / 2
        for k in range(1, m):
            ans[j] += A[k] * cos(k * x_p) + B[k] * sin(k * x_p)

    return ans


def drawFunction():
    plot.plot(X, f(X), label="Funkcja interpolowana")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()
def drawAprox(nodes: list, X: list, m: int) -> None:
    n = len(nodes)
    plot.suptitle("Aproksymacja stopnia " + str(m) + " na " + str(n) + " węzłach równoległych")
    plot.plot(X, f(X), label="Funckja")
    plot.plot(X, aproxtry(nodes, n, X, m), label="Aproksymacja")
    plot.plot(X, aproxtryLol(nodes, f(nodes), m),label ="LOlx")
    plot.scatter(nodes, f(nodes), color="red", label="Węzły")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


n = 3

X = np.arange(min_x, max_x + 0.01, 0.01)
# nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))

#aproxtry(nodes, n, X, m)
# drawAprox(nodes, X, m)
drawFunction()
