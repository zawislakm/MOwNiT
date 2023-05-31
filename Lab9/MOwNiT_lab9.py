import copy

import numpy
import numpy as np
import time
from decimal import Decimal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize


def find_diff(newX, n):
    oldX = create_X(n)
    diff = 0
    for i in range(n):
        diff = max(diff, abs(oldX[i] - newX[i]))
    return diff


def create_X(n):
    return np.array([float(-1) ** i for i in range(n)])


# numpy dot wykonuje mnozencie macierzowe dwoch tablic
# numpy subtrackt wykonuje odejmowanie dwoch jednowymiarowych tablic

def create_matrix_A(n, k=8, m=2):
    # zmienic tu
    A = np.array([[float(0) for _ in range(n)] for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = k
            if j > i:
                A[i, j] = (-1) ** j * m / j
            if j == i - 1:
                A[i, j] = m / i
            else:
                A[i, j] = 0
    return A


def create_vector_B(A, n):
    X = create_X(n)
    B = np.array([float(0) for _ in range(n)])
    for i in range(n):
        for j in range(n):
            B[i] += A[i, j] * X[j]
    return B


def f1(x, x_new):
    return max([abs(x[i] - x_new[i]) for i in range(len(x))])


def f2(x, A, B):
    return max(np.subtract(np.dot(A, x), B))


def jacoby(A):
    n = len(A[0])
    L = np.array([[float(0) for _ in range(n)] for _ in range(n)])
    D = np.array([[float(0) for _ in range(n)] for _ in range(n)])
    U = np.array([[float(0) for _ in range(n)] for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = A[i, j]
            if i == j:
                D[i, j] = A[i, j]
            if i < j:
                U[i, j] = A[i, j]

    N = numpy.linalg.inv(D)
    M = np.dot(-N, (L + U))
    return N, M


def jacoby_iter(A1, M1, N1, x1, B1, criterion, eps):
    A = copy.deepcopy(A1)
    M = copy.deepcopy(M1)
    N = copy.deepcopy(N1)
    x = copy.deepcopy(x1)
    B = copy.deepcopy(B1)

    iterations = 1
    x_new = np.add(np.dot(M, x), np.dot(N, B))
    if criterion == 0:
        while f1(x, x_new) > eps:
            # print(f1(x, x_new))
            x_new, x = np.add(np.dot(M, x_new), np.dot(N, B)), x_new
            iterations += 1
    elif criterion == 1:
        while f2(x_new, A, B) > eps:
            x_new, x = np.add(np.dot(M, x_new), np.dot(N, B)), x_new
            iterations += 1

    return iterations, x_new
