import math

import numpy as np
import time
from decimal import Decimal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

cols = ["Float32", "Float64"]


def gauss_float32(A, b, n):
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            L = A[i, k] / A[k, k]
            for j in range(k + 1, n):
                A[i, j] = A[i, j] - L * A[k, j]
            b[i] = b[i] - L * b[k]

    x = b.copy()
    for i in range(n - 1, -1, -1):
        S = np.float32(0.0)
        for j in range(i + 1, n):
            S = S + A[i, j] * x[j]
        x[i] = np.float32((x[i] - S) / A[i, i])
    return x


def create_X_float32(n):
    return np.array([np.float32((-1) ** i) for i in range(n)])


def create_matrix_A_float32(n):
    A = np.array([[np.float32(0) for _ in range(n)] for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if i == 0:
                A[i, j] = np.float32(1)
            else:
                A[i, j] = np.float32(1 / (i + j + 1))
    return A


def create_vector_B_float32(A, n):
    X = create_X_float32(n)
    B = np.array([np.float32(0) for _ in range(n)])
    for i in range(n):
        for j in range(n):
            B[i] += np.float32(A[i, j] * X[j])
    return B


def find_diff_float32(newX, n):
    oldX = create_X_float32(n)
    diff = np.float32(0)
    for i in range(n):
        diff = np.float32(max(diff, abs(oldX[i] - newX[i])))
    return diff


def create_matrix_A2_float32(n):
    A = np.array([[np.float32(0) for _ in range(n)] for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if j >= i:
                A[i, j] = np.float32(2 * (i + 1) / (j + 1))
            else:
                A[i, j] = np.float32(A[j, i])
    return A


def gauss_float64(A, B, n):
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            L = A[i, k] / A[k, k]
            for j in range(k + 1, n):
                A[i, j] = A[i, j] - L * A[k, j]
            B[i] = B[i] - L * B[k]

    x = B.copy()
    for i in range(n - 1, -1, -1):
        S = np.float64(0.0)
        for j in range(i + 1, n):
            S = S + A[i, j] * x[j]
        x[i] = np.float64((x[i] - S) / A[i, i])
    return x


def create_X_float64(n):
    return np.array([np.float64((-1) ** i) for i in range(n)])


def create_matrix_A_float64(n):
    A = np.array([[np.float64(0) for _ in range(n)] for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if i == 0:
                A[i, j] = np.float64(1)
            else:
                A[i, j] = np.float64(1 / (i + j + 1))
    return A


def create_matrix_A2_float64(n):
    A = np.array([[np.float64(0) for _ in range(n)] for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if j >= i:
                A[i, j] = np.float64(2 * (i + 1) / (j + 1))
            else:
                A[i, j] = np.float64(A[j, i])
    return A


#
#
# def create_matrix_A3(n, k, m):
#     A = np.array([[float(0) for _ in range(n)] for _ in range(n)])
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 A[i, j] = k
#             elif j == i + 1:
#                 A[i, j] = 1 / (i + 1 + m)
#             elif j == i - 1:
#                 A[i, j] = k / (i + 1 + m + 1)
#             else:
#                 A[i, j] = 0
#     return A


def create_vector_B_float64(A, n):
    X = create_X_float64(n)
    B = np.array([np.float64(0) for _ in range(n)])
    for i in range(n):
        for j in range(n):
            B[i] += np.float64(A[i, j] * X[j])
    return B


def find_diff_float64(newX, n):
    oldX = create_X_float64(n)
    diff = np.float64(0)
    for i in range(n):
        diff = np.float64(max(diff, abs(oldX[i] - newX[i])))
    return diff


def zad1():

    ens = [3, 4, 5, 7, 9, 12, 15, 18, 20, 30]
    df = pd.DataFrame(columns=cols, index=ens)

    for i in range(len(ens)):
        n = ens[i]
        print(n, end=" & ")
        A = create_matrix_A_float64(n)
        B = create_vector_B_float64(A, n)
        X = gauss_float64(A, B, n)
        # print(f'{find_diff_float64(X, n)}', end=" & ")
        df.iloc[i, 1] = find_diff_float64(X, n)

        A = create_matrix_A_float32(n)
        B = create_vector_B_float32(A, n)
        X = gauss_float32(A, B, n)
        # print(f'{find_diff_float32(X, n)}', end=" \\\\ \\hline\n")
        df.iloc[i, 0] = find_diff_float32(X, n)

    df.to_csv("Zad1_all.csv")
    df.applymap(lambda cell: round(cell, 6)).to_csv("Zad1_round.csv")


def zad2():

    ens = [3, 4, 5, 7, 9, 12, 15, 18, 20, 30,50,100,200]
    df = pd.DataFrame(columns=cols, index=ens)

    for i in range(len(ens)):
        n = ens[i]

        A = create_matrix_A2_float64(n)
        B = create_vector_B_float64(A, n)
        X = gauss_float64(A, B, n)
        df.iloc[i,1] = find_diff_float64(X,n)
        # print(f'{find_diff_float64(X, n):.4e}', end=" & ")

        A = create_matrix_A2_float32(n)
        B = create_vector_B_float32(A, n)
        X = gauss_float32(A, B, n)
        # print(f'{find_diff_float32(X, n):.4e}', end=" \\\\ \\hline\n")
        df.iloc[i,0] = find_diff_float32(X,n)

    df.to_csv("Zad2_all.csv")
    df.applymap(lambda cell: round(cell, 6)).to_csv("Zad2_round.csv")
    df.applymap(lambda cell: str(cell) ).to_csv("Zad2_round.csv")

zad2()

# #testy floatow
# print(np.finfo(np.float32))
# print(np.finfo(np.float64))
# x = np.float32(math.pi)
# y = np.float64(math.pi)
# print(x)
# print(y)
