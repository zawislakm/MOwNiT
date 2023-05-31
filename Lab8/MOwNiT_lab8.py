import math

import numpy as np
import time
from decimal import Decimal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
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
    ens = [i for i in range(2, 21)]
    # ens += [30, 50, 100, 150, 200, 300, 400, 500]

    df = pd.DataFrame(columns=["Float32", "Float64"], index=ens)

    for i in range(len(ens)):
        n = ens[i]
        A = create_matrix_A_float64(n)
        B = create_vector_B_float64(A, n)
        X = gauss_float64(A, B, n)
        df.iloc[i, 1] = "{:.4e}".format(find_diff_float64(X, n))  # Zapisuje w postaci naukowej

        A = create_matrix_A_float32(n)
        B = create_vector_B_float32(A, n)
        X = gauss_float32(A, B, n)
        df.iloc[i, 0] = "{:.4e}".format(find_diff_float32(X, n))  # Zapisuje w postaci naukowej

    df.to_csv("Zad1_all.csv")


# zad1()

def zad2():
    ens = [i for i in range(2, 21)]
    ens += [30, 50, 100, 150, 200, 300, 400, 500]
    print(ens)
    df = pd.DataFrame(columns=cols, index=ens)

    for i in range(len(ens)):
        n = ens[i]

        A = create_matrix_A2_float64(n)
        B = create_vector_B_float64(A, n)
        X = gauss_float64(A, B, n)
        df.iloc[i, 1] = "{:.4e}".format(find_diff_float64(X, n))

        A = create_matrix_A2_float32(n)
        B = create_vector_B_float32(A, n)
        X = gauss_float32(A, B, n)
        df.iloc[i, 0] = "{:.4e}".format(find_diff_float32(X, n))

    df.to_csv("Zad2_all.csv")


# zad2()


def norm(A):
    n = len(A)
    return max(sum(A[i, j] for j in range(n)) for i in range(n))


def conditioning_factor(A):
    A_inv = np.linalg.inv(A)
    return norm(A_inv) * norm(A)


def normalization():
    ens = [i for i in range(2, 21)]
    ens += [30, 50, 100, 150, 200, 300, 400, 500]

    df = pd.DataFrame(columns=["Zadanie 1", "Zadanie 2"], index=ens)

    for i in range(len(ens)):
        n = ens[i]
        A1 = create_matrix_A_float64(n)
        A2 = create_matrix_A2_float64(n)

        df.iloc[i, 0] = "{:.4e}".format(conditioning_factor(A1))
        df.iloc[i, 1] = "{:.4e}".format(conditioning_factor(A2))
    df.to_csv("Norm.csv")


# normalization()

def create_matrix_A3_float64(n):
    m = 5
    k = 6
    A = np.array([[np.float64(0) for _ in range(n)] for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = np.float64(k)
            elif j == i + 1:
                A[i, j] = np.float64(1 / (i + 1 + m))
            elif j == i - 1:
                A[i, j] = np.float64(k / (i + 1 + m + 1))
            else:
                A[i, j] = np.float64(0)
    return A


def calc_time(start=0):
    if start == 0:
        return time.time()
    else:
        return time.time() - start


def thomas(A, B, n):
    C = np.zeros(n)
    C[0] = A[0, 0]
    D = np.zeros(n)
    D[0] = B[0]

    for i in range(1, n):
        work = np.float64(A[i, i - 1] / C[i - 1])
        C[i] = np.float64(A[i, i] - work * A[i - 1, i])
        D[i] = np.float64(B[i] - work * D[i - 1])

    D[n - 1] = np.float64(D[n - 1] / C[n - 1])
    for i in range(n - 2, -1, -1):
        D[i] = np.float64((D[i] - A[i, i + 1] * D[i + 1]) / C[i])

    return D


import pandas as pd
import time


def zad3():
    ens = [i for i in range(2, 21)]
    ens += [30, 50, 100, 150, 200, 300, 400, 500, 600, 700, 900, 1000]
    print(ens)
    df = pd.DataFrame(columns=["Gauss-error", "Gauss-time", "Thomas-error", "Thomas-time"], index=ens)
    list_times_gauss = []
    list_times_thomas = []
    for i in range(len(ens)):
        n = ens[i]

        A = create_matrix_A3_float64(n)
        B = create_vector_B_float64(A, n)

        time_start = time.perf_counter()
        X_gauss = gauss_float64(A, B, n)
        time_gauss = time.perf_counter() - time_start
        df.iloc[i, 0] = "{:.4e}".format(find_diff_float64(X_gauss, n))
        df.iloc[i, 1] = "{:.4e}".format(time_gauss)
        list_times_gauss.append(time_gauss)

        time_start = time.perf_counter()
        X_thomas = thomas(A, B, n)
        time_thomas = time.perf_counter() - time_start
        dif = find_diff_float64(X_thomas, n)
        df.iloc[i, 2] = "{:.4e}".format(dif)
        df.iloc[i, 3] = "{:.4e}".format(time_thomas)
        list_times_thomas.append(time_thomas)

    df.to_csv("Zad3_all.csv")
    en_to_time(ens, list_times_gauss, list_times_thomas)


def en_to_time(ens, times_gauss, times_thomas):
    plot.scatter(ens, times_gauss, label="Gauss")
    plot.scatter(ens, times_thomas, label="Thomas")
    plot.xlabel("n")
    plot.ylabel("Czas (s)")
    plot.yscale('log')
    plot.title("Porównanie czasów")
    plot.legend()
    plot.show()


# zad3()

n = 4
A = create_matrix_A3_float64(n)
B = create_vector_B_float64(A,n)
X = thomas(A,B,n)
#
# print("Output:")
# X = thomas(A,B,n)
# print(X)
# X_diff = find_diff_float64(X,n)
# print(X_diff)


# #testy floatow
# print(np.finfo(np.float32))
# print(np.finfo(np.float64))
# x = np.float32(math.pi)
# y = np.float64(math.pi)
# print(x)
# print(y)


