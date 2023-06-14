import copy
import random
import time
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import csv


def find_diff(newX, n):
    oldX = create_X(n)
    diff = 0
    for i in range(n):
        diff = max(diff, abs(oldX[i] - newX[i]))
    return diff


def create_X(n):
    return np.array([float(-1) ** i for i in range(n)])


def create_matrix_A(n, k=8, m=2):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = k
            elif j > i:
                A[i, j] = (-1) ** j * m / j
            elif j == i - 1:
                A[i, j] = m / i
            else:
                A[i, j] = 0
    return A


def create_vector_B(A, n):
    X = create_X(n)
    B = np.zeros(n)
    for i in range(n):
        B[i] = np.dot(A[i], X)
    return B


def f1(x, x_new, A, B):
    return max([abs(x[i] - x_new[i]) for i in range(len(x))])


def f2(x, x_new, A, B):
    return max(np.subtract(np.dot(A, x_new), B))


def jacoby(A):
    n = len(A[0])
    L = np.zeros((n, n))
    D = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = A[i, j]
            elif i == j:
                D[i, j] = A[i, j]
            elif i < j:
                U[i, j] = A[i, j]

    N = np.linalg.inv(D)
    M = -np.dot(N, (L + U))
    return N, M


def jacoby_iter(A1, M1, N1, x1, B1, criterion, eps, max_iters=10000):
    A = copy.deepcopy(A1)
    M = copy.deepcopy(M1)
    N = copy.deepcopy(N1)
    x = copy.deepcopy(x1)
    B = copy.deepcopy(B1)

    iterations = 1
    x_new = np.add(np.dot(M, x), np.dot(N, B))

    while criterion(x, x_new, A, B) > eps and iterations < max_iters:
        x_new, x = np.add(np.dot(M, x_new), np.dot(N, B)), x_new
        iterations += 1

    return iterations, x_new


def zad1(criterion, start_values=0):
    ens = [3, 4, 7, 10, 15, 20, 30, 50, 70, 100, 200, 500]
    ros = [10e-2, 10e-4, 10e-8, 10e-10]
    df = pd.DataFrame(columns=ros, index=ens)
    for i, n in enumerate(ens):
        X = np.array([float(start_values) for _ in range(n)])
        A = create_matrix_A(n)
        B = create_vector_B(A, n)
        N, M = jacoby(A)
        for j, p in enumerate(ros):
            time_start = time.perf_counter()
            iterations, x_new = jacoby_iter(A, M, N, X, B, criterion, p)
            time_end = time.perf_counter() - time_start
            dif = find_diff(x_new, n)
            df.iloc[i, j] = (iterations, "{:.4e}".format(dif), "{:.4e}".format(time_end))

    df.applymap(lambda cell: cell[0]).to_csv(f'z1start{start_values}iter.csv')
    df.applymap(lambda cell: cell[1]).to_csv(f'z1start{start_values}erro.csv')
    df.applymap(lambda cell: cell[2]).to_csv(f'z1start{start_values}time.csv', index=False)

zad1(f1,0)
zad1(f1,10)


def zad2():
    ens = [3, 4, 7, 10, 15, 20, 30, 50, 70, 100, 200, 500, 800, 1000, 1200, 1500]
    df = pd.DataFrame(columns=["Radius"], index=ens)
    for i, n in enumerate(ens):
        A = create_matrix_A(n)
        N, M = jacoby(A)
        radius = max(np.linalg.eigvals(M))
        radius = np.real(radius)
        df.iloc[i, 0] = f"{radius:.4e}"
    df.to_csv(f'z2.csv')


# zad2()


def file_reader(file_path) -> list:
    p1_list = []
    p2_list = []
    p3_list = []
    p4_list = []

    with open(file_path) as file:
        writer = csv.reader(file)
        next(writer)
        for p1, p2, p3, p4 in writer:
            p1_list.append(float(p1))
            p2_list.append(float(p2))
            p3_list.append(float(p3))
            p4_list.append(float(p4))

    return [p1_list, p2_list, p3_list, p4_list]


def times(plot_title):
    path_names = [("z1start0time.csv", "wektor [0,0,0,...]"), ("z1start10time.csv", "wektor [10,10,10,...]")]
    ros_names = ["1e-01", "1e-03", "1e-04", "1e-09"]
    ens = [3, 4, 7, 10, 15, 20, 30, 50, 70, 100, 200, 500]

    x = np.linspace(0, len(ens)-1, len(ens))  # Równomiernie rozmieszczone wartości na osi x

    for path_name, vector_type in path_names:
        output = file_reader(path_name)
        for i in range(4):
            plot.scatter(x, output[i], label=f"p = {ros_names[i]}, {vector_type}")

    plot.xlabel("n")
    plot.ylabel("Czas (s)")
    plot.xticks(x, ens)
    plot.yscale('log')
    plot.legend()
    plot.title(plot_title)
    plot.show()


# times("Porównanie czasu obliczeń dla kryterium 1")
