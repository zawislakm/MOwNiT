import numpy as np
import pandas as pd
from math import pi, cos, sin, e
import matplotlib.pyplot as plot
import csv
import math

# mine
k = 1
m = 2

min_x = 0
max_x = 3 * pi


def f(x):
    if not isinstance(x, float):
        return [sin(m * i) * sin(k * i ** 2 / pi) for i in x]
    return sin(m * x) * sin(k * x ** 2 / pi)




def aprox(nodes: list, n: int, X: list, m: int):
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


def drawFunction():
    plot.plot(X, f(X), label="Zadana funckja")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


def drawAprox(nodes: list, X: list, m: int) -> None:
    n = len(nodes)
    plot.suptitle("Aproksymacja stopnia " + str(m) + " na " + str(n) + " węzłach równoległych")
    plot.plot(X, f(X), label="Funckja")
    plot.plot(X, aprox(nodes, n, X, m), label="Aproksymacja")
    plot.scatter(nodes, f(nodes), color="red", label="Węzły")
    plot.xlabel("X")
    plot.ylabel("Y")
    plot.legend()
    plot.show()


def drawAproxAll(nodes: list, X: list):
    ems = [2, 3, 5, 9, 15, 20]
    n = len(nodes)
    fig, axs = plot.subplots(2, 3)
    fig.suptitle("Aproksymacja na " + str(n) + " węzłach równoległych")
    for i in range(len(ems)):
        row = i // 3
        col = i % 3

        axs[row][col].plot(X, f(X), label="Funckja")
        axs[row][col].plot(X, aprox(nodes, n, X, ems[i]), label="Aproksymacja")
        axs[row][col].scatter(nodes, f(nodes), color="red", label="Węzły")
        axs[row][col].set_xlabel("X")
        axs[row][col].set_ylabel("Y")
        axs[row][col].set_title("Stopień wielomianu " + str(ems[i]))
        axs[row][col].legend()

    tableAproxGivenN(X, n, ems)
    tableAproxGivenN(X, n, ems)
    fig.set_size_inches(15, 10)
    plot.show()


def drawAproxAllM(X: list, m: int):
    ens = [50, 60, 80, 100, 200, 300]

    fig, axs = plot.subplots(2, 3)
    fig.suptitle("Aproksymacja stopnia  " + str(m))
    for i in range(len(ens)):
        if m > (ens[i]) / 2:
            continue
        nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (ens[i] - 1))
        print(nodes[-1])
        row = i // 3
        col = i % 3
        print(i,row,col)
        axs[row][col].plot(X, f(X), label="Funckja")
        axs[row][col].plot(X, aprox(nodes, ens[i], X, m), label="Aproksymacja n=" + str(ens[i]))
        axs[row][col].scatter(nodes, f(nodes), color="red", label="Węzły")
        axs[row][col].set_xlabel("X")
        axs[row][col].set_ylabel("Y")
        # axs[row][col].set_title("Liczba węzłów " + str(ens[i]))
        axs[row][col].legend()

    tableAproxGivenM(X, ens, m)


    fig.set_size_inches(15, 10)
    fig.tight_layout()
    plot.show()


def drawAproxBetween(X: list):
    n1 = 20
    n2 = 30
    ems = [9,9]
    fig, axs = plot.subplots(2, 2)
    # fig.suptitle("Aproksymacje 9 stopnia")
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

        n1 = 40
        n2 = 60
        n1nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n1 - 1))
        n2nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n2 - 1))

    fig.set_size_inches(15, 10)
    fig.tight_layout()
    plot.show()


def tableAproxGivenN(X, n, ems):
    outcome = []

    parallel_nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
    for em in ems:
        if em > n:
            break
        ans = aprox(parallel_nodes, n, X, em)
        outcome.append(
            [em, round(diffrecne(X, ans, n), 5), round(sqrdiffrence(X, ans, n), 5)])

    df = pd.DataFrame(outcome,
                      columns=["m", "Natural max error", "Natural square error"])

    filename = 'results' + str(n) + '.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["m", "Natural max error", "Natural square error"])
        for row in outcome:
            writer.writerow(row)
    return df


def tableAproxGivenM(X, ens, m):
    outcome = []

    for en in ens:
        if m > (en - 1) / 2:
            continue
        noddes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (en - 1))
        ans = aprox(noddes, en, X, m)
        outcome.append(
            [en, round(diffrecne(X, ans, en), 5), round(sqrdiffrence(X, ans, en), 5)])

    df = pd.DataFrame(outcome,
                      columns=["m", "Natural max error", "Natural square error"])

    filename = 'resultsM' + str(m) + '.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["n", "Natural max error", "Natural square error"])
        for row in outcome:
            writer.writerow(row)
    return df


def tableAprox(X):
    ens = [10, 20, 30, 40, 60, 80, 100, 200]
    ems = [2, 3, 5, 9, 15, 20]

    outcome = []
    dif_outcome = [["-" for _ in range(len(ems) + 1)] for _ in range(len(ens) + 1)]
    sqdif_outcome = [["-" for _ in range(len(ems) + 1)] for _ in range(len(ens) + 1)]

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
            if em > (n - 1) / 2:
                continue
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


n = 200
nm = 20
X = np.arange(min_x, max_x + 0.01, 0.01)
nodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (n - 1))
print(nodes[0], nodes[-1])
print(max_x)

# drawAproxAll(nodes,X)
# drawAproxAll(nodes, X)
# drawAproxAllM(X, nm)
# drawAprox(nodes, X, nm)
# drawAproxBetween(X)
# drawFunction()
# drawAproxAllM(X,nm)
# drawAproxAll(nodes,X)
# drawAproxMati(nodes, X, nm)
# drawAproxBot(nodes, X, nm)
# aproxtry(nodes, n, X, m)
# drawAprox(nodes, X, m)
# drawAproxBetween(X)
# drawAproxAll(nodes, X)
# tableAprox(X)
# tableAprox(X)
# tableAprox(X)

# drawFunction()
# drawAproxBetween(X)

# ens = [10, 20, 30, 40, 60, 80, 100, 200]
#
# for en in ens:
#     ennodes = np.arange(min_x, max_x + 0.01, (max_x - min_x) / (en - 1))
#     drawAproxAll(ennodes,X)


# tableAprox(X)
