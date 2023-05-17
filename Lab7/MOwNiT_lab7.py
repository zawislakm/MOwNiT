import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

a = -0.4
b = 1
n = 10
m = 17

sns.set(rc={"figure.dpi": 200, "savefig.dpi": 200})

import matplotlib.colors as colors


def show_heatmap(df, name='', annot=True, norm=None, xlabel='Start x values', ylabel='p values', title='', **kwargs):
    plt.figure(figsize=(15, 10))
    cmap = sns.color_palette("YlGnBu")
    s = sns.heatmap(df, cmap=cmap, annot=annot, norm=colors.LogNorm(), mask=df.isnull(), **kwargs)
    s.set_xlabel(xlabel, fontsize=16)
    s.set_ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=20, y=1.025)

    # Ustawienie koloru napisów na czarny
    for text in s.texts:
        text.set_color('black')

    plt.savefig(f'heat_map{name}.png')
    # plt.show()


def f(x):
    return m * x * math.pow(math.e, -n) - m * math.pow(math.e, -n * x) + 1 / m


def df(x):
    return m * n * math.pow(math.e, -n * x) + m * math.pow(math.e, -n)


def next_step_newton(x):
    return x - f(x) / df(x)


epsilon = 10e-24
max_iteration = 100000

ros = [10e-3, 10e-4, 10e-6, 10e-8, 10e-10, 10e-16]

wolfram_solution = 0.56590291432518767530


def error_value(x):
    return abs(wolfram_solution - x)
    # return abs(round(wolfram_solution - x,8))


def stop_criterion1(x_curr, x_prev, p) -> bool:
    return abs(x_curr - x_prev) < p


def stop_criterion2(x_curr, x_prev, p) -> bool:
    return abs(f(x_curr)) < p


def newton_raphson(x0, stop_criterion, p):
    x_prev = float('inf')
    x_curr = x0
    iterations = 0

    while not stop_criterion(x_curr, x_prev, p) and iterations < max_iteration:
        x_curr, x_prev = next_step_newton(x_curr), x_curr
        iterations += 1

    return x_curr, iterations
    # return round(x_curr, 8), iterations


def calculate_newton(stop_criterion, step=0.1):
    start_x = [i / 10 for i in range(int(a * 10), int((b + step) * 10), int(step * 10))]
    print(start_x)

    df = pd.DataFrame(columns=ros, index=start_x)

    for i in range(len(start_x)):
        for j in range(len(ros)):
            value, iter = newton_raphson(start_x[i], stop_criterion, ros[j])
            df.iloc[i, j] = (value, iter, error_value(value))

    df.applymap(lambda cell: round(cell[0], 6)).to_csv('netwon_values.csv')
    df.applymap(lambda cell: cell[1]).to_csv('newton_interations.csv')
    # df.applymap(lambda cell: cell[2]).to_csv('newton_error.csv')
    show_heatmap(df.applymap(lambda cell: cell[2]))


# calculate_newton(stop_criterion1)


def next_step_secant(xi0, xi1, start):
    try:
        return xi1 - ((xi1 - xi0) / (f(xi1) - f(xi0))) * f(xi1)
    except ZeroDivisionError:
        if start == 1:
            print(xi0, xi1)
        return None


def secant_method(x0, x1, stop_criterion, p):
    xi0 = None
    xi1 = x0
    xi2 = x1
    iterations = 0
    start = x0

    while not stop_criterion(xi1, xi2, p) and iterations < max_iteration:
        xi2, xi1, xi0 = next_step_secant(xi2, xi1, start), xi2, xi1

        if xi2 is None:
            return xi1, iterations
        iterations += 1

    return xi2, iterations
    # return round(xi2, 8), iterations


def calculate_secant(stop_criterion, step=0.1):
    start_x_forward = [i / 10 for i in range(int(a * 10), int((b) * 10), int(step * 10))]
    start_x_backward = [i / 10 for i in range(int(b * 10), int((a) * 10), int(-step * 10))]

    print(f"Forward: {start_x_forward}")
    print(f"Backward: {start_x_backward}")

    df_forward = pd.DataFrame(columns=ros, index=start_x_forward)
    df_backward = pd.DataFrame(columns=ros, index=start_x_backward)

    for i in range(len(start_x_forward)):
        for j in range(len(ros)):
            for_value, for_inter = secant_method(start_x_forward[i], b, stop_criterion, ros[j])
            df_forward.iloc[i, j] = (for_value, for_inter, error_value(for_value))

            # backward
            back_value, back_inter = secant_method(start_x_backward[i], a, stop_criterion, ros[j])
            df_backward.iloc[i, j] = (back_value, back_inter, error_value(back_value))
        # print("Ended:" ,start_x_forward[i],ros[i])

    df_forward.applymap(lambda cell: round(cell[0], 6)).to_csv('secant_values_forward.csv')
    df_forward.applymap(lambda cell: cell[1]).to_csv('secant_interations_forward.csv')
    # df_forward.applymap(lambda cell: cell[2]).to_csv('secant_error_forward.csv')

    # bacwakrd
    df_backward.applymap(lambda cell: round(cell[0], 6)).to_csv('secant_values_backward.csv')
    df_backward.applymap(lambda cell: cell[1]).to_csv('secant_interations_backward.csv')
    # df_backward.applymap(lambda cell: cell[2]).to_csv('secant_error_backward.csv')

    # heatmap
    show_heatmap(df_forward.applymap(lambda cell: cell[2]), "forward")
    show_heatmap(df_backward.applymap(lambda cell: cell[2]), "backward")


# calculate_secant(stop_criterion2)

# exercise 2

def f1(X):
    x1, x2, x3 = X[0], X[1], X[2]
    return x1 ** 2 - 4 * x2 ** 2 + x3 ** 3 - 1


def f2(X):
    x1, x2, x3 = X[0], X[1], X[2]
    return 2 * x1 ** 2 + 4 * x2 ** 2 - 3 * x3


def f3(X):
    x1, x2, x3 = X[0], X[1], X[2]
    return x1 ** 2 - 2 * x2 + x3 ** 2 - 1


def F(X):
    return [f1(X), f2(X), f3(X)]


def J1(X):
    x1, x2, x3 = X[0], X[1], X[2]
    return [2 * x1, - 8 * x2, 3 * x3 ** 2]


def J2(X):
    x1, x2, x3 = X[0], X[1], X[2]
    return [4 * x1, 8 * x2, - 3]


def J3(X):
    x1, x2, x3 = X[0], X[1], X[2]
    return [2 * x1, -2, 2 * x3]


def J(X):
    return [J1(X), J2(X), J3(X)]


def ex2_stop_criterion1(x_curr, x_prev, p) -> bool:
    n = len(x_curr)
    for i in range(n):
        if abs(x_curr[i] - x_prev[i]) >= p:
            return False

    return True


def ex2_stop_criterion2(x_curr, x_prev, p) -> bool:
    if f1(x_curr) >= p:
        return False
    if f2(x_curr) >= p:
        return False
    if f3(x_curr) >= p:
        return False
    return True


def ex2_step(X):
    S = np.linalg.solve(J(X), F(X))
    return X - S


def newton_matrix(X, stop_critertion, p, max_intrations=1000):
    x_prev = [float('inf') for _ in range(len(X))]
    x_curr = X
    iters = 0

    while not stop_critertion(x_curr, x_prev, p) and iters < max_intrations:
        try:
            x_curr, x_prev = ex2_step(x_curr), x_curr
        except np.linalg.LinAlgError:
            x_curr = None
            break
        iters += 1
    return x_curr, iters


x1wolfram = [-1, 1, -0.917716, 0.917716]
x2wolfram = [0.5, 0.5, 0.084058, 0.084058]
x3wolfram = [1, 1, 0.570889, 0.570889]


def get_ans() -> list:
    t = []
    for i in range(4):
        # t.append([x1wolfram[i],x2wolfram[i],x2wolfram[i]])
        t.append(x1wolfram[i])
    print(t)
    return t


def solution_match(ans, p=1e-7) -> int:
    # print(ans)
    x1, x2, x3 = round(ans[0], 6), round(ans[1], 6), round(ans[2], 6)
    for i in range(4):
        if abs(x1 - x1wolfram[i]) < p and abs(x2 - x2wolfram[i]) < p and abs(x3 - x3wolfram[i]) < p:
            return i

    return -1


def newton_all(stop_criterion):
    array = [round(i * 0.1, 2) for i in range(-5, 6)]
    print(array)
    ans = get_ans()
    count = 0
    df = pd.DataFrame(0, columns=ros, index=ans)
    for a in array:
        for b in array:
            for c in array:
                X = [a, b, c]

                for i in range(len(ros)):
                    p = ros[i]

                    ans, iter = newton_matrix(X, stop_criterion, p)

                    if ans is not None:
                        sol = solution_match(ans)
                        if sol != -1:
                            df.iloc[sol, i] += 1

    # df.to_csv('newtonEX2.csv')


# newton_all(ex2_stop_criterion1)
