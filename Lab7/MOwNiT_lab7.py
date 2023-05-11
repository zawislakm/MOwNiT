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


def show_heatmap(df,name='', annot=True, norm=None, xlabel='Start x values', ylabel='p values', title='', **kwargs):
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

wolfram_solution = 0.56590291432518767530  # 30 znaków po przecinku


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
    # napisz funkcje która bedzie rysować mape ciepła dla dataframe
    show_heatmap(df.applymap(lambda cell: cell[2]))


# calculate_newton(stop_criterion1)


def next_step_secant(xi0, xi1):
    try:
        return xi1 - ((xi1 - xi0) / (f(xi1) - f(xi0))) * f(xi1)
    except ZeroDivisionError:
        # print("Error: Division by zero occurred")
        # Możesz dodać odpowiednie działanie w przypadku wystąpienia dzielenia przez zero
        # Na przykład, zwrócić None lub inny sygnalizator błędu.
        # W tym przykładzie wypisujemy komunikat o błędzie.
        return None


def secant_method(x0, x1, stop_criterion, p):
    xi0 = None
    xi1 = x0
    xi2 = x1
    iterations = 0

    while not stop_criterion(xi1, xi2, p) and iterations < max_iteration:
        xi2, xi1, xi0 = next_step_secant(xi2, xi1), xi2, xi1

        if xi2 is None:
            return xi1,iterations
        iterations += 1

    return xi2,iterations
    # return round(xi2, 8), iterations


def calculate_secant(stop_criterion, step=0.1):
    start_x_forward = [i / 10 for i in range(int(a * 10), int((b + step) * 10), int(step * 10))]
    start_x_backward = [i / 10 for i in range(int(b * 10), int((a - step) * 10), int(-step * 10))]

    print(f"Forward: {start_x_forward}")
    print(f"Backward: {start_x_backward}")

    df_forward = pd.DataFrame(columns=ros, index=start_x_forward)
    df_backward = pd.DataFrame(columns=ros, index=start_x_backward)

    for i in range(len(start_x_forward)):
        for j in range(len(ros)):
            for_value, for_inter = secant_method(start_x_forward[i], b, stop_criterion, ros[j])
            df_forward.iloc[i, j] = (for_value, for_inter, error_value(for_value))
            back_value, back_inter = secant_method(start_x_backward[i], a, stop_criterion, ros[j])
            df_backward.iloc[i, j] = (back_value, back_inter, error_value(back_value))
        # print("Ended:" ,start_x_forward[i],ros[i])

    df_forward.applymap(lambda cell: round(cell[0],6)).to_csv('secant_values_forward.csv')
    df_forward.applymap(lambda cell: cell[1]).to_csv('secant_interations_forward.csv')
    # df_forward.applymap(lambda cell: cell[2]).to_csv('secant_error_forward.csv')

    df_backward.applymap(lambda cell: round(cell[0],6)).to_csv('secant_values_backward.csv')
    df_backward.applymap(lambda cell: cell[1]).to_csv('secant_interations_backward.csv')
    # df_backward.applymap(lambda cell: cell[2]).to_csv('secant_error_backward.csv')

    show_heatmap(df_forward.applymap(lambda cell: cell[2]),"forward")
    show_heatmap(df_backward.applymap(lambda cell: cell[2]),"backward")


calculate_secant(stop_criterion1)


def newton_method(x, mode):
    x1 = x - f(x) / df(x)

    for i in range(1, max_iteration):

        if mode == 1:
            if abs(x - x1) < epsilon:
                return "moduł różnicy: {}".format(abs(x - x1)), i, x1
        else:
            if abs(f(x1)) < epsilon:
                return "moduł funkcji: {}".format(abs(f(x1))), i, x1
            x0, x1 = x1, x1 - f(x1) / df(x1)
    return -1, -1, x1


def secant_method(a, b, mode):
    x0, x1 = a, b

    for i in range(0, max_iteration):
        if mode == 1:
            if abs(x0 - x1) < epsilon:
                return "moduł różnicy: {}".format(abs(x0 - x1)), i, x1
        else:
            if abs(f(x1)) < epsilon:
                return "moduł funkcji: {}".format(abs(f(x1))), i, x1
        if f(x1) - f(x0) == 0:
            return "-", "-", "-"
        x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        # if(f(x1)*f(tmp)<=0):
        #     x0, x1 = x, x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
        # else:
        #     x0, x1 = x1, x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
    return -1, -1, x1
