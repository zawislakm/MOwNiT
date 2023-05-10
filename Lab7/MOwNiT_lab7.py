import math

a = -0.4
b = 1
n = 10
m = 17


def f(x):
    return m * x * math.pow(math.e, -n) - m * math.pow(math.e, -n * x) + 1 / m


def df(x):
    return m*n*math.pow(math.e,-n*x) + m*math.pow(math.e,-n)


epsilon = 10e-24
max_iteration = 100000


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
