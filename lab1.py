# вариант 9
# f(x) = 2*lnx - 1/x
# f'(x) = 2/x + 1/(x**2)
# f"(x) = -2/(x**2) - 2/(x**3)
import math
import numpy as np
import matplotlib.pyplot as plt


class DiffFunction:
    def __init__(self, *derivatives):
        self.derivatives = derivatives

    def __getitem__(self, item):
        return self.derivatives[item]

    def __call__(self, *args, **kwargs):
        return self.derivatives[0](*args, **kwargs)


def try_spread(arr_like):
    try:
        return [*iter(arr_like)]
    except:
        return [arr_like]


def iterative_solver(
        start,
        step_func,
        epsilon=(10 ** (-7)),
        halt_predicate=lambda pr, nx, e: abs(nx - pr) < e
):
    steps = [*try_spread(start)]
    steps.append(step_func(steps))
    while not halt_predicate(steps[-2], steps[-1], epsilon):
        steps.append(step_func(steps))
    return steps[-1], steps


def newton(f: DiffFunction, a, b, epsilon=10 ** (-7)):
    step_function = lambda x: x[-1] - (f(x[-1]) / f[1](x[-1]))
    if f[0](a) * f[2](a) > 0:
        return iterative_solver(a, step_function, epsilon)
    else:
        return iterative_solver(b, step_function, epsilon)


def chords(f: DiffFunction, a, b, epsilon=10 ** (-7)):
    if f(b) * f[2](b) <= 0:
        a, b = b, a

    def step_function(x):
        return x[-1] - f(x[-1]) * (b - x[-1]) / (f(b) - f(x[-1]))

    return iterative_solver(a, step_function, epsilon)


def secants(f: DiffFunction, a, b, epsilon=10 ** (-7)):
    start = [a, b]

    def step_function(x):
        return x[-1] - f(x[-1]) * (x[-1] - x[-2]) / (f(x[-1]) - f(x[-2]))

    return iterative_solver(start, step_function, epsilon)


def finite_sum_newton(f: DiffFunction, a, b, epsilon=10 ** (-7), h=0.01):
    h = abs(h)

    def step_function(x):
        return x[-1] - h * f(x[-1]) / (f(x[-1] + h) - f(x[-1]))

    return iterative_solver(min(a, b), step_function, epsilon)


function = DiffFunction(
    lambda x: 2 * math.log(x) - 1 / x,
    lambda x: 2 / x + 1 / (x ** 2),
    lambda x: -2 / (x ** 2) - 2 / (x ** 3)
)

print(finite_sum_newton(function, 0.1, 2))
