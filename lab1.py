# вариант 9
# f(x) = 2*lnx - 1/x
# f'(x) = 2/x + 1/(x**2)
# f"(x) = -2/(x**2) - 2/(x**3)
import math
import numpy as np
import matplotlib.pyplot as plt

from util import meta


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


@meta(name='Метод Ньютона')
def newton(f: DiffFunction, a, b, epsilon=10 ** (-7)):
    step_function = lambda x: x[-1] - (f(x[-1]) / f[1](x[-1]))
    if f[0](a) * f[2](a) > 0:
        return iterative_solver(a, step_function, epsilon)
    else:
        return iterative_solver(b, step_function, epsilon)


@meta(name='Метод Хорд')
def chords(f: DiffFunction, a, b, epsilon=10 ** (-7)):
    if f(b) * f[2](b) <= 0:
        a, b = b, a

    def step_function(x):
        return x[-1] - f(x[-1]) * (b - x[-1]) / (f(b) - f(x[-1]))

    return iterative_solver(a, step_function, epsilon)


@meta(name='Метод Секущих')
def secants(f: DiffFunction, a, b, epsilon=10 ** (-7)):
    start = [a, b]

    def step_function(x):
        return x[-1] - f(x[-1]) * (x[-1] - x[-2]) / (f(x[-1]) - f(x[-2]))

    return iterative_solver(start, step_function, epsilon)


@meta(name='Конечноразностный метод Ньютона')
def finite_sum_newton(f: DiffFunction, a, b, epsilon=10 ** (-7), h=0.01):
    h = abs(h)

    def step_function(x):
        return x[-1] - h * f(x[-1]) / (f(x[-1] + h) - f(x[-1]))

    return iterative_solver(a, step_function, epsilon)


@meta(name='Метод Стефенсена')
def stephensen(f: DiffFunction, a, b, epsilon=10 ** (-7)):
    def step_function(x):
        return x[-1] - ((f(x[-1])) ** 2) / (f(x[-1] + f(x[-1])) - f(x[-1]))

    return iterative_solver(b, step_function, epsilon)


@meta(name='Метод простых итераций')
def simple_iter(f: DiffFunction, a, b, epsilon=10 ** (-7), tau=None):
    c = (a + b) / 2
    if tau is None:
        tau = 1 / f[1](c)  # как метод одной касательной, лучше, когда [a, b] близко к корню ??

    def step_function(x):
        return x[-1] - tau * f(x[-1])

    return iterative_solver(a, step_function, epsilon)
