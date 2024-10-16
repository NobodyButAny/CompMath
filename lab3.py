# Вариант 9
# x*ln(x) e=10**(-4) [2, 6]
# Примем за точное значение 22.8653
import numpy as np

from util import meta


@meta(name="Метод левых прямоугольников")
def rectangular_left(f, a, b, e=10 ** (-4), steps_start=10):
    span = b - a
    N = steps_start
    convergence = False

    h_n = span / N
    res_n = h_n * np.sum([
        f(a + i * h_n)
        for i in range(N)
    ])
    while not convergence:
        h_2n = span / (2 * N)
        res_2n = h_2n * np.sum([
            f(a + i * h_2n)
            for i in range(2 * N)
        ])
        convergence = abs(res_2n - res_n) <= e
        N *= 2
        res_n = res_2n

    return res_2n, h_2n, N * 2


@meta(name="Метод правых прямоугольников")
def rectangular_right(f, a, b, e=10 ** (-4), steps_start=10):
    span = b - a
    N = steps_start
    convergence = False

    h_n = span / N
    res_n = h_n * np.sum([
        f(a + i * h_n)
        for i in range(1, N + 1)
    ])
    while not convergence:
        h_2n = span / (2 * N)
        res_2n = h_2n * np.sum([
            f(a + i * h_2n)
            for i in range(1, 2 * N + 1)
        ])
        convergence = abs(res_2n - res_n) <= e
        N *= 2
        res_n = res_2n

    return res_2n, h_2n, N * 2


@meta(name="Метод средних прямоугольников")
def rectangular_middle(f, a, b, e=10 ** (-4), steps_start=10):
    span = b - a
    N = steps_start
    convergence = False

    h_n = span / N
    res_n = h_n * np.sum([
        f(a + i * h_n + h_n / 2)
        for i in range(N)
    ])
    while not convergence:
        h_2n = span / (2 * N)
        res_2n = h_2n * np.sum([
            f(a + i * h_2n + h_2n / 2)
            for i in range(2 * N)
        ])
        convergence = abs(res_2n - res_n) <= e
        N *= 2
        res_n = res_2n

    return res_2n, h_2n, N * 2


@meta(name="Метод трапеций")
def trapezoid(f, a, b, e=10 ** (-4), steps_start=10):
    span = b - a
    N = steps_start
    convergence = False
    median = (f(a) + f(b)) / 2

    h_n = span / N
    res_n = h_n * (median + np.sum([
        f(a + i * h_n)
        for i in range(1, N)
    ]))
    while not convergence:
        h_2n = span / (2 * N)
        res_2n = h_2n * (median + np.sum([
            f(a + i * h_2n)
            for i in range(1, 2 * N)
        ]))
        convergence = abs(res_2n - res_n) <= e
        N *= 2
        res_n = res_2n

    return res_2n, h_2n, N * 2


@meta(name="Метод Симпсона")
def simpson_integrator(f, a, b, e=10 ** (-4), steps_start=10):
    convergence = False
    span = b - a

    N = steps_start
    h_n = span / N
    x_n = np.fromfunction(
        lambda i: a + h_n * i,
        (N,)
    )
    s1_n = 2 * np.sum([
        f(x_n[2 * i])
        for i in range(1, (N // 2) - 1)
    ])
    s2_n = 4 * np.sum([
        f(x_n[2 * i - 1])
        for i in range(1, N // 2)
    ])
    res_n = h_n / 3 * (f(a) + s1_n + s2_n + f(b))
    while not convergence:
        h_2n = span / (2 * N)
        x_2n = np.fromfunction(
            lambda i: a + h_2n * i,
            (2 * N,)
        )
        s1_2n = 2 * np.sum([
            f(x_2n[2 * i])
            for i in range(1, N)
        ])
        s2_2n = 4 * np.sum([
            f(x_2n[2 * i - 1])
            for i in range(1, N + 1)
        ])
        res_2n = h_2n / 3 * (f(a) + s1_2n + s2_2n + f(b))
        convergence = abs(res_2n - res_n) <= e
        N *= 2
        res_n = res_2n

    return res_2n, h_2n, N * 2
