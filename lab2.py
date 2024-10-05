# Вариант 9

import numpy as np
import numpy.linalg as la

A = np.array([
    [1.85, 0.7, -0.12, -0.18],
    [0.16, 0.19, 0.79, 0.11],
    [1.13, 2.77, 0.18, -0.2],
    [1.14, 1.01, 0.55, 3.22]
])

B = np.array([8.41, -0.23, 13.91, 9.58])


def is_diagonally_dominant(A):
    return all([
        abs(A[i][i]) >= sum(np.abs(A[i])) - abs(A[i][i])
        for i in range(len(A))
    ])


def seidel_iter(A: np.ndarray, B: np.ndarray, e=10 ** (-4), steps_at_least=None):
    X = np.ones(B.shape)
    N = len(X)

    steps = 0
    norm_diff_predicate = False
    steps_predicate = True
    X_nx = X.copy()
    while not norm_diff_predicate or not steps_predicate:
        for i in range(N):
            s1 = sum(
                A[i, j] * X_nx[j]
                for j in range(N)
                if j != i
            )
            # s2 = sum(
            #     A[i, j] * X[j]
            #     for j in range(i + 1, N)
            # )
            X_nx[i] = (B[i] - s1) / A[i, i]
        norm_diff_predicate = la.norm(X_nx - X) <= e
        if steps_at_least is not None:
            steps_predicate = steps >= steps_at_least
        X = X_nx.copy()
        steps += 1
    return X, steps


def gauss_elimination(A: np.ndarray, B: np.ndarray):
    N = len(B)
    aug_matr = np.column_stack((A, B)).astype(dtype='float')

    # Прямой проход
    for i in range(N):
        aug_matr[i] = aug_matr[i] / aug_matr[i, i]
        for k in range(i + 1, N):
            aug_matr[k] = aug_matr[k] - (aug_matr[i] * aug_matr[k, i])

    # Обратный проход
    x = np.zeros(B.shape)
    matr, col = aug_matr[:, :-1], aug_matr[:, -1]
    x[-1] = col[-1] / matr[-1, -1]

    for m in reversed(range(N - 1)):
        x[m] = col[m] - np.dot(x, matr[m])
    return x


x = np.array([
    [1, 2, 3],
    [0, 2, 1],
    [1, 2, 4]
])
y = np.array([15, 7, 18])

print('\n answer ', gauss_elimination(A, B))
