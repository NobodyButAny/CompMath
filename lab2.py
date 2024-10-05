# Вариант 9
from argparse import ArgumentError

import numpy as np
import numpy.linalg as la

A = np.array([
    [1.85, 0.7, -0.12, -0.18],
    [0.16, 0.19, 0.79, 0.11],
    [1.13, 2.77, 0.18, -0.2],
    [1.14, 1.01, 0.55, 3.22]
])

B = np.array([8.41, -0.23, 13.91, 9.58])


def is_square_matrix(arr: np.ndarray):
    return len(arr.shape) == 2 and arr.shape[0] == arr.shape[1]


def is_row_vector(arr: np.ndarray):
    return len(arr.shape) == 1


def is_diagonally_dominant(A):
    return all([
        abs(A[i][i]) >= sum(np.abs(A[i])) - abs(A[i][i])
        for i in range(len(A))
    ])


def maximize_diagonal(A: np.ndarray):
    if len(A.shape) != 2:
        raise ValueError("A should be a matrix!")

    if A.shape[1] < A.shape[0]:
        raise ValueError("Axis-1 should be >= axis-0, or A should be square!")

    res = A.copy()
    for row_i in range(A.shape[0]):
        max_i = np.argmax(res[row_i:, row_i])
        res[[row_i, max_i]] = res[[max_i, row_i]]
    return res


def seidel_iter(A: np.ndarray, B: np.ndarray, e=10 ** (-4)):
    X = np.ones(B.shape)
    N = len(X)

    steps = 0
    norm_diff_predicate = False
    X_nx = X.copy()
    while not norm_diff_predicate:
        for i in range(N):
            s1 = sum(
                A[i, j] * X_nx[j]
                for j in range(N)
                if j != i
            )
            s2 = sum(
                A[i, j] * X[j]
                for j in range(i + 1, N)
            )
            X_nx[i] = (B[i] - s1 - s2) / A[i, i]
        norm_diff_predicate = la.norm(X_nx - X) <= e
        X = X_nx.copy()
        steps += 1
    return X, steps


def gauss_elimination(A: np.ndarray, B: np.ndarray):
    if not is_square_matrix(A):
        raise ValueError('A should be a square matrix!')

    if not is_row_vector(B) or B.shape[0] != A.shape[0]:
        raise ValueError(f'B should be size {A.shape[0]} row vector!')

    N = len(B)
    aug_matr = np.column_stack((A, B)).astype(dtype='float')

    # Прямой проход
    for i in range(N):
        aug_matr[i] = aug_matr[i] / aug_matr[i, i]
        for k in range(i + 1, N):
            aug_matr[k] = aug_matr[k] - (aug_matr[i] * aug_matr[k, i])

    # Обратный проход
    x = np.zeros(B.shape)
    main, col = aug_matr[:, :-1], aug_matr[:, -1]

    x[-1] = col[-1] / main[-1, -1]
    for m in reversed(range(N - 1)):
        x[m] = col[m] - np.dot(x, main[m])
    return x


x = np.array([
    [1, 2, 3],
    [0, 2, 1],
    [1, 2, 4]
], dtype='float')
y = np.array([15, 7, 18], dtype='float')
