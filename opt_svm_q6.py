""" Solve a system of linear equations wherein Ax = b by solving a corresponding optimizations problem ie (Ax-B).T(Ax-b) -> min """

import numpy as np

A = [
    [0.99, 0.87, 0.51, 0.27],
    [0.34, 0.22, 0.28, 0.34],
    [0.86, 0.16, 0.14, 0.76],
    [0.74, 0.36, 0.34, 0.67]
]
b = [0.22, 0.56, 0.12, 0.32]
tol = 0.001
# 1.9250001434580333 -4.800059678539262 6.003006880246984 -2.115673076371321

A = np.array(A)
b = np.array(b)


def loss(x):
    return f(x)**2


def gradient(x):
    eps = 0.001
    grad = np.zeros(len(x))
    for i in range(len(x)):
        x_eps = np.copy(x)
        x_eps[i] = x[i] + eps
        grad[i] = (
            loss(x_eps) - loss(x)
        ) / eps
    print('gradient_____',grad)
    return grad


def armijo(x, maxiter=100, c1=.5,  c2=1.9):
    alpha = 0.4
    for _ in range(maxiter):
        while not(f(x - alpha * gradient(x)) <= f(x) - c1 * (gradient(x)**2) * alpha).all():
            alpha *= 0.7

        while not (f(x - gradient(x) * alpha * c2) > f(x) - c1 * gradient(x) ** 2 * c2 * alpha).all():
            alpha *= c2
        x += - alpha*gradient(x)
    # print('Alpha: ', alpha)
    return alpha


def f(x):
    return np.linalg.norm(A @ x.T - b.T)


def minimize():
    x = np.zeros(len(A))
    while loss(x) > tol:
        alpha = armijo(x)
        x += alpha * -gradient(x)
        # print('Loss: ', loss(x))
    return x


print(*minimize())
