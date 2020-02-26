"""
    Description
    -----------
    Solve an equation of n variables Ax = b with non-linear conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) strtrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
"""
import numpy as np


def gradient(x, tol):
    return np.array([
        (
            f(x + tol * np.array([
                float(j == i) for j in range(len(x))]
            )) - f(x)
        ) / tol for i in range(len(x))])


def non_linear_conjugate_gd(f, x):
    x = np.array(x)
    maxiter = 100
    tol = 1 * 10 ** -6

    dir_start = None
    anti_grad_start = None

    for _ in range(maxiter):
        grad = gradient(x, tol)
        anti_grad = -grad

        if dir_start is not None:
            beta = (anti_grad.T @ anti_grad) /\
            (anti_grad_start.T @ anti_grad_start)
            dir_ = np.array(anti_grad - beta * dir_start)
        dir_ = np.array(anti_grad)
        alpha = .1

        while f(x + alpha * dir_) > f(x) - 0.9 * alpha * (dir_.T @ dir_):
            alpha /= 1.5
        while f(x + alpha * dir_) <= f(x) - 0.9 * alpha * (dir_.T @ dir_):
            alpha *= 1.5
        alpha /= 1.5
        x = x + alpha * dir_
        anti_grad_start = anti_grad
        dir_start = dir_
    return x


def f(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2 + 9


print(non_linear_conjugate_gd(f, np.array([0, 0])))
