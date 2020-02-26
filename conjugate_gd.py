"""
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
"""

import numpy as np

def conjugate_grad(A, b, x=None):
    n = len(b)
    if not x:
        x = np.ones(n)
    r = np.dot(A, x) - b
    p = - r
    r_k_norm = np.dot(r, r)

    for i in range(2*n):
        Ap = np.dot(A, p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            print('Breaking')
            break
        p = beta * p - r
    return x

import random
if __name__ == '__main__':
    n = 10
    P = np.random.normal(size=[n, n])
    # print("P ", P)
    A = np.dot(P.T, P)
    b = np.ones(n)
    x = conjugate_grad(A, b)

    print(np.dot(A, x) - b)