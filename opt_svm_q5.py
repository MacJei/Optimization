import numpy as np
import math

x = [
    [0.1, 0.5, 3.0],
    [1.0, -1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.1, 0.5, 1.0]
]
y = [1, 1, -1, 1]

# 3.0 -1.0 1.0 0.0
ones = np.array([[1]*len(x)]).T
x = np.concatenate((x, ones), axis=1)


def loss_function(a_vec):
    return sum((max(0, 1 - (y[i]*(x[i]@a_vec)))) for i in range(len(y)))


def deriv(a_vec):
    epsilon = np.random.rand(1)[0]/100
    f_a_vec = loss_function(a_vec)
    partial_deriv = []
    for i in range(len(a_vec)):
        a_vec[i] += epsilon
        partial_deriv.append((-f_a_vec + loss_function(a_vec))/epsilon)
        a_vec[i] -= epsilon
    return np.array(partial_deriv)


gr = (1+(5**0.5))/2
a = 0
b = 1


def gss(f, a_vec, a_vec_deriv, a, b):
    tol = 0.001
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc = f(a_vec - c*a_vec_deriv)
    fd = f(a_vec - d*a_vec_deriv)
    while abs(c - d) > tol:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) / gr
            fc = f(a_vec - c*a_vec_deriv)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) / gr
            fd = f(a_vec - d*a_vec_deriv)
    return (b + a) / 2


def optimizer():
    a_vec = np.random.rand(len(x[0]))
    a_vec_deriv = deriv(a_vec)
    for _ in range(10000):
        alpha = gss(loss_function, a_vec, a_vec_deriv, a, b)
        a_vec -= alpha*a_vec_deriv
        a_vec_deriv = deriv(a_vec)
    return a_vec


print(*optimizer())
