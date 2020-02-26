import numpy as np
from scipy import optimize as opt

x = -5
c1 = 0.5
tol = 10**-8


def f(x):
    return x**2 + 4*x - 7


def f_(x):
    return (f(x+tol)-f(x-tol))/(2*tol)


for _ in range(100):
    alpha, _, _, _, _, _ = opt.line_search(
                                            f,
                                            f_,
                                            np.array(x),
                                            np.array(-f_(x)),
                                            c1=c1)
    x += -(alpha)*f_(x)
print('x:::', x)