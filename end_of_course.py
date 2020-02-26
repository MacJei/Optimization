import numpy as np
import random

a = [2.699147802102447, 2.7745549105199063, 2.9020196976319053]
f_threshold = 0.01

a_update = np.ones(len(a))


def f(x):
    res = 0
    for i in range(len(x)-1):
        res += (x[i] * x[i + 1] - a[i]) ** 2
    return res


def deriv(a_vec):
    eps = 1 * 10 ** -12
    f_a_vec = f(a_vec)
    partial_deriv = []
    for i in range(len(a_vec)):
        a_vec[i] += eps
        partial_deriv.append((f(a_vec)-f_a_vec)/eps)
        a_vec[i] -= eps
    return np.array(partial_deriv)


def sgd(start=a_update, alpha=.2, max_iters=100000):
    for i in range(max_iters):
        gradient_ = deriv(a_update)
        for i in range(len(a_update)):
            a_update[i] -= alpha*gradient_[i]
        if f(a_update) == f_threshold:
            break


sgd()
print(*a_update)

# 1.68025 1.60644 1.72711
