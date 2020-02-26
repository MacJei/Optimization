import numpy as np
import types


def approx_grad_and_f(f, x, f_at_x=None, epsilon=0.0001):
    assert type(f) == types.FunctionType
    assert type(x) == np.ndarray
    if f_at_x is None:
        f_at_x = f(x)
    return np.array([
        (
            f(x + epsilon * np.array([
                float(j == i) for j in range(len(x))]
            )) - f_at_x
        ) / epsilon
        for i in range(len(x))
    ]), f_at_x


def ncgd_optimize(f, x, max_iterations=1000):
    assert type(f) == types.FunctionType
    assert type(x) == np.ndarray
    x = np.copy(x)
    prev_direction = None
    prev_antigrad = None
    for _ in range(max_iterations):
        approx_grad, f_at_x = approx_grad_and_f(f, x)
        antigrad = -approx_grad
        if prev_direction is not None:
            beta = (antigrad.T @ antigrad) / (prev_antigrad.T @ prev_antigrad)
            direction = antigrad - beta * prev_direction
        else:
            direction = antigrad
        alpha = 1.
        while f(x + alpha * direction) > f_at_x - 0.9 * alpha * (direction.T @ direction):
            alpha /= 1.5
        while f(x + alpha * direction) <= f_at_x - 0.9 * alpha * (direction.T @ direction):
            alpha *= 1.5
        alpha /= 1.5
        x = x + alpha * direction
        prev_antigrad = antigrad
        prev_direction = direction
    return x


n_calls = 0


def f(x):
    global n_calls
    n_calls += 1
    return (x[0]-2)**2 + (x[1]-3)**2 + 9


print(
    ncgd_optimize(
        f,
        np.array([0, 0])
    )
)
print(n_calls)
