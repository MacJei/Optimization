""" SGD take in many points. From those points chose a single data point. Get gradient at chosen point and move along that gradient.  """
import random

# Calculate alpha somewhere else


def stochastic_gd(f, points, alpha, tol=.001,  maxiter=100):
    eps = 0.001
    limit = len(points)
    for _ in range(maxiter):
        here = points[random.randint(0, limit)]
        grad = (f(here + eps) - f(here)) / tol
        here += - alpha * grad
    return here


def f(x):
    return  # Crazy ass function
