""" Nelder mead uses a sample of n+1 vertices  to find the gradient of a population, then by reflecting, expanding, contracting and shrinking, get the approximately best minima"""
import numpy as np
import math


def nelder_mead(f, x_start, step=0.1, no_improve_thr=10e-6, no_improv_break=10,  max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5):

    # init
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(len(x_start)):
        x = x_start
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * len(x_start)
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres


def f(x):
    return math.sin(x[0]) * math.cos(x[1]) # * (1. / (abs(x[2]) + 1))
    # return (x[0]**2 + x[1] - 11)**2 + (x + x[1]**2-7)**2


print(
    nelder_mead(
        f,
        np.array([8, 10])
    )
)
