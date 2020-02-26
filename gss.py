import math
gr = (math.sqrt(5)-1)/2


def gss(f, a, b, tol):
    c = b - (gr * (b-a))
    d = a + (gr * (b-a))

    fc = f(c)
    fd = f(d)

    while abs(c - d) > tol:
        if fd > fc:
            b = d
            d = c
            fd = fc
            c = b - (gr * (b-a))
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (gr * (b-a))
            fd = f(d)
    return (c+d)/2


print(gss(f, a, b, tol))
