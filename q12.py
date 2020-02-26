from math import sin, cos
tol = 0.0001
x_init = 1.0
y_init = -1.0


def f(x, y): return cos(x*y)-0.5


def g(x, y): return sin(x+y)-0.3

# 1.1869509 -0.8822584


def f_x(x, y):
    return (f(x + tol, y) - f(x, y)) / (tol)


def f_y(x, y):
    return (f(x, y + tol) - f(x, y)) / (tol)


def g_x(x, y):
    return (g(x + tol, y) - g(x, y)) / (tol)


def g_y(x, y):
    return (g(x, y + tol) - g(x, y)) / (tol)


def newtons(x, y):
    NMAX = 50
    for i in range(NMAX):
        ff = f(x, y)
        fx = f_x(x, y)
        fy = f_y(x, y)
        gg = g(x, y)
        gx = g_x(x, y)
        gy = g_y(x, y)
        d = (fx * gy - fy * gx)
        x += ((-ff * gy + fy * gg) / d)
        y += ((gx * ff - fx * gg) / d)
    return x, y


a, b = newtons(x_init, y_init)

print("%.7f %.7f" % (a, b))
