from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# example functions
def f(x):
    return 3*np.cos(x)

def f1(s, i):
    return -.6 * s * i + .1 * (1 - s - i)


def f2(s, i):
    return 0.6 * s * i - .34 * i


def logistic(y, lasty):
    return 1 * y * (1 - lasty)


def eulersmethod(a, x, y, approx, step):
    if approx - x == 0:
        return 0
    step = (approx - x) / step
    xv = np.zeros(((approx - x)/step + 1))
    yv = np.zeros(((approx - x)/step + 1))
    i = 0
    if approx > x:
        while x <= approx:
            xv[i] = x
            yv[i] = y
            w = (a(x, y)) * step
            x += step
            y += w
            i += 1

    else:
        while x >= approx:
            xv[i] = x

            yv[i] = y
            w = (a(x, y)) * step
            x += step
            y += w
            i += 1
    print (xv, yv)
    plt.plot(xv, yv)
    plt.show()

eulersmethod(f1, 0, 1, 2, 15)


def eulersmethod2(a, x, s, b, i, approx, step):
        step = (approx - x) / step
        time = np.zeros(((approx - x) / step + 1))
        f1v = np.zeros(((approx - x) / step + 1))
        f2v = np.zeros(((approx - x) / step + 1))
        l = 0
        while x <= approx:
            time[l] = x
            f1v[l] = s
            f2v[l] = i
            w1 = a(s, i) * step
            w2 = b(s, i) * step
            x += step
            s += w1
            i += w2
            l += 1
        plt.plot(time, f1v)
        plt.show()
        plt.plot(time, f2v)
        plt.show()
        plt.plot(f1v, f2v)
        plt.show()
eulersmethod2(f1, 0, .9, f2, .1, 100, 100)


def rungekutta(func1, x, y, approx, step):
    h = (approx - x) / step
    time = []
    f1_v = []
    for k in range(step + 1):
        time.append(h * k)
        f1_v.append(y)

        k1 = (func1(x))

        k2 = (func1(x + .5 * h))

        k3 = (func1(x + .5 * h))

        k4 = (func1(x + h))

        y += h * ((k1 + 2 * k2 + 2 * k3 + k4) / 6)
        x += h
    plt.plot(time, f1_v)
    plt.show()
rungekutta(f, 0, 1, 10, 100)


def rungekutta2(func1, func2, start, y1, y2, approx, step):
        h = (approx - start) / step
        time = []
        f1_v = []
        f2_v = []
        r = []
        for k in range(step + 1):
            time.append(h * k)
            f1_v.append(y1)
            f2_v.append(y2)
            r.append(1-y1-y2)

            k1_1 = (func1(y1, y2))
            k1_2 = (func2(y1, y2))

            k2_1 = (func1(y1 + .5 * k1_1 * h, y2 + 0.5 * k1_1 * h))
            k2_2 = (func2(y1 + .5 * k1_2 * h, y2 + 0.5 * k1_2 * h))

            k3_1 = (func1(y1 + .5 * k2_1 * h, y2 + 0.5 * k2_1 * h))
            k3_2 = (func2(y1 + .5 * k2_2 * h, y2 + 0.5 * k2_2 * h))

            k4_1 = (func1(y1 + k3_1 * h, y2 + k3_1 * h))
            k4_2 = (func2(y1 + k3_2 * h, y2 + k3_2 * h))

            y1 += h*((k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6)
            y2 += h*((k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6)
        plt.plot(time, f1_v)
        plt.plot(time, f2_v)
        plt.plot(time, r)
        plt.show()
        plt.plot(f1_v, f2_v)
        plt.show()
rungekutta2(f1, f2, 0, .9, .1, 200, 200)


def abmethod(func, x, y, approx, step):
    h = (approx - x) / step
    time = []
    values = []
    for k in range(3):
        time.append(h * k)
        values.append(y)

        k1 = (func(x))

        k2 = (func(x + .5 * h))

        k3 = (func(x + .5 * h))

        k4 = (func(x + h))

        y += h * ((k1 + 2 * k2 + 2 * k3 + k4) / 6)
        x += h
    l = 3
    while l < step:
        time.append(h * l)
        y += (h/24)*(55*func(time[l - 1] + h) - 59*func(time[l - 1]) + 37*func(time[l - 2]) - 9*func(time[l - 3]))
        values.append(y)
        l += 1

    plt.plot(time, values)
    plt.show()

abmethod(f, 0, 1, 100, 100)

# Approximats delayed differential equations
def dde(func, x, y, approx, step, delay):
    h = (approx - x) / step
    time = []
    values = []
    for k in range(delay + 3):
        time.append(h * k)
        values.append(y)
    l = delay + 3
    while l < step:
        time.append(h * l)
        y += (h/24)*(55*func(y, values[l - delay]) - 59*func(values[l - 1], values[l - 1 - delay]) +
                     37*func(values[l - 2], values[l - 2 - delay]) - 9*func(values[l - 3], values[l - 3 - delay]))
        values.append(y)
        l += 1
    plt.plot(time, values)
    plt.show()
dde(logistic, 0, 2, 100, 1000, 10)
