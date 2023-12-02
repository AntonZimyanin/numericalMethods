from math import sqrt


# wolfram : integral_0.6^1.724 sqrt(x + x^3)dx = 1.89818


def trapezoidal(a, b, n, f=lambda x: sqrt(x + x**3), eps=1e-4):
    h = (b - a) / n
    I_prev = h * (f(a) + f(b)) / 2

    while True:
        h /= 2
        x = [a + i * h for i in range(1, 2 * n, 2)]
        I_curr = I_prev / 2 + h * sum(f(x_i) for x_i in x)
        if abs(I_curr - I_prev) <= 3 * eps:
            return I_curr
        I_prev = I_curr
        n *= 2


def simpson(a, b, n, f=lambda x: sqrt(x + x**3), eps=1e-5):
    h = (b - a) / n
    I_prev = h * (f(a) + 4 * f((a + b) / 2) + f(b)) / 6

    while True:
        h /= 2
        n *= 2
        x = [a + i * h for i in range(1, n, 2)]
        I_curr = I_prev / 2 + h * sum(f(x_i) for x_i in x)
        if abs(I_curr - I_prev) < 15 * eps:
            return I_curr
        I_prev = I_curr


def simpson_volume(a, b, c, d, n, f=lambda x, y: 1 / (x + y) ** 2, eps=1e-5):
    m = n
    h = (b - a) / n
    k = (d - c) / n
    I_prev = 0
    for i in range(n + 1):
        for j in range(m + 1):
            x = a + i * h
            y = c + j * k
            if (i == 0 or i == n) and (j == 0 or j == m):
                coef = 1
            elif i == 0 or i == n or j == 0 or j == m:
                coef = 2
            else:
                coef = 4
            I_prev += coef * f(x, y)
    I_prev *= h * k / 4

    while True:
        h /= 2
        k /= 2
        n *= 2
        m *= 2
        I_curr = 0
        for i in range(n + 1):
            for j in range(m + 1):
                x = a + i * h
                y = c + j * k
                if (i == 0 or i == n) and (j == 0 or j == m):
                    coef = 1
                elif i % 2 == 0 and j % 2 == 0:
                    coef = 1
                elif i % 2 == 1 and j % 2 == 1:
                    coef = 16
                else:
                    coef = 4
                I_curr += coef * f(x, y)
        I_curr *= h * k / 9
        if abs(I_curr - I_prev) < 15 * eps:
            return I_curr
        I_prev = I_curr


if __name__ == "__main__":
    n = 100
    a, b = [0.6, 1.724]
    func_list = [trapezoidal, simpson]

    for i in func_list:
        print(f"{i.__name__} method =  {i(a, b, n)}")

    # y = 1/ (x + y) ** 2
    a, b = [3.0, 4.0]
    c, d = [1.0, 2.0]
    print(f"simpson method double integral = {simpson_volume(a, b, c, d, n)}")
