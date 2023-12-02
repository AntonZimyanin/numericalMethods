import matplotlib.pyplot as plt
import numpy as np

from project.firstLab.Gaussian import gaussian

# from project.firstLab.ldl_t import LDL_T


x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 87, 165, 210, 238, 252, 239, 211, 158, 90, -5])


def multiply_mx_to_arr(a, b):
    result = np.zeros((len(a), 1))

    for i in range(len(a)):
        for j in range(1):
            for k in range(len(b)):
                result[i][0] += a[i][k] * b[k]

    return result


def multiply(a, b):

    result = np.zeros((len(a), len(b[0])))
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]

    return result


def variance(X, coeffs):
    y_pred = multiply(X, np.array([coeffs]))
    y_pred = np.dot(X, coeffs)
    residuals = y - y_pred
    mean_residual = np.mean(residuals)

    return np.mean((residuals - mean_residual) ** 2)


def PolyCoefficients(x, coeffs):
    """Returns a polynomial for ``x`` values for the ``coeffs`` provided.
    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """

    o = len(coeffs)
    # print(f'# This is a polynomial of order {o}.')
    y = 0
    for i in range(o):
        y += coeffs[i] * x**i
    return y


def least_square_method():
    X = np.column_stack((np.ones(len(x)), x, x**2))
    XTX = multiply(X.T, X)
    XTy = multiply_mx_to_arr(X.T, y)
    # print(XTX, XTy, sep='\n')

    aug_matrix = np.column_stack((XTX, XTy))
    coeffs = gaussian(aug_matrix)
    # print(coeffs)
    # coeffs = np.linalg.solve(XTX, XTy)
    print("Коэффициенты полинома: ", *coeffs, sep="\n")

    residual_variance = variance(X, coeffs)
    print("Остаточная дисперсия: ", residual_variance, sep="\n")

    # import multipolyfit as mpf
    # stacked_x = np.array([x,x+1,x-1])
    # coeffs = mpf(stacked_x, y)
    x2 = np.arange(min(x) - 1, max(x) + 1, 0.01)  # use more points for a smoother plot
    y2 = np.polyval(coeffs, x2)  # Evaluates the polynomial for each x2 value

    # plt.figure(figsize=(10, 6))
    # plt.plot(x, PolyCoefficients(x, coeffs))
    # plt.plot(x, y, "o", label="Исходные данные")
    # plt.plot(x, np.polyval(coeffs, x), "r", label="Аппроксимация")

    # plt.legend()
    # plt.show()

    # plt.plot(x, y, 'o', label='Исходные данные')
    # # plt.plot(x, np.polyval(coeffs, x), 'r', label='Аппроксимация')
    # plt.plot(x2, y2, 'r', label='Аппроксимация')
    # plt.legend()
    # plt.show()

    def f(x):
        return 1 + 100 * x - 10 * x**2

    x_cords = range(-60, 60)
    y_cords = [1 + 100 * x - 10 * x**2 for x in x_cords]

    plt.plot(x_cords, y_cords)
    plt.show()
    # plt.plot(x, y, label="y = 1 + 100x - 10x^2")
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('График функции y = 1 + 100x - 10x^2')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
