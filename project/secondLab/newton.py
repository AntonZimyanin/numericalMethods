from typing import List
from typing import Union

import numpy as np

from project.firstLab.Gaussian import gaussian

# from project.secondLab.config import LIM_TIER
# from project.secondLab.config import eps, x1, x2, initial_guess
# x1 = -1.485348762093568
# x2 = 0.763065261962855


def jacobian_iter(f, x):
    n = len(x)
    eps = np.sqrt(np.finfo(float).eps)
    Fx = f(x)
    jac = np.zeros([n, n])
    for i in range(n):
        xx = x.copy()
        xx[i] += eps
        jac[:, i] = (f(xx) - Fx) / eps
    return jac


def newton(f, u, eps=1e-6, lim_iter=50):
    for i in range(lim_iter):
        F = f(u)
        J = jacobian_iter(f, u)
        aug_matrix = np.column_stack((J, -F))
        delta = np.linalg.solve(J, -F)
        u += delta
        if (
            np.linalg.norm(delta, ord=np.inf) < eps
            and np.linalg.norm(F, ord=np.inf) < eps
        ):
            break
    return u


# def jacobian_iter(x):
#     n = len(x)
#     eps = np.sqrt(np.finfo(float).eps)
#     Fx = get_equations(x)
#     jac = np.zeros([n, n])
#     for i in range(n):
#         xx = x.copy()
#         xx[i] += eps
#         jac[:, i] = (get_equations(xx) - Fx) / eps
#     return jac


# def get_equations(vars) -> Union[List, None]:
#     """
#     :param vars: x1, x2
#     :return: List[float, float] or None
#     """
#     x1, x2 = vars
#     eq1 = x1**2 * x2**2 - 3 * x1**2 - 6 * x2**3 + 8
#     eq2 = x1**4 - 9 * x2 + 2

#     return np.array([eq1, eq2])


# def get_delta(initial_guess, dx) -> List:
#     if np.all(np.all(initial_guess) < 1):
#         delta2 = np.abs(dx).max()
#     else:
#         delta2 = np.abs((dx / initial_guess)).max()

#     delta1 = np.abs(get_equations(initial_guess)).max()

#     return [delta1, delta2]


# def newton(eps=1e-6, lim_iter=50):
#     initial_guess = np.array([-1.5, 1.5])
#     # print(jacobian_iter(initial_guess))
#     # jacobian = np.array(
#     #     [
#     #         [
#     #             2 * initial_guess[0] * initial_guess[1] ** 2 - 6 * initial_guess[0],
#     #             2 * initial_guess[0] ** 2 * initial_guess[1] - 18,
#     #         ],
#     #         [4 * initial_guess[0] ** 3, -9],
#     #     ]
#     # )
#     # print(jacobian)
#     jacobian = jacobian_iter(initial_guess)

#     if np.all(np.abs(get_equations(initial_guess))) < eps:
#         print(f"***********************************")
#         print(f"x1 = {initial_guess[0]}")
#         print(f"x2 = {initial_guess[1]}")
#         return

#     jacobian = jacobian_iter(initial_guess)
#     # jacobian = np.array([
#     #     [2 * initial_guess[0] * initial_guess[1]**2 - 6 * initial_guess[0], 2 * initial_guess[0]**2 * initial_guess[1] - 18],
#     #     [4 * initial_guess[0]**3, -9]
#     # ])
#     print("Iteration\tdelta1\t\t\tdelta2")

#     for i in range(1, lim_iter):
#         values = get_equations(initial_guess)
#         # jacobian = np.array(
#         #     [
#         #         [
#         #             2 * initial_guess[0] * initial_guess[1] ** 2 - 6 * initial_guess[0],
#         #             2 * initial_guess[0] ** 2 * initial_guess[1]
#         #             - 18 * initial_guess[1] ** 2,
#         #         ],
#         #         [4 * initial_guess[0] ** 3, -9],
#         #     ]
#         # )

# aug_matrix = np.column_stack((jacobian, -values))

#         dx = gaussian(aug_matrix)

#         initial_guess += dx

#         delta1, delta2 = get_delta(initial_guess, dx)

#         # initial_guess = xK[initial_guess] + dx[update]

#         print(f"Iteration {i}\t{delta1}\t{delta2}")
#         if delta1 <= eps and delta2 <= eps:
#             print(f"\nIteration {i}")
#             print(f"x1 = {initial_guess[0]}")
#             print(f"x2 = {initial_guess[1]}")
#             print(f"***********************************")
#             return

#     print("Итерациооный метод не сошелся")
