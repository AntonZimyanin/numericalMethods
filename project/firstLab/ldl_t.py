from typing import Optional

import numpy as np

from project.firstLab.config import Matrix

# from config import Matrix

LEN_MATRIX = 3


def ldl_t(A: Matrix, l: Matrix, d: Matrix, l_t: Matrix = Optional) -> None:
    """
    :params: L, D, L^T
    modificate matrix
    """
    d[0, 0] = A[0, 0]
    len_mtx = len(A)

    # l[1][0] = l_t[0][1] = mtx[1][0] / d[0][0]
    # l[2][0] = l_t[0][2] = mtx[2][0] / d[0][0]

    # d[1][1] = mtx[1][1] - d[0][0] * l[1][0] ** 2
    # l[2][1] = l_t[1][2] = (mtx[2][1] - d[0][0] * l[1][0] * l[2][0]) / d[1][1]
    # d[2][2] = mtx[2][2] - d[0][0] * l[2][0] ** 2 - d[1][1] * l[2][1] ** 2

    # for i in range(len_mtx):
    #     l[i, i] = l_t = 1.0

    print("\nL\n", l)
    print("\nD\n", d)
    print("\nL^T\n", l_t)

    for i in range(len_mtx):
        l[i, i] = 1.0

        for j in range(i, len_mtx):
            for k in range(i):
                A[j, i] -= l[j, k] * d[k, k] * l[i, k]

            if i == j:
                d[i, i] = A[i, i]
            else:
                l[j, i] = A[j, i] / d[i, i]

    l_t = l.copy().transpose()

    # print("\nL\n", l)
    # print("\nD\n", d)
    # print("\nL^T\n", l_t)

    # new_a = np.zeros((LEN_MATRIX, LEN_MATRIX), dtype=float)

    # print("new solution")
    # for i in range(len_mtx):
    #     l[i, i] = 1.0

    #     for j in range(i, len_mtx):
    #         new_a[j, i] = A[j, i]
    #         for k in range(i):
    #             new_a[j, i] -= new_a[i, k] * l[j, k]

    #         if i == j:
    #             d[i, i] = new_a[i, i]
    #         else:
    #             l[j, i] = new_a[j, i] / d[j, j]

    # print("\nL\n", l)
    # print("\nD\n", d)
    # print("\nL^T\n", l_t)


def ldl_t_solution(l: Matrix, d: Matrix, l_t: Matrix, b: Matrix):
    # L * y = b

    augmented_matrix = np.column_stack((l, b))
    print(f"\nL * y = b: \n {augmented_matrix}\n")

    mtx_len = len(augmented_matrix)
    num_columns = len(augmented_matrix[0])
    y = np.zeros((mtx_len, 1), dtype=float)

    for i in range(0, mtx_len):
        b_i = augmented_matrix[i][-1]
        for j in range(0, i):
            b_i -= augmented_matrix[i][j] * y[j][0]
            # print(f"{ b_i} -= {augmented_matrix[i][j]} * {y[j][0]}")
        if augmented_matrix[i][i] == 0:
            ZeroDivisionError()
        y[i][0] = b_i

        # y[i][0] = b_i / augmented_matrix[i][i]

    # print("\ny\n", y)

    """
    :params: y: Matrix y (answer), matrix_d: Matrix d
    new matirx z
    """

    z = np.zeros((mtx_len, 1), dtype=float)

    for i in range(mtx_len):
        z[i][0] = y[i][0] / d[i][i]

    # print("\nz\n", z)

    new_augmented_matrix = np.column_stack((l_t, z))
    # print(new_augmented_matrix)

    mtx_len = len(new_augmented_matrix)
    num_columns = len(new_augmented_matrix[0])

    x = np.zeros((mtx_len, 1), dtype=float)

    for i in range(mtx_len - 1, -1, -1):
        b_i = new_augmented_matrix[i][-1]
        for j in range(i + 1, num_columns - 1):
            b_i -= new_augmented_matrix[i][j] * x[j][0]
        if new_augmented_matrix[i][i] == 0:
            raise ZeroDivisionError()
        x[i][0] = b_i

    return x


def LDL_T(A, B):
    l = np.zeros((len(A), len(A[0])), dtype=float)
    d = np.zeros((len(A), len(A[0])), dtype=float)
    l_t = np.zeros((len(A), len(A[0])), dtype=float)

    ldl_t(A, l, d, l_t)
    l_t = l.transpose()

    x = np.zeros((len(A), 1), dtype=float)

    x = ldl_t_solution(l, d, l_t, B)

    return x


# if __name__ == "__main__":

#     A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)

#     B = np.array(
#         [8, 12, 16],
#         dtype=float,
#     )

#     l = np.zeros((LEN_MATRIX, 3), dtype=float)
#     d = np.zeros((LEN_MATRIX, 3), dtype=float)
#     l_t = np.zeros((LEN_MATRIX, 3), dtype=float)

#     ldl_t(A, l, d, l_t)
#     l_t = l.transpose()

#     print(f"\nl:\n{l}\n, \nd:\n{d}\n, \nl_t:\n{l_t}\n", sep="\n\n")

#     x = np.zeros((LEN_MATRIX, 1), dtype=float)

#     x = ldl_t_solution(l, d, l_t, B)
#     solition = [x[i][0] for i in range(LEN_MATRIX)]
#     print(f"soliton:")
#     print(*solition, sep="\n")
