"""Var 3"""
from decimal import Decimal
from typing import Union

import numpy as np

from project.firstLab.config import Matrix

# from project.firstLab.config import A
# from project.firstLab.config import B


def swap_mx_row(matrix: Matrix, row_num: int) -> None:
    """
    swap max row and

    ***
    modificate matrix

    first row is mx
    j --> mx_row in matrix
    ***

    :params: matrix
    :return: None
    """
    mx = matrix[row_num][0]
    mx_row = row_num
    mx_len = len(matrix)
    for j in range(1, mx_len):
        if matrix[j][0] > mx:
            mx_row = j
            mx = matrix[j][0]

    if mx != matrix[row_num][0]:
        matrix[row_num], matrix[mx_row] = matrix[mx_row].copy(), matrix[row_num].copy()


def diagonal_view(augmented_matrix: Matrix) -> None:
    """
    reduce the matrix to diagonal form
    """

    mtx_len = len(augmented_matrix)

    for row_num in range(0, mtx_len):

        '''теперь на месте row_num стоит "максимальная строка"'''
        swap_mx_row(augmented_matrix, row_num)

        mx_row = row_num

        """
        divide the matrix
        """

        # variables for division
        len_div_mtx = len(augmented_matrix[mx_row])
        divider = augmented_matrix[mx_row][mx_row]

        if divider == 0:
            raise ZeroDivisionError()

        for i in range(0, len_div_mtx):
            augmented_matrix[mx_row][i] = augmented_matrix[mx_row][i] / divider

        """
        end div
        """

        """
        multiply current row (array) and subtract from the next row (array)
        mult_num --> multiply number in current row (array)
        mult_array --> multiply array is temporality array (row)
                       for substarct next row (array)
        """

        for j in range(1, mtx_len - mx_row):
            next_row_num = mx_row + j

            mult_num = -augmented_matrix[next_row_num][mx_row]
            mult_array = [i * mult_num for i in augmented_matrix[mx_row]]
            res = [x + y for x, y in zip(mult_array, augmented_matrix[next_row_num])]
            augmented_matrix[next_row_num] = res


def gaussian(augmented_matrix: Matrix) -> Matrix:
    """
    :paramaugmented_matrix (Matrix): A 2D list representing an augmented matrix, where the last column
      contains the constants of a system of linear equations.

    :return: Matrix

    return exemple:
    [
        [-17.0]
        [  3.0]
        [-20.0]
                ]

    """

    diagonal_view(augmented_matrix)

    mtx_len = len(augmented_matrix)
    num_columns = len(augmented_matrix[0])

    mtx_answer = np.zeros(len(augmented_matrix), dtype=float)

    # a * x = b --> b_i

    # for i in range(mtx_len - 1, -1, -1):
    #     b_i = augmented_matrix[i][-1]
    #     for j in range(i + 1, num_columns - 1):
    #         print(f"j = {j} : {augmented_matrix[i][j]}*{ mtx_answer[j][0]}")
    #         b_i -= augmented_matrix[i][j] * mtx_answer[j][0]
    #     print(f"{ b_i} / {augmented_matrix[i][i]}")
    #     mtx_answer[i][0] = b_i / augmented_matrix[i][i]

    for i in range(mtx_len - 1, -1, -1):
        b_i = augmented_matrix[i][-1]
        for j in range(i + 1, num_columns - 1):
            b_i -= augmented_matrix[i][j] * mtx_answer[j]
        if augmented_matrix[i][i] == 0:
            raise ZeroDivisionError()
        mtx_answer[i] = b_i / augmented_matrix[i][i]

    return mtx_answer


def get_new_matrix_b(x: Matrix, matrix_a: Matrix) -> Matrix:
    """
    :params: x: Matrix X (answer), matrix_a: Matrix A
    :return: new matirx B
    """
    num_row = len(matrix_a)
    matrix_b = np.zeros((len(matrix_a), 1), dtype=float)

    for row in range(num_row):
        for col in range(num_row):
            matrix_b[row][0] += x[col][0] * matrix_a[row][col]

    return matrix_b


def get_residual_vector(first_b: Matrix, second_b: Matrix) -> Matrix:

    mtx_len = len(first_b)
    residual_vector = np.zeros((len(first_b), 1), dtype=float)

    for i in range(mtx_len):
        residual_vector[i][0] = second_b[i][0] - first_b[i]

    return residual_vector


def substract_matrix(mtx1: Matrix, mtx2: Matrix) -> Matrix:
    num_row = len(mtx1)
    num_col = len(mtx1[0])

    solution = np.zeros((num_row, num_col), dtype=Decimal)

    # compare matrix dimension
    for i in range(num_row):
        for j in range(num_col):
            solution[i][j] = Decimal.from_float(mtx1[i][j]) - Decimal.from_float(
                mtx2[i][j]
            )

    return solution


def get_relative_error(
    second_solution: Matrix, third_solution: Matrix
) -> Union[float, str]:
    """
    Relative error = absolute error / measured value
    """

    absolute_error = substract_matrix(third_solution, second_solution)
    # print("!!!!!!!!!!!!\n", absolute_error)
    absolute_error_mx = absolute_error.max()
    # print("!!!!!!!!!!!!\n", absolute_error_mx)

    measured_value = second_solution.max()

    if measured_value == 0:
        return f"{absolute_error}/{measured_value}"

    # print(f"{absolute_error}/{measured_value}")

    return float(absolute_error_mx) / measured_value


def print_solution_matrix(number: str, mtx: Matrix) -> None:
    print(f"{number} solution:")
    len_mtx = len(mtx)
    print(*[f"X{i+1} = {mtx[i][0]}" for i in range(len_mtx)], sep="\n")


def guassian_main():

    A = np.array(
        [[2.30, 3.50, 1.70], [5.70, -2.70, 2.30], [-0.80, 5.30, -1.80]], dtype=float
    )

    B = np.array(
        [
            -6.49,
            19.20,
            -5.09,
        ],
        dtype=float,
    )
    # print(B)
    augmented_matrix = np.column_stack((A, B))
    print(augmented_matrix)

    solution = gaussian(augmented_matrix)
    print(solution)

    print_solution_matrix("First", solution)

    # print("diagonal_view:", augmented_matrix)
    second_matrix_b = get_new_matrix_b(solution, A)

    second_augmented_matrix = np.column_stack((A, second_matrix_b))
    second_solution = gaussian(second_augmented_matrix)

    print_solution_matrix("Second", second_solution)

    residual_vector = get_residual_vector(B, second_matrix_b)
    print("вектор невязки (Вектор остатков)")
    print(*[float(i) for i in residual_vector], sep="\n")

    third_matrix_b = get_new_matrix_b(second_solution, A)
    third_augmented_matrix = np.column_stack((A, third_matrix_b))

    third_solution = gaussian(third_augmented_matrix)

    print_solution_matrix("Third", third_solution)

    relative_error = get_relative_error(second_solution, third_solution)

    print("Относительная погрешность", relative_error)
