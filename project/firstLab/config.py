import numpy as np

LEN_MATRIX = 3

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


A = [[5, 2, 3], [2, 6, 1], [3, 1, 7]]


B = [10, 20, 30]


augmented_matrix = np.column_stack((A, B))
len_aug_mtx = len(augmented_matrix)


class Matrix:
    pass
