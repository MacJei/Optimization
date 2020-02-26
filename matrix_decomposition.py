""" Try to decompose a matrix
    We use qr because it decomposes a matrix independent of the shaper
    LR decomposition requires the matrix to be a square matrix
"""
import numpy as np
import scipy.linalg as la

A = [
        [0.99, 0.87, 0.51, 0.27],
        [0.34, 0.22, 0.28, 0.34],
        [0.86, 0.16, 0.14, 0.76],
        [0.74, 0.36, 0.34, 0.67]
    ]
A = np.array(A)
print(A)

row, col = la.qr(A)
print(row)
print(col)
print(row@col)