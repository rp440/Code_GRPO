"""
Matrix Operations and Utilities

Contains matrix multiplication functions and matrix generation utilities.
"""

import random


def manual_matrix_multiply_2x2(A, B):
    """Manually compute 2x2 matrix multiplication C = A * B"""
    C = [[0, 0], [0, 0]]
    C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0]
    C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1]
    C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0]
    C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1]
    return C


def generate_random_2x2_matrix(low=-99, high=99):
    """Generate a random 2x2 matrix with integer values"""
    return [[random.randint(low, high) for _ in range(2)] for _ in range(2)]


def count_multiplications_in_dsl(dsl_script):
    """Count the number of multiplication operations in a DSL script"""
    import re
    num_multiplications = 0
    for line in dsl_script.split('\n'):
        if re.search(r"=\s*[\w\[\],\s\.\d\-]+\s*\*\s*[\w\[\],\s\.\d\-]+", line.strip()):
            num_multiplications += 1
    return num_multiplications 