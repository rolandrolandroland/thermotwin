"""Dependency-free linear algebra for small information matrices."""

import math
from typing import Sequence, Tuple


Matrix = Tuple[Tuple[float, ...], ...]


def _validated_matrix(matrix: Sequence[Sequence[float]]) -> Matrix:
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    if not rows or not rows[0]:
        raise ValueError("matrix must be nonempty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("matrix rows must have equal lengths")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("matrix entries must be finite")
    return rows


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    rows = _validated_matrix(matrix)
    return tuple(tuple(row[index] for row in rows) for index in range(len(rows[0])))


def matrix_multiply(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> Matrix:
    left_rows = _validated_matrix(left)
    right_rows = _validated_matrix(right)
    if len(left_rows[0]) != len(right_rows):
        raise ValueError("matrix inner dimensions must agree")
    right_columns = transpose(right_rows)
    return tuple(
        tuple(
            sum(a * b for a, b in zip(left_row, right_column))
            for right_column in right_columns
        )
        for left_row in left_rows
    )


def matrix_add(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> Matrix:
    left_rows = _validated_matrix(left)
    right_rows = _validated_matrix(right)
    if (
        len(left_rows) != len(right_rows)
        or len(left_rows[0]) != len(right_rows[0])
    ):
        raise ValueError("matrix dimensions must agree")
    return tuple(
        tuple(a + b for a, b in zip(left_row, right_row))
        for left_row, right_row in zip(left_rows, right_rows)
    )


def inverse_and_determinant(
    matrix: Sequence[Sequence[float]],
) -> Tuple[Matrix, float]:
    """Return a square matrix inverse and determinant using pivoted elimination."""

    rows = _validated_matrix(matrix)
    size = len(rows)
    if len(rows[0]) != size:
        raise ValueError("matrix must be square")
    augmented = [
        list(row) + [1.0 if row_index == column else 0.0 for column in range(size)]
        for row_index, row in enumerate(rows)
    ]
    determinant = 1.0
    sign = 1.0
    scale = max(abs(value) for row in rows for value in row)
    tolerance = max(1e-15, 1e-12 * scale)

    for column in range(size):
        pivot_row = max(
            range(column, size),
            key=lambda index: abs(augmented[index][column]),
        )
        pivot = augmented[pivot_row][column]
        if abs(pivot) <= tolerance:
            raise ValueError("matrix is singular or numerically rank deficient")
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = (
                augmented[pivot_row], augmented[column]
            )
            sign *= -1.0
        pivot = augmented[column][column]
        determinant *= pivot
        augmented[column] = [value / pivot for value in augmented[column]]
        for row_index in range(size):
            if row_index == column:
                continue
            factor = augmented[row_index][column]
            if factor == 0.0:
                continue
            augmented[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row_index], augmented[column]
                )
            ]
    inverse = tuple(
        tuple(row[size:]) for row in augmented
    )
    return inverse, sign * determinant


def gram_matrix(jacobian: Sequence[Sequence[float]]) -> Matrix:
    rows = _validated_matrix(jacobian)
    return matrix_multiply(transpose(rows), rows)
