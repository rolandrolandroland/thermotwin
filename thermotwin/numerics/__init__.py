"""Dependency-free numerical helpers shared across ThermoTwin layers."""

from .matrices import gram_matrix, inverse_and_determinant, matrix_multiply
from .integration import (
    IntegrationDivergenceError,
    first_rising_crossing_bracket,
    interpolated_value,
    linear_interpolation,
    trapezoidal_integral,
)
from .statistics import interpolated_quantile

__all__ = [
    "IntegrationDivergenceError",
    "first_rising_crossing_bracket",
    "gram_matrix",
    "interpolated_value",
    "inverse_and_determinant",
    "interpolated_quantile",
    "linear_interpolation",
    "matrix_multiply",
    "trapezoidal_integral",
]
