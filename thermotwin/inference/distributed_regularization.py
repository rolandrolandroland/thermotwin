"""Shared coefficient regularization for distributed inverse estimators."""

from typing import Sequence, TypeVar


Scalar = TypeVar("Scalar")


def second_difference_roughness(values: Sequence[Scalar]):
    """Return the mean squared second difference of an ordered coefficient set.

    The implementation deliberately works for both Python scalars and scalar
    PyTorch tensors without importing PyTorch into the conventional inference
    layer.  Using this one function in both estimators prevents their explicit
    smoothness priors from silently drifting apart.
    """

    if len(values) < 3:
        if len(values) == 0:
            return 0.0
        return values[0] * 0.0
    differences = tuple(
        values[index + 2] - 2.0 * values[index + 1] + values[index]
        for index in range(len(values) - 2)
    )
    return sum(value * value for value in differences) / len(differences)
