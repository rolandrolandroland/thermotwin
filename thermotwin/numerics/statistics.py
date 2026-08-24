"""Small dependency-free statistical helpers."""

import math
from typing import Sequence


def interpolated_quantile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated empirical quantile."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a quantile of empty values")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


__all__ = ["interpolated_quantile"]
