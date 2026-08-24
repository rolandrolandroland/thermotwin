"""Generic interpolation, integration, and scalar bracketing helpers."""

import math
from typing import Callable, Optional, Sequence, Tuple


class IntegrationDivergenceError(RuntimeError):
    """Raised when a numerical trajectory leaves the model's valid domain."""


def linear_interpolation(
    left_time: float,
    right_time: float,
    left_value: float,
    right_value: float,
    time: float,
) -> float:
    """Linearly interpolate one value, including a zero-width safe limit."""

    if right_time == left_time:
        return left_value
    fraction = (time - left_time) / (right_time - left_time)
    return left_value + fraction * (right_value - left_value)


def trapezoidal_integral(
    time: Sequence[float],
    values: Sequence[float],
    *,
    start_time: float,
    end_time: float,
) -> float:
    """Integrate a continuous sampled history over an exactly clipped interval."""

    if len(time) != len(values) or len(time) < 2:
        raise ValueError("time and values must have equal length of at least two")
    if end_time <= start_time:
        raise ValueError("integration end time must exceed start time")
    if start_time < time[0] or end_time > time[-1]:
        raise ValueError("integration interval lies outside sampled time")

    total = 0.0
    for left_time, right_time, left_value, right_value in zip(
        time, time[1:], values, values[1:]
    ):
        clipped_left = max(left_time, start_time)
        clipped_right = min(right_time, end_time)
        if clipped_right <= clipped_left:
            continue
        clipped_left_value = linear_interpolation(
            left_time,
            right_time,
            left_value,
            right_value,
            clipped_left,
        )
        clipped_right_value = linear_interpolation(
            left_time,
            right_time,
            left_value,
            right_value,
            clipped_right,
        )
        total += 0.5 * (clipped_left_value + clipped_right_value) * (
            clipped_right - clipped_left
        )
    return total


def interpolated_value(
    time: Sequence[float],
    values: Sequence[float],
    target_time: float,
) -> float:
    """Return a linearly interpolated value inside a sampled interval."""

    for left_time, right_time, left_value, right_value in zip(
        time, time[1:], values, values[1:]
    ):
        if left_time <= target_time <= right_time:
            return linear_interpolation(
                left_time,
                right_time,
                left_value,
                right_value,
                target_time,
            )
    if target_time == time[-1]:
        return values[-1]
    raise ValueError("target time lies outside sampled time")


def first_rising_crossing_bracket(
    response: Callable[[float], float],
    *,
    target: float,
    maximum_input: float,
    subdivisions: int,
) -> Optional[Tuple[float, float]]:
    """Bracket the first sampled below-to-above target crossing."""

    if not math.isfinite(target) or not math.isfinite(maximum_input):
        raise ValueError("bracketing target and maximum input must be finite")
    if maximum_input <= 0.0:
        raise ValueError("bracketing maximum input must be positive")
    if (
        isinstance(subdivisions, bool)
        or not isinstance(subdivisions, int)
        or subdivisions < 2
    ):
        raise ValueError("bracketing subdivisions must be an integer at least two")

    previous_input = 0.0
    previous_response = response(previous_input)
    if not math.isfinite(previous_response):
        raise ValueError("bracketing response must be finite")
    if previous_response >= target:
        return (0.0, 0.0)
    for index in range(1, subdivisions + 1):
        candidate_input = maximum_input * index / subdivisions
        candidate_response = response(candidate_input)
        if not math.isfinite(candidate_response):
            raise ValueError("bracketing response must be finite")
        if previous_response < target <= candidate_response:
            return previous_input, candidate_input
        previous_input = candidate_input
        previous_response = candidate_response
    return None


__all__ = [
    "IntegrationDivergenceError",
    "first_rising_crossing_bracket",
    "interpolated_value",
    "linear_interpolation",
    "trapezoidal_integral",
]
