"""Time-dependent current inputs shared by ThermoTwin model layers."""

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class PiecewiseConstantCurrent:
    """A right-continuous current schedule with known switching times.

    ``values`` must contain one more entry than ``transition_times``. The first
    value applies before the first transition; at a transition, the value to
    its right applies immediately.
    """

    transition_times: Tuple[float, ...]
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        transition_times = tuple(self.transition_times)
        values = tuple(self.values)
        object.__setattr__(self, "transition_times", transition_times)
        object.__setattr__(self, "values", values)

        if len(values) != len(transition_times) + 1:
            raise ValueError(
                "current values must contain one more entry than transitions"
            )
        if any(
            not math.isfinite(time) or time < 0.0
            for time in transition_times
        ):
            raise ValueError(
                "current transition times must be finite and nonnegative"
            )
        if any(
            later <= earlier
            for earlier, later in zip(
                transition_times, transition_times[1:]
            )
        ):
            raise ValueError("current transition times must strictly increase")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("current values must be finite")

    @classmethod
    def constant(cls, current: float) -> "PiecewiseConstantCurrent":
        """Return a schedule that holds one current for the entire run."""

        return cls(transition_times=(), values=(current,))

    @classmethod
    def step(
        cls,
        *,
        transition_time: float,
        before_current: float,
        after_current: float,
    ) -> "PiecewiseConstantCurrent":
        """Return one current before a time and another from that time onward."""

        return cls(
            transition_times=(transition_time,),
            values=(before_current, after_current),
        )

    @classmethod
    def pulse(
        cls,
        *,
        start_time: float,
        end_time: float,
        pulse_current: float,
        baseline_current: float = 0.0,
    ) -> "PiecewiseConstantCurrent":
        """Return a baseline, one rectangular current pulse, then baseline."""

        return cls(
            transition_times=(start_time, end_time),
            values=(baseline_current, pulse_current, baseline_current),
        )

    @classmethod
    def periodic_pulse(
        cls,
        *,
        duration: float,
        period: float,
        duty_cycle: float,
        pulse_current: float,
        baseline_current: float = 0.0,
    ) -> "PiecewiseConstantCurrent":
        """Return a repeating pulse that begins in its on state.

        Transitions at or beyond ``duration`` are omitted. This keeps the
        finite-horizon schedule compact and makes its last sampled value
        unambiguous under the package's right-continuous convention.
        """

        for name, value in (
            ("duration", duration),
            ("period", period),
            ("duty cycle", duty_cycle),
            ("pulse current", pulse_current),
            ("baseline current", baseline_current),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if duration < 0.0:
            raise ValueError("duration must be nonnegative")
        if period <= 0.0:
            raise ValueError("period must be positive")
        if duty_cycle <= 0.0 or duty_cycle >= 1.0:
            raise ValueError(
                "duty cycle must lie strictly between zero and one"
            )

        on_duration = period * duty_cycle
        off_duration = period - on_duration
        transition_times = []
        values = [pulse_current]
        transition_time = on_duration
        entering_on_state = False
        tolerance = 1e-12 * max(1.0, duration)
        while transition_time < duration - tolerance:
            transition_times.append(transition_time)
            values.append(
                pulse_current if entering_on_state else baseline_current
            )
            transition_time += (
                on_duration if entering_on_state else off_duration
            )
            entering_on_state = not entering_on_state
        return cls(tuple(transition_times), tuple(values))

    def value_at(self, time: float) -> float:
        """Return the right-continuous current value at ``time`` in seconds."""

        if not math.isfinite(time):
            raise ValueError("current query time must be finite")
        return self.values[bisect_right(self.transition_times, time)]

    def next_transition_after(self, time: float) -> Optional[float]:
        """Return the first transition strictly after ``time``, if one exists."""

        if not math.isfinite(time):
            raise ValueError("current query time must be finite")
        index = bisect_right(self.transition_times, time)
        if index == len(self.transition_times):
            return None
        return self.transition_times[index]


CurrentInput = Union[float, PiecewiseConstantCurrent]


def current_at(current: CurrentInput, time: float) -> float:
    """Resolve either a scalar or scheduled current at one time."""

    if isinstance(current, PiecewiseConstantCurrent):
        return current.value_at(time)
    value = float(current)
    if not math.isfinite(value):
        raise ValueError("current must be finite")
    return value
