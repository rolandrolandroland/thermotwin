"""Sparse virtual measurements from the distributed thermoelectric model."""

from dataclasses import dataclass
import math
import random
from typing import NamedTuple, Sequence, Tuple

from ..core.controls import current_at
from ..physics.distributed import evaluate_distributed_state
from ..simulation.distributed import (
    DistributedLegExperiment,
    DistributedTemperatureTrajectory,
    run_distributed_leg_experiment,
)


DISTRIBUTED_CHANNELS = (
    "cold_face_temperature",
    "hot_face_temperature",
    "voltage",
    "cold_side_heat",
    "hot_side_heat",
)


@dataclass(frozen=True)
class DistributedObservationChannels:
    """Select the terminal and face quantities exposed to inference."""

    cold_face_temperature: bool = True
    hot_face_temperature: bool = True
    voltage: bool = True
    cold_side_heat: bool = False
    hot_side_heat: bool = False

    def names(self) -> Tuple[str, ...]:
        selected = tuple(
            name for name in DISTRIBUTED_CHANNELS if getattr(self, name)
        )
        if not selected:
            raise ValueError("at least one distributed observation channel is required")
        return selected


class DistributedObservation(NamedTuple):
    time: float
    channel: str
    value: float


@dataclass(frozen=True)
class DistributedObservationSet:
    """Ordered sparse observations and the units implied by channel names."""

    observations: Tuple[DistributedObservation, ...]

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        object.__setattr__(self, "observations", observations)
        if not observations:
            raise ValueError("distributed observation set cannot be empty")
        for item in observations:
            if item.channel not in DISTRIBUTED_CHANNELS:
                raise ValueError(f"unknown distributed channel: {item.channel}")
            if not math.isfinite(item.time) or item.time < 0.0:
                raise ValueError("observation times must be finite and nonnegative")
            if not math.isfinite(item.value):
                raise ValueError("observation values must be finite")

    def values(self) -> Tuple[float, ...]:
        return tuple(item.value for item in self.observations)

    def keys(self) -> Tuple[Tuple[float, str], ...]:
        return tuple((item.time, item.channel) for item in self.observations)


def regular_distributed_observation_times(
    duration: float,
    interval: float,
) -> Tuple[float, ...]:
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("observation interval must be finite and positive")
    count = int(duration // interval)
    times = [index * interval for index in range(count + 1)]
    tolerance = 1e-12 * max(1.0, duration)
    if times[-1] < duration - tolerance:
        times.append(duration)
    else:
        times[-1] = duration
    return tuple(times)


def _interpolate_scalar(
    left_time: float,
    right_time: float,
    left_value: float,
    right_value: float,
    target_time: float,
) -> float:
    if target_time == left_time or right_time == left_time:
        return left_value
    if target_time == right_time:
        return right_value
    fraction = (target_time - left_time) / (right_time - left_time)
    return left_value + fraction * (right_value - left_value)


def interpolate_distributed_state(
    trajectory: DistributedTemperatureTrajectory,
    time: float,
) -> Tuple[float, Tuple[float, ...], float]:
    """Interpolate only continuous temperature states, never jump diagnostics."""

    if not math.isfinite(time) or time < trajectory.time[0] or time > trajectory.time[-1]:
        raise ValueError("interpolation time lies outside the trajectory")
    if time == trajectory.time[-1]:
        return trajectory.cold_face[-1], trajectory.cells[-1], trajectory.hot_face[-1]
    right_index = next(
        index
        for index, candidate in enumerate(trajectory.time)
        if candidate >= time
    )
    if trajectory.time[right_index] == time:
        return (
            trajectory.cold_face[right_index],
            trajectory.cells[right_index],
            trajectory.hot_face[right_index],
        )
    left_index = right_index - 1
    left_time = trajectory.time[left_index]
    right_time = trajectory.time[right_index]
    return (
        _interpolate_scalar(
            left_time,
            right_time,
            trajectory.cold_face[left_index],
            trajectory.cold_face[right_index],
            time,
        ),
        tuple(
            _interpolate_scalar(left_time, right_time, left, right, time)
            for left, right in zip(
                trajectory.cells[left_index], trajectory.cells[right_index]
            )
        ),
        _interpolate_scalar(
            left_time,
            right_time,
            trajectory.hot_face[left_index],
            trajectory.hot_face[right_index],
            time,
        ),
    )


def observe_distributed_trajectory(
    experiment: DistributedLegExperiment,
    trajectory: DistributedTemperatureTrajectory,
    *,
    times: Sequence[float],
    channels: DistributedObservationChannels = DistributedObservationChannels(),
) -> DistributedObservationSet:
    """Sample temperatures, then recompute discontinuous diagnostics exactly."""

    selected_channels = channels.names()
    observation_times = tuple(float(value) for value in times)
    if not observation_times:
        raise ValueError("at least one distributed observation time is required")
    if any(
        right <= left for left, right in zip(observation_times, observation_times[1:])
    ):
        raise ValueError("distributed observation times must strictly increase")
    observations = []
    for time in observation_times:
        cold, cells, hot = interpolate_distributed_state(trajectory, time)
        diagnostics = evaluate_distributed_state(
            experiment.material,
            experiment.geometry,
            experiment.face_parameters,
            cold_face_temperature=cold,
            cell_temperatures=cells,
            hot_face_temperature=hot,
            current=current_at(experiment.current, time),
            cold_reservoir_temperature=experiment.cold_reservoir_temperature,
            hot_reservoir_temperature=experiment.hot_reservoir_temperature,
            cold_external_heat=experiment.cold_external_heat,
            hot_external_heat=experiment.hot_external_heat,
        )
        values = {
            "cold_face_temperature": cold,
            "hot_face_temperature": hot,
            "voltage": diagnostics.voltage,
            "cold_side_heat": diagnostics.cold_side_heat,
            "hot_side_heat": diagnostics.hot_side_heat,
        }
        observations.extend(
            DistributedObservation(time, channel, values[channel])
            for channel in selected_channels
        )
    return DistributedObservationSet(tuple(observations))


def run_distributed_virtual_experiment(
    experiment: DistributedLegExperiment,
    *,
    observation_interval: float,
    channels: DistributedObservationChannels = DistributedObservationChannels(),
) -> DistributedObservationSet:
    result = run_distributed_leg_experiment(experiment)
    return observe_distributed_trajectory(
        experiment,
        result.trajectory,
        times=regular_distributed_observation_times(
            experiment.duration, observation_interval
        ),
        channels=channels,
    )


def add_distributed_gaussian_noise(
    observations: DistributedObservationSet,
    *,
    standard_deviations: dict[str, float],
    seed: int,
) -> DistributedObservationSet:
    """Add independent channel-scaled Gaussian noise reproducibly."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("noise seed must be an integer")
    for channel in {item.channel for item in observations.observations}:
        if channel not in standard_deviations:
            raise ValueError(f"missing noise standard deviation for {channel}")
        value = standard_deviations[channel]
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("noise standard deviations must be finite and nonnegative")
    source = random.Random(seed)
    return DistributedObservationSet(
        tuple(
            DistributedObservation(
                item.time,
                item.channel,
                item.value + source.gauss(0.0, standard_deviations[item.channel]),
            )
            for item in observations.observations
        )
    )
