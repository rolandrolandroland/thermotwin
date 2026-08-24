"""Ideal sensor observations sampled from contact-model synthetic truth."""

from bisect import bisect_left
from dataclasses import dataclass
from enum import Enum
import math
from typing import Tuple

from ..simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
    run_four_node_contact_experiment,
)
from ..physics.four_node import FourNodeContactTemperatureTrajectory
from ..core.controls import CurrentInput, current_at
from .metadata import (
    ContactExperimentMetadata,
    DatasetProvenance,
    MetadataSetting,
    ObservationProcessStep,
)


class TemperatureSensorLocation(str, Enum):
    """Thermal node observed by an ideal temperature sensor."""

    COLD_FACE = "cold_face"
    HOT_FACE = "hot_face"
    COLD_EXCHANGER = "cold_exchanger"
    HOT_EXCHANGER = "hot_exchanger"


@dataclass(frozen=True)
class IdealTemperatureSensor:
    """A named ideal sensor attached to one modeled thermal node."""

    name: str
    location: TemperatureSensorLocation

    def __post_init__(self) -> None:
        clean_name = self.name.strip()
        if not clean_name:
            raise ValueError("sensor name must be nonempty")
        object.__setattr__(self, "name", clean_name)
        try:
            location = TemperatureSensorLocation(self.location)
        except ValueError as error:
            raise ValueError("sensor location is not supported") from error
        object.__setattr__(self, "location", location)


@dataclass(frozen=True)
class IdealVirtualTestStand:
    """Sensor configuration and regular sampling interval."""

    sensors: Tuple[IdealTemperatureSensor, ...]
    sampling_interval: float

    def __post_init__(self) -> None:
        sensors = tuple(self.sensors)
        object.__setattr__(self, "sensors", sensors)
        if not sensors:
            raise ValueError("at least one sensor is required")
        names = tuple(sensor.name for sensor in sensors)
        if len(set(names)) != len(names):
            raise ValueError("sensor names must be unique")
        if (
            not math.isfinite(self.sampling_interval)
            or self.sampling_interval <= 0.0
        ):
            raise ValueError("sampling interval must be finite and positive")


@dataclass(frozen=True)
class TemperatureObservation:
    """One ideal temperature reading and its aligned current input."""

    time: float
    sensor_name: str
    location: TemperatureSensorLocation
    temperature: float
    current: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.time) or self.time < 0.0:
            raise ValueError("observation time must be finite and nonnegative")
        clean_name = self.sensor_name.strip()
        if not clean_name:
            raise ValueError("observation sensor name must be nonempty")
        object.__setattr__(self, "sensor_name", clean_name)
        try:
            location = TemperatureSensorLocation(self.location)
        except ValueError as error:
            raise ValueError("observation location is not supported") from error
        object.__setattr__(self, "location", location)
        if not math.isfinite(self.temperature):
            raise ValueError("observed temperature must be finite")
        if not math.isfinite(self.current):
            raise ValueError("observed current must be finite")


@dataclass(frozen=True)
class ObservationDataset:
    """Long-form ideal observations with explicit schema metadata."""

    observations: Tuple[TemperatureObservation, ...]
    sensors: Tuple[IdealTemperatureSensor, ...]
    sampling_interval: float
    time_unit: str = "s"
    temperature_unit: str = "K"
    current_unit: str = "A"
    provenance: DatasetProvenance | None = None

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        sensors = tuple(self.sensors)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "sensors", sensors)
        if not observations:
            raise ValueError("observation dataset must be nonempty")
        if not sensors:
            raise ValueError("observation dataset must define its sensors")
        sensor_names = tuple(sensor.name for sensor in sensors)
        if len(set(sensor_names)) != len(sensor_names):
            raise ValueError("observation dataset sensor names must be unique")
        if (
            not math.isfinite(self.sampling_interval)
            or self.sampling_interval <= 0.0
        ):
            raise ValueError("sampling interval must be finite and positive")
        if not all(
            unit.strip()
            for unit in (
                self.time_unit,
                self.temperature_unit,
                self.current_unit,
            )
        ):
            raise ValueError("observation units must be nonempty")
        if self.provenance is not None and not isinstance(
            self.provenance,
            DatasetProvenance,
        ):
            raise ValueError("dataset provenance must be valid metadata")

        sensor_by_name = {sensor.name: sensor for sensor in sensors}
        known_names = set(sensor_by_name)
        observed_names = {
            observation.sensor_name for observation in observations
        }
        if not observed_names.issubset(known_names):
            raise ValueError("every observation must reference a known sensor")
        if any(
            observation.location
            is not sensor_by_name[observation.sensor_name].location
            for observation in observations
        ):
            raise ValueError(
                "observation locations must match their sensor definitions"
            )
        if any(
            later.time < earlier.time
            for earlier, later in zip(observations, observations[1:])
        ):
            raise ValueError("observations must be ordered by time")
        keys = tuple(
            (observation.time, observation.sensor_name)
            for observation in observations
        )
        if len(set(keys)) != len(keys):
            raise ValueError(
                "a sensor may have at most one observation at each time"
            )

    @property
    def measurement_times(self) -> Tuple[float, ...]:
        """Return ordered unique measurement times."""

        times = []
        for observation in self.observations:
            if not times or observation.time != times[-1]:
                times.append(observation.time)
        return tuple(times)

    def observations_for(
        self,
        sensor_name: str,
    ) -> Tuple[TemperatureObservation, ...]:
        """Return all readings for one known sensor in time order."""

        known_names = {sensor.name for sensor in self.sensors}
        if sensor_name not in known_names:
            raise ValueError(f"unknown sensor name: {sensor_name}")
        return tuple(
            observation
            for observation in self.observations
            if observation.sensor_name == sensor_name
        )


def regular_measurement_times(
    duration: float,
    sampling_interval: float,
) -> Tuple[float, ...]:
    """Return regular times from zero, always including the exact duration."""

    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration must be finite and nonnegative")
    if not math.isfinite(sampling_interval) or sampling_interval <= 0.0:
        raise ValueError("sampling interval must be finite and positive")
    if duration == 0.0:
        return (0.0,)

    tolerance = 1e-12 * max(1.0, duration)
    times = [0.0]
    index = 1
    while True:
        candidate = index * sampling_interval
        if candidate >= duration or math.isclose(
            candidate,
            duration,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            break
        times.append(candidate)
        index += 1
    times.append(duration)
    return tuple(times)


def _validate_trajectory(
    trajectory: FourNodeContactTemperatureTrajectory,
) -> None:
    sample_count = len(trajectory.time)
    histories = (
        trajectory.cold_face,
        trajectory.hot_face,
        trajectory.cold_exchanger,
        trajectory.hot_exchanger,
    )
    if sample_count == 0:
        raise ValueError("trajectory must be nonempty")
    if any(len(history) != sample_count for history in histories):
        raise ValueError(
            "time and all four temperature histories must have equal lengths"
        )
    if any(not math.isfinite(time) for time in trajectory.time):
        raise ValueError("trajectory times must be finite")
    if trajectory.time[0] != 0.0:
        raise ValueError("trajectory must begin at time zero")
    if any(
        later <= earlier
        for earlier, later in zip(trajectory.time, trajectory.time[1:])
    ):
        raise ValueError("trajectory times must strictly increase")
    if any(
        not math.isfinite(temperature)
        for history in histories
        for temperature in history
    ):
        raise ValueError("trajectory temperatures must be finite")


def _interpolate_history(
    time: float,
    trajectory_times: Tuple[float, ...],
    history: Tuple[float, ...],
) -> float:
    """Linearly interpolate one trajectory history at one valid time."""

    tolerance = 1e-12 * max(1.0, trajectory_times[-1])
    if time < trajectory_times[0] - tolerance or time > trajectory_times[-1] + tolerance:
        raise ValueError("measurement time lies outside the trajectory")

    index = bisect_left(trajectory_times, time)
    if index < len(trajectory_times) and math.isclose(
        trajectory_times[index],
        time,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        return history[index]
    if index > 0 and math.isclose(
        trajectory_times[index - 1],
        time,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        return history[index - 1]
    if index == 0 or index == len(trajectory_times):
        raise ValueError("measurement time cannot be bracketed")

    left_time = trajectory_times[index - 1]
    right_time = trajectory_times[index]
    fraction = (time - left_time) / (right_time - left_time)
    return history[index - 1] + fraction * (
        history[index] - history[index - 1]
    )


def observe_contact_trajectory(
    trajectory: FourNodeContactTemperatureTrajectory,
    *,
    current: CurrentInput,
    test_stand: IdealVirtualTestStand,
    experiment_metadata: ContactExperimentMetadata | None = None,
) -> ObservationDataset:
    """Sample ideal long-form sensor observations from hidden trajectory truth."""

    _validate_trajectory(trajectory)
    measurement_times = regular_measurement_times(
        trajectory.time[-1],
        test_stand.sampling_interval,
    )
    histories = {
        TemperatureSensorLocation.COLD_FACE: trajectory.cold_face,
        TemperatureSensorLocation.HOT_FACE: trajectory.hot_face,
        TemperatureSensorLocation.COLD_EXCHANGER: trajectory.cold_exchanger,
        TemperatureSensorLocation.HOT_EXCHANGER: trajectory.hot_exchanger,
    }

    observations = []
    for time in measurement_times:
        sample_current = current_at(current, time)
        for sensor in test_stand.sensors:
            observations.append(
                TemperatureObservation(
                    time=time,
                    sensor_name=sensor.name,
                    location=sensor.location,
                    temperature=_interpolate_history(
                        time,
                        trajectory.time,
                        histories[sensor.location],
                    ),
                    current=sample_current,
                )
            )

    provenance = None
    if experiment_metadata is not None:
        provenance = DatasetProvenance(
            experiment=experiment_metadata,
            observation_steps=(
                ObservationProcessStep(
                    name="ideal_sampling",
                    settings=(
                        MetadataSetting(
                            "source_integration_time_step_s",
                            experiment_metadata.integration_time_step,
                        ),
                        MetadataSetting(
                            "output_sampling_interval_s",
                            test_stand.sampling_interval,
                        ),
                    ),
                ),
            ),
        )

    return ObservationDataset(
        observations=tuple(observations),
        sensors=test_stand.sensors,
        sampling_interval=test_stand.sampling_interval,
        provenance=provenance,
    )


def ideal_four_sensor_test_stand(
    *,
    sampling_interval: float = 1.0,
) -> IdealVirtualTestStand:
    """Return the ideal baseline with one sensor at each thermal node."""

    return IdealVirtualTestStand(
        sensors=(
            IdealTemperatureSensor(
                "cold_face_sensor",
                TemperatureSensorLocation.COLD_FACE,
            ),
            IdealTemperatureSensor(
                "hot_face_sensor",
                TemperatureSensorLocation.HOT_FACE,
            ),
            IdealTemperatureSensor(
                "cold_exchanger_sensor",
                TemperatureSensorLocation.COLD_EXCHANGER,
            ),
            IdealTemperatureSensor(
                "hot_exchanger_sensor",
                TemperatureSensorLocation.HOT_EXCHANGER,
            ),
        ),
        sampling_interval=sampling_interval,
    )


def run_ideal_contact_reference_test_stand(
    *,
    sampling_interval: float = 1.0,
) -> ObservationDataset:
    """Run hidden contact truth and return only its ideal observations."""

    experiment = constant_current_contact_reference_experiment()
    truth = run_four_node_contact_experiment(experiment).trajectory
    return observe_contact_trajectory(
        truth,
        current=experiment.current,
        test_stand=ideal_four_sensor_test_stand(
            sampling_interval=sampling_interval
        ),
        experiment_metadata=ContactExperimentMetadata.from_experiment(
            experiment,
            experiment_name="contact_reference",
            regime_name="constant_current_reference",
        ),
    )
