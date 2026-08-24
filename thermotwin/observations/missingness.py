"""Deterministic missing-temperature records for virtual experiments."""

from dataclasses import dataclass, replace
import math
from typing import Optional, Tuple

from .metadata import (
    MetadataSetting,
    ObservationProcessStep,
    append_observation_process_step,
)

from .bias import FixedTemperatureBias
from .lag import (
    FirstOrderTemperatureLag,
    run_lagged_noisy_biased_contact_reference_test_stand,
)
from .noise import GaussianTemperatureNoise
from .test_stand import (
    ObservationDataset,
    TemperatureObservation,
    run_ideal_contact_reference_test_stand,
)


@dataclass(frozen=True)
class TemperatureSensorOutage:
    """One inclusive interval when one named sensor is unavailable."""

    sensor_name: str
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_name, str) or not self.sensor_name.strip():
            raise ValueError("outage sensor name must be nonempty")
        object.__setattr__(self, "sensor_name", self.sensor_name.strip())

        for field_name, value in (
            ("start time", self.start_time),
            ("end time", self.end_time),
        ):
            try:
                value_is_finite = math.isfinite(value)
            except TypeError as error:
                raise ValueError(
                    f"outage {field_name} must be finite and nonnegative"
                ) from error
            if not value_is_finite or value < 0.0:
                raise ValueError(
                    f"outage {field_name} must be finite and nonnegative"
                )

        if self.end_time < self.start_time:
            raise ValueError("outage end time must not precede start time")
        object.__setattr__(self, "start_time", float(self.start_time))
        object.__setattr__(self, "end_time", float(self.end_time))

    def includes(self, time: float) -> bool:
        """Return whether a time lies in this inclusive outage interval."""

        tolerance = 1e-12 * max(1.0, self.end_time)
        at_or_after_start = time > self.start_time or math.isclose(
            time,
            self.start_time,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        at_or_before_end = time < self.end_time or math.isclose(
            time,
            self.end_time,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        return at_or_after_start and at_or_before_end


@dataclass(frozen=True)
class DeterministicTemperatureMissingness:
    """Inclusive, named sensor-outage intervals."""

    outages: Tuple[TemperatureSensorOutage, ...] = ()

    def __post_init__(self) -> None:
        try:
            outages = tuple(self.outages)
        except TypeError as error:
            raise ValueError(
                "temperature outages must be sensor-outage objects"
            ) from error
        if not all(
            isinstance(outage, TemperatureSensorOutage)
            for outage in outages
        ):
            raise ValueError(
                "temperature outages must be sensor-outage objects"
            )

        for index, left in enumerate(outages):
            for right in outages[index + 1 :]:
                if left.sensor_name != right.sensor_name:
                    continue
                intervals_overlap = max(
                    left.start_time,
                    right.start_time,
                ) <= min(left.end_time, right.end_time)
                if intervals_overlap:
                    raise ValueError(
                        "outage intervals for one sensor must not overlap"
                    )
        object.__setattr__(self, "outages", outages)

    def removes(self, observation: TemperatureObservation) -> bool:
        """Return whether one observation is unavailable."""

        return any(
            outage.sensor_name == observation.sensor_name
            and outage.includes(observation.time)
            for outage in self.outages
        )


@dataclass(frozen=True)
class TemperatureMissingnessResult:
    """Incomplete observations and the configuration that removed records."""

    dataset: ObservationDataset
    missingness_model: DeterministicTemperatureMissingness


@dataclass(frozen=True)
class IncompleteTemperatureResult:
    """Incomplete lagged, biased, noisy observations with provenance."""

    dataset: ObservationDataset
    lag_model: FirstOrderTemperatureLag
    bias_model: FixedTemperatureBias
    noise_model: GaussianTemperatureNoise
    missingness_model: DeterministicTemperatureMissingness


def apply_deterministic_temperature_missingness(
    dataset: ObservationDataset,
    missingness_model: DeterministicTemperatureMissingness,
) -> TemperatureMissingnessResult:
    """Omit unavailable readings without changing retained observations."""

    sensor_names = {sensor.name for sensor in dataset.sensors}
    outage_sensor_names = {
        outage.sensor_name for outage in missingness_model.outages
    }
    unknown_names = outage_sensor_names - sensor_names
    if unknown_names:
        joined_names = ", ".join(sorted(unknown_names))
        raise ValueError(
            f"outages reference unknown sensors: {joined_names}"
        )

    retained_observations = tuple(
        observation
        for observation in dataset.observations
        if not missingness_model.removes(observation)
    )
    if not retained_observations:
        raise ValueError("missingness configuration removes every observation")

    step_settings = []
    for index, outage in enumerate(missingness_model.outages):
        step_settings.extend(
            (
                MetadataSetting(
                    f"outage_{index}_sensor",
                    outage.sensor_name,
                ),
                MetadataSetting(
                    f"outage_{index}_start_s",
                    outage.start_time,
                ),
                MetadataSetting(
                    f"outage_{index}_end_s",
                    outage.end_time,
                ),
            )
        )
    provenance = dataset.provenance
    if len(retained_observations) < len(dataset.observations):
        provenance = append_observation_process_step(
            provenance,
            ObservationProcessStep(
                "deterministic_temperature_missingness",
                tuple(step_settings),
            ),
        )
    incomplete_dataset = replace(
        dataset,
        observations=retained_observations,
        provenance=provenance,
    )
    return TemperatureMissingnessResult(
        dataset=incomplete_dataset,
        missingness_model=missingness_model,
    )


def reference_deterministic_temperature_missingness(
) -> DeterministicTemperatureMissingness:
    """Return the generic inclusive 20--30 s cold-face outage."""

    return DeterministicTemperatureMissingness(
        outages=(
            TemperatureSensorOutage(
                sensor_name="cold_face_sensor",
                start_time=20.0,
                end_time=30.0,
            ),
        )
    )


def run_missing_contact_reference_test_stand(
    *,
    sampling_interval: float = 1.0,
    missingness_model: Optional[
        DeterministicTemperatureMissingness
    ] = None,
) -> TemperatureMissingnessResult:
    """Apply deterministic missingness to ideal reference observations."""

    ideal_dataset = run_ideal_contact_reference_test_stand(
        sampling_interval=sampling_interval
    )
    selected_missingness_model = (
        reference_deterministic_temperature_missingness()
        if missingness_model is None
        else missingness_model
    )
    return apply_deterministic_temperature_missingness(
        ideal_dataset,
        selected_missingness_model,
    )


def run_incomplete_contact_reference_test_stand(
    *,
    sampling_interval: float = 1.0,
    lag_model: Optional[FirstOrderTemperatureLag] = None,
    bias_model: Optional[FixedTemperatureBias] = None,
    noise_model: Optional[GaussianTemperatureNoise] = None,
    missingness_model: Optional[
        DeterministicTemperatureMissingness
    ] = None,
) -> IncompleteTemperatureResult:
    """Apply lag, sampling, bias, noise, then deterministic missingness."""

    complete = run_lagged_noisy_biased_contact_reference_test_stand(
        sampling_interval=sampling_interval,
        lag_model=lag_model,
        bias_model=bias_model,
        noise_model=noise_model,
    )
    selected_missingness_model = (
        reference_deterministic_temperature_missingness()
        if missingness_model is None
        else missingness_model
    )
    incomplete = apply_deterministic_temperature_missingness(
        complete.dataset,
        selected_missingness_model,
    )
    return IncompleteTemperatureResult(
        dataset=incomplete.dataset,
        lag_model=complete.lag_model,
        bias_model=complete.bias_model,
        noise_model=complete.noise_model,
        missingness_model=selected_missingness_model,
    )
