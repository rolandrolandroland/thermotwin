"""First-order dynamic lag for virtual temperature sensors."""

from bisect import bisect_left
from dataclasses import dataclass, replace
import math
from typing import Optional, Tuple

from ..simulation.four_node_experiments import constant_current_contact_reference_experiment
from ..core.controls import CurrentInput, current_at
from .metadata import (
    MetadataSetting,
    ObservationProcessStep,
    append_observation_process_step,
)
from .bias import (
    FixedTemperatureBias,
    apply_fixed_temperature_bias,
    reference_fixed_temperature_bias,
)
from .noise import (
    GaussianTemperatureNoise,
    apply_gaussian_temperature_noise,
    reference_gaussian_temperature_noise,
)
from .test_stand import (
    ObservationDataset,
    TemperatureObservation,
    regular_measurement_times,
    run_ideal_contact_reference_test_stand,
)


@dataclass(frozen=True)
class FirstOrderTemperatureLag:
    """First-order sensor time constants in seconds."""

    default_time_constant: float = 0.0
    sensor_time_constants: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        try:
            default_time_constant_is_finite = math.isfinite(
                self.default_time_constant
            )
        except TypeError as error:
            raise ValueError(
                "default sensor time constant must be finite and nonnegative"
            ) from error
        if (
            not default_time_constant_is_finite
            or self.default_time_constant < 0.0
        ):
            raise ValueError(
                "default sensor time constant must be finite and nonnegative"
            )
        object.__setattr__(
            self,
            "default_time_constant",
            float(self.default_time_constant),
        )

        try:
            configured_time_constants = tuple(self.sensor_time_constants)
        except TypeError as error:
            raise ValueError(
                "sensor time constants must be name-value pairs"
            ) from error
        normalized_time_constants = []
        for item in configured_time_constants:
            try:
                sensor_name, time_constant = item
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "sensor time constants must be name-value pairs"
                ) from error
            if not isinstance(sensor_name, str) or not sensor_name.strip():
                raise ValueError("lag sensor name must be nonempty")
            try:
                time_constant_is_finite = math.isfinite(time_constant)
            except TypeError as error:
                raise ValueError(
                    "sensor time constants must be finite and nonnegative"
                ) from error
            if not time_constant_is_finite or time_constant < 0.0:
                raise ValueError(
                    "sensor time constants must be finite and nonnegative"
                )
            normalized_time_constants.append(
                (sensor_name.strip(), float(time_constant))
            )

        names = tuple(name for name, _ in normalized_time_constants)
        if len(set(names)) != len(names):
            raise ValueError("sensor time-constant names must be unique")
        object.__setattr__(
            self,
            "sensor_time_constants",
            tuple(normalized_time_constants),
        )

    def time_constant_for(self, sensor_name: str) -> float:
        """Return one sensor's time constant in seconds."""

        overrides = dict(self.sensor_time_constants)
        return overrides.get(sensor_name, self.default_time_constant)


@dataclass(frozen=True)
class TemperatureLagResult:
    """Lagged observations and the configuration that generated them."""

    dataset: ObservationDataset
    lag_model: FirstOrderTemperatureLag


@dataclass(frozen=True)
class LaggedNoisyBiasedTemperatureResult:
    """Lagged, biased, noisy observations with complete provenance."""

    dataset: ObservationDataset
    lag_model: FirstOrderTemperatureLag
    bias_model: FixedTemperatureBias
    noise_model: GaussianTemperatureNoise


def apply_first_order_temperature_lag(
    dataset: ObservationDataset,
    lag_model: FirstOrderTemperatureLag,
) -> TemperatureLagResult:
    """Filter sensor histories at their supplied times without mutation.

    Between adjacent observations, the input temperature is interpolated
    linearly and the first-order sensor ODE is integrated exactly for that
    piecewise-linear target. This avoids the half-sample lead introduced by
    treating the interval's right-end temperature as a constant target. The
    first reported value initializes the sensor state exactly.
    """

    sensor_names = {sensor.name for sensor in dataset.sensors}
    override_names = {
        name for name, _ in lag_model.sensor_time_constants
    }
    unknown_names = override_names - sensor_names
    if unknown_names:
        joined_names = ", ".join(sorted(unknown_names))
        raise ValueError(
            f"lag overrides reference unknown sensors: {joined_names}"
        )

    previous_time = {}
    previous_input_temperature = {}
    previous_filtered_temperature = {}
    lagged_observations = []
    for observation in dataset.observations:
        sensor_name = observation.sensor_name
        time_constant = lag_model.time_constant_for(sensor_name)
        if sensor_name not in previous_time or time_constant == 0.0:
            filtered_temperature = observation.temperature
        else:
            time_step = observation.time - previous_time[sensor_name]
            if time_step <= 0.0:
                raise ValueError(
                    "each sensor's observation times must strictly increase"
                )
            decay = math.exp(-time_step / time_constant)
            previous_target = previous_input_temperature[sensor_name]
            target_slope = (
                observation.temperature - previous_target
            ) / time_step
            filtered_temperature = (
                decay * previous_filtered_temperature[sensor_name]
                + (1.0 - decay) * previous_target
                + target_slope
                * (
                    time_step
                    - time_constant * (1.0 - decay)
                )
            )
        previous_time[sensor_name] = observation.time
        previous_input_temperature[sensor_name] = observation.temperature
        previous_filtered_temperature[sensor_name] = filtered_temperature
        lagged_observations.append(
            replace(
                observation,
                temperature=filtered_temperature,
            )
        )

    step_settings = [
        MetadataSetting(
            "default_time_constant_s",
            lag_model.default_time_constant,
        ),
    ]
    step_settings.extend(
        MetadataSetting(f"sensor_time_constant_s:{sensor_name}", value)
        for sensor_name, value in lag_model.sensor_time_constants
    )
    lag_changes_values = any(
        lag_model.time_constant_for(sensor.name) > 0.0
        for sensor in dataset.sensors
    )
    provenance = dataset.provenance
    if lag_changes_values:
        provenance = append_observation_process_step(
            provenance,
            ObservationProcessStep(
                "first_order_temperature_lag",
                tuple(step_settings),
            ),
        )
    lagged_dataset = replace(
        dataset,
        observations=tuple(lagged_observations),
        provenance=provenance,
    )
    return TemperatureLagResult(
        dataset=lagged_dataset,
        lag_model=lag_model,
    )


def _interpolate_sensor_history(
    time: float,
    sensor_observations: Tuple[TemperatureObservation, ...],
) -> float:
    times = tuple(observation.time for observation in sensor_observations)
    temperatures = tuple(
        observation.temperature for observation in sensor_observations
    )
    tolerance = 1e-12 * max(1.0, times[-1])
    index = bisect_left(times, time)
    if index < len(times) and math.isclose(
        times[index],
        time,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        return temperatures[index]
    if index > 0 and math.isclose(
        times[index - 1],
        time,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        return temperatures[index - 1]
    if index == 0 or index == len(times):
        raise ValueError("requested lagged measurement cannot be bracketed")

    left_time = times[index - 1]
    right_time = times[index]
    fraction = (time - left_time) / (right_time - left_time)
    return temperatures[index - 1] + fraction * (
        temperatures[index] - temperatures[index - 1]
    )


def _resample_lagged_dataset(
    dense_dataset: ObservationDataset,
    *,
    sampling_interval: float,
    current: CurrentInput,
) -> ObservationDataset:
    measurement_times = regular_measurement_times(
        dense_dataset.measurement_times[-1],
        sampling_interval,
    )
    histories = {
        sensor.name: dense_dataset.observations_for(sensor.name)
        for sensor in dense_dataset.sensors
    }
    observations = []
    for time in measurement_times:
        sample_current = current_at(current, time)
        for sensor in dense_dataset.sensors:
            observations.append(
                TemperatureObservation(
                    time=time,
                    sensor_name=sensor.name,
                    location=sensor.location,
                    temperature=_interpolate_sensor_history(
                        time,
                        histories[sensor.name],
                    ),
                    current=sample_current,
                )
            )
    return ObservationDataset(
        observations=tuple(observations),
        sensors=dense_dataset.sensors,
        sampling_interval=sampling_interval,
        time_unit=dense_dataset.time_unit,
        temperature_unit=dense_dataset.temperature_unit,
        current_unit=dense_dataset.current_unit,
        provenance=append_observation_process_step(
            dense_dataset.provenance,
            ObservationProcessStep(
                "output_sampling",
                (
                    MetadataSetting(
                        "sampling_interval_s",
                        sampling_interval,
                    ),
                ),
            ),
        ),
    )


def reference_first_order_temperature_lag() -> FirstOrderTemperatureLag:
    """Return the generic 2 s cold-face-only lag learning baseline."""

    return FirstOrderTemperatureLag(
        sensor_time_constants=(("cold_face_sensor", 2.0),),
    )


def run_lagged_contact_reference_test_stand(
    *,
    sampling_interval: float = 1.0,
    lag_model: Optional[FirstOrderTemperatureLag] = None,
) -> TemperatureLagResult:
    """Filter dense reference truth before sampling lagged observations."""

    experiment = constant_current_contact_reference_experiment()
    dense_ideal_dataset = run_ideal_contact_reference_test_stand(
        sampling_interval=experiment.time_step
    )
    selected_lag_model = (
        reference_first_order_temperature_lag()
        if lag_model is None
        else lag_model
    )
    dense_lagged = apply_first_order_temperature_lag(
        dense_ideal_dataset,
        selected_lag_model,
    )
    sampled_lagged = _resample_lagged_dataset(
        dense_lagged.dataset,
        sampling_interval=sampling_interval,
        current=experiment.current,
    )
    return TemperatureLagResult(
        dataset=sampled_lagged,
        lag_model=selected_lag_model,
    )


def run_lagged_noisy_biased_contact_reference_test_stand(
    *,
    sampling_interval: float = 1.0,
    lag_model: Optional[FirstOrderTemperatureLag] = None,
    bias_model: Optional[FixedTemperatureBias] = None,
    noise_model: Optional[GaussianTemperatureNoise] = None,
) -> LaggedNoisyBiasedTemperatureResult:
    """Apply lag before fixed bias and Gaussian noise."""

    lagged = run_lagged_contact_reference_test_stand(
        sampling_interval=sampling_interval,
        lag_model=lag_model,
    )
    selected_bias_model = (
        reference_fixed_temperature_bias()
        if bias_model is None
        else bias_model
    )
    selected_noise_model = (
        reference_gaussian_temperature_noise()
        if noise_model is None
        else noise_model
    )
    biased = apply_fixed_temperature_bias(
        lagged.dataset,
        selected_bias_model,
    )
    noisy = apply_gaussian_temperature_noise(
        biased.dataset,
        selected_noise_model,
    )
    return LaggedNoisyBiasedTemperatureResult(
        dataset=noisy.dataset,
        lag_model=lagged.lag_model,
        bias_model=selected_bias_model,
        noise_model=selected_noise_model,
    )
