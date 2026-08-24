"""Fixed per-sensor temperature bias for virtual observations."""

from dataclasses import dataclass, replace
import math
from typing import Optional, Tuple

from .metadata import (
    MetadataSetting,
    ObservationProcessStep,
    append_observation_process_step,
)

from .noise import (
    GaussianTemperatureNoise,
    apply_gaussian_temperature_noise,
    reference_gaussian_temperature_noise,
)
from .test_stand import (
    ObservationDataset,
    run_ideal_contact_reference_test_stand,
)


@dataclass(frozen=True)
class FixedTemperatureBias:
    """Constant additive temperature offsets in kelvin."""

    default_bias: float = 0.0
    sensor_biases: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.default_bias):
            raise ValueError("default temperature bias must be finite")

        normalized_biases = []
        for item in self.sensor_biases:
            if len(item) != 2:
                raise ValueError("sensor biases must be name-value pairs")
            sensor_name, bias = item
            if not isinstance(sensor_name, str) or not sensor_name.strip():
                raise ValueError("bias sensor name must be nonempty")
            if not math.isfinite(bias):
                raise ValueError("sensor temperature biases must be finite")
            normalized_biases.append((sensor_name.strip(), float(bias)))

        names = tuple(name for name, _ in normalized_biases)
        if len(set(names)) != len(names):
            raise ValueError("sensor bias names must be unique")
        object.__setattr__(
            self,
            "sensor_biases",
            tuple(normalized_biases),
        )

    def bias_for(self, sensor_name: str) -> float:
        """Return the configured additive bias for one sensor."""

        overrides = dict(self.sensor_biases)
        return overrides.get(sensor_name, self.default_bias)


@dataclass(frozen=True)
class TemperatureBiasResult:
    """Biased observations and the configuration that generated them."""

    dataset: ObservationDataset
    bias_model: FixedTemperatureBias


@dataclass(frozen=True)
class NoisyBiasedTemperatureResult:
    """Observations with both configurations retained for provenance."""

    dataset: ObservationDataset
    noise_model: GaussianTemperatureNoise
    bias_model: FixedTemperatureBias


def apply_fixed_temperature_bias(
    dataset: ObservationDataset,
    bias_model: FixedTemperatureBias,
) -> TemperatureBiasResult:
    """Return a biased dataset without modifying the input dataset."""

    sensor_names = {sensor.name for sensor in dataset.sensors}
    override_names = {name for name, _ in bias_model.sensor_biases}
    unknown_names = override_names - sensor_names
    if unknown_names:
        joined_names = ", ".join(sorted(unknown_names))
        raise ValueError(
            f"bias overrides reference unknown sensors: {joined_names}"
        )

    biased_observations = tuple(
        replace(
            observation,
            temperature=(
                observation.temperature
                + bias_model.bias_for(observation.sensor_name)
            ),
        )
        for observation in dataset.observations
    )
    step_settings = [
        MetadataSetting("default_bias_K", bias_model.default_bias),
    ]
    step_settings.extend(
        MetadataSetting(f"sensor_bias_K:{sensor_name}", bias)
        for sensor_name, bias in bias_model.sensor_biases
    )
    bias_changes_values = any(
        bias_model.bias_for(sensor.name) != 0.0
        for sensor in dataset.sensors
    )
    provenance = dataset.provenance
    if bias_changes_values:
        provenance = append_observation_process_step(
            provenance,
            ObservationProcessStep(
                "fixed_temperature_bias",
                tuple(step_settings),
            ),
        )
    biased_dataset = replace(
        dataset,
        observations=biased_observations,
        provenance=provenance,
    )
    return TemperatureBiasResult(
        dataset=biased_dataset,
        bias_model=bias_model,
    )


def reference_fixed_temperature_bias() -> FixedTemperatureBias:
    """Return the generic +0.10 K cold-face bias learning baseline."""

    return FixedTemperatureBias(
        sensor_biases=(("cold_face_sensor", 0.10),),
    )


def run_biased_contact_reference_test_stand(
    *,
    sampling_interval: float = 1.0,
    bias_model: Optional[FixedTemperatureBias] = None,
) -> TemperatureBiasResult:
    """Return bias-only observations from the contact reference truth."""

    ideal_dataset = run_ideal_contact_reference_test_stand(
        sampling_interval=sampling_interval
    )
    selected_bias_model = (
        reference_fixed_temperature_bias()
        if bias_model is None
        else bias_model
    )
    return apply_fixed_temperature_bias(
        ideal_dataset,
        selected_bias_model,
    )


def run_noisy_biased_contact_reference_test_stand(
    *,
    sampling_interval: float = 1.0,
    noise_model: Optional[GaussianTemperatureNoise] = None,
    bias_model: Optional[FixedTemperatureBias] = None,
) -> NoisyBiasedTemperatureResult:
    """Return observations with noise followed by fixed sensor bias."""

    ideal_dataset = run_ideal_contact_reference_test_stand(
        sampling_interval=sampling_interval
    )
    selected_noise_model = (
        reference_gaussian_temperature_noise()
        if noise_model is None
        else noise_model
    )
    selected_bias_model = (
        reference_fixed_temperature_bias()
        if bias_model is None
        else bias_model
    )
    noisy = apply_gaussian_temperature_noise(
        ideal_dataset,
        selected_noise_model,
    )
    biased = apply_fixed_temperature_bias(
        noisy.dataset,
        selected_bias_model,
    )
    return NoisyBiasedTemperatureResult(
        dataset=biased.dataset,
        noise_model=selected_noise_model,
        bias_model=selected_bias_model,
    )
