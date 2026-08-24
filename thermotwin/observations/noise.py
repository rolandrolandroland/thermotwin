"""Reproducible Gaussian temperature noise for virtual observations."""

from dataclasses import dataclass, replace
import math
import random
from typing import Optional, Tuple

from .metadata import (
    MetadataSetting,
    ObservationProcessStep,
    append_observation_process_step,
)

from .test_stand import (
    ObservationDataset,
    run_ideal_contact_reference_test_stand,
)


@dataclass(frozen=True)
class GaussianTemperatureNoise:
    """Independent zero-mean Gaussian temperature-noise configuration."""

    default_standard_deviation: float = 0.05
    random_seed: int = 2026
    sensor_standard_deviations: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.default_standard_deviation)
            or self.default_standard_deviation < 0.0
        ):
            raise ValueError(
                "default noise standard deviation must be finite and "
                "nonnegative"
            )
        if not isinstance(self.random_seed, int) or isinstance(
            self.random_seed,
            bool,
        ):
            raise ValueError("random seed must be an integer")

        normalized_overrides = []
        for override in self.sensor_standard_deviations:
            if len(override) != 2:
                raise ValueError(
                    "sensor noise overrides must be name-value pairs"
                )
            sensor_name, standard_deviation = override
            if not isinstance(sensor_name, str) or not sensor_name.strip():
                raise ValueError("noise override sensor name must be nonempty")
            if (
                not math.isfinite(standard_deviation)
                or standard_deviation < 0.0
            ):
                raise ValueError(
                    "sensor noise standard deviations must be finite and "
                    "nonnegative"
                )
            normalized_overrides.append(
                (sensor_name.strip(), float(standard_deviation))
            )

        names = tuple(name for name, _ in normalized_overrides)
        if len(set(names)) != len(names):
            raise ValueError("sensor noise override names must be unique")
        object.__setattr__(
            self,
            "sensor_standard_deviations",
            tuple(normalized_overrides),
        )

    def standard_deviation_for(self, sensor_name: str) -> float:
        """Return the configured standard deviation for one sensor."""

        overrides = dict(self.sensor_standard_deviations)
        return overrides.get(sensor_name, self.default_standard_deviation)


@dataclass(frozen=True)
class TemperatureNoiseResult:
    """Noisy observations and the configuration that generated them."""

    dataset: ObservationDataset
    noise_model: GaussianTemperatureNoise


def apply_gaussian_temperature_noise(
    dataset: ObservationDataset,
    noise_model: GaussianTemperatureNoise,
) -> TemperatureNoiseResult:
    """Return a noisy dataset without modifying the ideal input dataset."""

    sensor_names = {sensor.name for sensor in dataset.sensors}
    override_names = {
        name for name, _ in noise_model.sensor_standard_deviations
    }
    unknown_names = override_names - sensor_names
    if unknown_names:
        joined_names = ", ".join(sorted(unknown_names))
        raise ValueError(
            f"noise overrides reference unknown sensors: {joined_names}"
        )

    random_source = random.Random(noise_model.random_seed)
    noisy_observations = []
    for observation in dataset.observations:
        standard_deviation = noise_model.standard_deviation_for(
            observation.sensor_name
        )
        standard_normal_error = random_source.gauss(0.0, 1.0)
        temperature_error = standard_deviation * standard_normal_error
        noisy_observations.append(
            replace(
                observation,
                temperature=observation.temperature + temperature_error,
            )
        )

    step_settings = [
        MetadataSetting(
            "default_standard_deviation_K",
            noise_model.default_standard_deviation,
        ),
        MetadataSetting("random_seed", noise_model.random_seed),
    ]
    step_settings.extend(
        MetadataSetting(
            f"sensor_standard_deviation_K:{sensor_name}",
            standard_deviation,
        )
        for sensor_name, standard_deviation in (
            noise_model.sensor_standard_deviations
        )
    )
    noise_changes_values = any(
        noise_model.standard_deviation_for(sensor.name) > 0.0
        for sensor in dataset.sensors
    )
    provenance = dataset.provenance
    if noise_changes_values:
        provenance = append_observation_process_step(
            provenance,
            ObservationProcessStep(
                "gaussian_temperature_noise",
                tuple(step_settings),
            ),
        )
    noisy_dataset = replace(
        dataset,
        observations=tuple(noisy_observations),
        provenance=provenance,
    )
    return TemperatureNoiseResult(
        dataset=noisy_dataset,
        noise_model=noise_model,
    )


def reference_gaussian_temperature_noise() -> GaussianTemperatureNoise:
    """Return the generic 0.05 K, seed-2026 learning baseline."""

    return GaussianTemperatureNoise(
        default_standard_deviation=0.05,
        random_seed=2026,
    )


def run_noisy_contact_reference_test_stand(
    *,
    sampling_interval: float = 1.0,
    noise_model: Optional[GaussianTemperatureNoise] = None,
) -> TemperatureNoiseResult:
    """Return noisy observations from the hidden four-node reference truth."""

    ideal_dataset = run_ideal_contact_reference_test_stand(
        sampling_interval=sampling_interval
    )
    selected_noise_model = (
        reference_gaussian_temperature_noise()
        if noise_model is None
        else noise_model
    )
    return apply_gaussian_temperature_noise(
        ideal_dataset,
        selected_noise_model,
    )
