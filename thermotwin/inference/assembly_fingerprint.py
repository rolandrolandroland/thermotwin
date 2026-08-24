"""Standardized pulse fingerprints for synthetic assembly quality control."""

from dataclasses import dataclass
import math
from typing import NamedTuple, Sequence, Tuple

from ..core.controls import PiecewiseConstantCurrent
from ..observations.noise import GaussianTemperatureNoise, apply_gaussian_temperature_noise
from .sparse_sensors import (
    ACCESSIBLE_SENSOR_NAMES,
    simulate_accessible_observations,
)
from ..observations.test_stand import ObservationDataset


@dataclass(frozen=True)
class AssemblySpecification:
    """Synthetic assembly identifier and hidden cold-interface resistance."""

    name: str
    cold_contact_resistance: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("assembly name must be nonempty")
        object.__setattr__(self, "name", self.name.strip())
        if (
            not math.isfinite(self.cold_contact_resistance)
            or self.cold_contact_resistance <= 0.0
        ):
            raise ValueError("assembly contact resistance must be positive and finite")


class AssemblyFingerprint(NamedTuple):
    name: str
    true_cold_contact_resistance: float
    inferred_cold_contact_resistance: float
    lower_95: float
    upper_95: float
    truth_covered: bool
    cold_exchanger_drop: float
    hot_exchanger_rise: float
    classification: str


class AssemblyFingerprintStudyResult(NamedTuple):
    pulse_current: float
    pulse_duration: float
    noise_standard_deviation: float
    fingerprints: Tuple[AssemblyFingerprint, ...]


def reference_assembly_batch() -> Tuple[AssemblySpecification, ...]:
    return (
        AssemblySpecification("low_loss", 0.15),
        AssemblySpecification("reference_a", 0.24),
        AssemblySpecification("reference_b", 0.26),
        AssemblySpecification("elevated_loss", 0.35),
        AssemblySpecification("high_loss", 0.50),
    )


def standardized_fingerprint_current() -> PiecewiseConstantCurrent:
    """Use the feasible 0.8 A, 20 s pulse selected by the design study."""

    return PiecewiseConstantCurrent.pulse(
        start_time=5.0,
        end_time=25.0,
        pulse_current=0.8,
    )


def _prediction_map(dataset: ObservationDataset):
    return {
        (item.sensor_name, item.time): item.temperature
        for item in dataset.observations
    }


def _profiled_mse(
    observed: ObservationDataset,
    predicted: ObservationDataset,
) -> float:
    prediction = _prediction_map(predicted)
    differences = {name: [] for name in ACCESSIBLE_SENSOR_NAMES}
    for item in observed.observations:
        differences[item.sensor_name].append(
            item.temperature - prediction[(item.sensor_name, item.time)]
        )
    biases = {
        name: sum(values) / len(values)
        for name, values in differences.items()
    }
    residuals = tuple(
        prediction[(item.sensor_name, item.time)]
        + biases[item.sensor_name]
        - item.temperature
        for item in observed.observations
    )
    return sum(value * value for value in residuals) / len(residuals)


def _predict(resistance: float) -> ObservationDataset:
    dataset, _ = simulate_accessible_observations(
        standardized_fingerprint_current(),
        cold_contact_resistance=resistance,
        sensor_time_constant=1.5,
        sampling_interval=1.0,
        dense_time_step=0.2,
    )
    return dataset


def _fit_resistance(observed: ObservationDataset) -> float:
    left = 0.08
    right = 0.70
    golden = 0.5 * (1.0 + math.sqrt(5.0))
    interior_left = right - (right - left) / golden
    interior_right = left + (right - left) / golden
    left_loss = _profiled_mse(observed, _predict(interior_left))
    right_loss = _profiled_mse(observed, _predict(interior_right))
    for _ in range(44):
        if left_loss <= right_loss:
            right = interior_right
            interior_right = interior_left
            right_loss = left_loss
            interior_left = right - (right - left) / golden
            left_loss = _profiled_mse(observed, _predict(interior_left))
        else:
            left = interior_left
            interior_left = interior_right
            left_loss = right_loss
            interior_right = left + (right - left) / golden
            right_loss = _profiled_mse(observed, _predict(interior_right))
    return 0.5 * (left + right)


def _local_interval(
    observed: ObservationDataset,
    estimate: float,
    noise_standard_deviation: float,
) -> Tuple[float, float]:
    step = max(1e-4, 0.01 * estimate)
    minus_map = _prediction_map(_predict(estimate - step))
    plus_map = _prediction_map(_predict(estimate + step))
    derivatives = {name: [] for name in ACCESSIBLE_SENSOR_NAMES}
    for item in observed.observations:
        derivatives[item.sensor_name].append(
            (
                plus_map[(item.sensor_name, item.time)]
                - minus_map[(item.sensor_name, item.time)]
            )
            / (2.0 * step)
        )
    centered_derivatives = []
    for values in derivatives.values():
        mean_value = sum(values) / len(values)
        centered_derivatives.extend(value - mean_value for value in values)
    information = sum(value * value for value in centered_derivatives) / (
        noise_standard_deviation ** 2
    )
    standard_error = 1.0 / math.sqrt(information)
    return estimate - 1.96 * standard_error, estimate + 1.96 * standard_error


def _classification(estimate: float) -> str:
    if estimate < 0.20:
        return "low_interface_loss"
    if estimate <= 0.30:
        return "reference_band"
    return "elevated_interface_loss"


def run_assembly_fingerprint_study(
    assemblies: Sequence[AssemblySpecification] = reference_assembly_batch(),
    *,
    noise_standard_deviation: float = 0.02,
) -> AssemblyFingerprintStudyResult:
    assemblies = tuple(assemblies)
    if not assemblies:
        raise ValueError("at least one assembly is required")
    if len({assembly.name for assembly in assemblies}) != len(assemblies):
        raise ValueError("assembly names must be unique")
    if not math.isfinite(noise_standard_deviation) or noise_standard_deviation <= 0.0:
        raise ValueError("noise standard deviation must be finite and positive")
    fingerprints = []
    for index, assembly in enumerate(assemblies):
        ideal, _ = simulate_accessible_observations(
            standardized_fingerprint_current(),
            cold_contact_resistance=assembly.cold_contact_resistance,
            sensor_time_constant=1.5,
            sampling_interval=1.0,
            dense_time_step=0.2,
        )
        noisy = apply_gaussian_temperature_noise(
            ideal,
            GaussianTemperatureNoise(
                default_standard_deviation=noise_standard_deviation,
                random_seed=3100 + index,
            ),
        ).dataset
        estimate = _fit_resistance(noisy)
        lower, upper = _local_interval(
            noisy, estimate, noise_standard_deviation
        )
        cold_history = noisy.observations_for(ACCESSIBLE_SENSOR_NAMES[0])
        hot_history = noisy.observations_for(ACCESSIBLE_SENSOR_NAMES[1])
        fingerprints.append(
            AssemblyFingerprint(
                name=assembly.name,
                true_cold_contact_resistance=assembly.cold_contact_resistance,
                inferred_cold_contact_resistance=estimate,
                lower_95=lower,
                upper_95=upper,
                truth_covered=lower <= assembly.cold_contact_resistance <= upper,
                cold_exchanger_drop=(
                    cold_history[0].temperature
                    - min(item.temperature for item in cold_history)
                ),
                hot_exchanger_rise=(
                    max(item.temperature for item in hot_history)
                    - hot_history[0].temperature
                ),
                classification=_classification(estimate),
            )
        )
    return AssemblyFingerprintStudyResult(
        pulse_current=0.8,
        pulse_duration=20.0,
        noise_standard_deviation=noise_standard_deviation,
        fingerprints=tuple(fingerprints),
    )


def format_assembly_fingerprint_report(
    result: AssemblyFingerprintStudyResult,
) -> str:
    lines = [
        "ThermoTwin synthetic assembly thermal fingerprints",
        (
            f"standardized pulse: {result.pulse_current:.1f} A for "
            f"{result.pulse_duration:.0f} s"
        ),
        "name | inferred R_c (K/W) | 95% interval | classification",
    ]
    for item in result.fingerprints:
        lines.append(
            f"{item.name} | {item.inferred_cold_contact_resistance:.4f} | "
            f"[{item.lower_95:.4f}, {item.upper_95:.4f}] | {item.classification}"
        )
    return "\n".join(lines)


def main() -> None:
    print(format_assembly_fingerprint_report(run_assembly_fingerprint_study()))


if __name__ == "__main__":
    main()
