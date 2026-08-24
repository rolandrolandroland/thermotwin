"""Shared utilities for contact-resistance robustness studies."""

from dataclasses import replace
import math
from typing import Callable, NamedTuple, Sequence, Tuple

from ...inference.contact_resistance import (
    ALL_SENSOR_NAMES,
    REFERENCE_COLD_CONTACT_RESISTANCE,
    ContactResistanceDatasetSplit,
    ContactResistanceRegimeDataset,
    ContactResistanceSearchConfig,
    contact_resistance_observation_rmse,
    contact_resistance_training_loss,
    fit_cold_contact_resistance,
    simulate_contact_resistance_observations,
)
from ...observations.metadata import (
    MetadataSetting,
    ObservationProcessStep,
    append_observation_process_step,
)
from ...observations.lag import (
    FirstOrderTemperatureLag,
    apply_first_order_temperature_lag,
)
from ...observations.test_stand import (
    ObservationDataset,
    regular_measurement_times,
)


ROBUSTNESS_SEARCH_CONFIG = ContactResistanceSearchConfig(
    resistance_tolerance=1e-6,
    max_iterations=64,
)


class ContactResistanceRobustnessCaseResult(NamedTuple):
    """One fitted measurement case and its validation diagnostics."""

    name: str
    fitted_sensor_names: Tuple[str, ...]
    inferred_cold_contact_resistance: float
    signed_parameter_error: float
    absolute_parameter_error: float
    training_observation_count: int
    training_information_curvature: float
    search_evaluations: int
    training_observation_rmse: float
    validation_observation_rmse: float
    test_observation_rmse: float
    training_truth_rmse: float
    validation_truth_rmse: float
    test_truth_rmse: float


def map_contact_resistance_dataset_split(
    datasets: ContactResistanceDatasetSplit,
    transform: Callable[[ContactResistanceRegimeDataset], ObservationDataset],
) -> ContactResistanceDatasetSplit:
    """Transform observations while preserving regimes and whole splits."""

    if not isinstance(datasets, ContactResistanceDatasetSplit):
        raise ValueError("datasets must be a contact-resistance split")

    def transform_group(
        group: Sequence[ContactResistanceRegimeDataset],
    ) -> Tuple[ContactResistanceRegimeDataset, ...]:
        return tuple(
            replace(dataset, observations=transform(dataset))
            for dataset in group
        )

    return ContactResistanceDatasetSplit(
        train=transform_group(datasets.train),
        validation=transform_group(datasets.validation),
        test=transform_group(datasets.test),
    )


def downsample_observation_dataset(
    dataset: ObservationDataset,
    sampling_interval: float,
) -> ObservationDataset:
    """Keep regular output times from a more densely sampled dataset."""

    target_times = regular_measurement_times(
        dataset.measurement_times[-1],
        sampling_interval,
    )
    tolerance = 1e-12 * max(1.0, dataset.measurement_times[-1])
    retained = tuple(
        observation
        for observation in dataset.observations
        if any(
            math.isclose(
                observation.time,
                target,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            for target in target_times
        )
    )
    retained_times = tuple(
        time
        for time in target_times
        if any(
            math.isclose(
                observation.time,
                time,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            for observation in retained
        )
    )
    if retained_times != target_times:
        raise ValueError("dense dataset does not contain every output time")
    return replace(
        dataset,
        observations=retained,
        sampling_interval=float(sampling_interval),
        provenance=(
            dataset.provenance
            if sampling_interval == dataset.sampling_interval
            else append_observation_process_step(
                dataset.provenance,
                ObservationProcessStep(
                    "output_downsampling",
                    (
                        MetadataSetting(
                            "sampling_interval_s",
                            sampling_interval,
                        ),
                    ),
                ),
            )
        ),
    )


def lag_contact_resistance_dataset_split(
    ideal_datasets: ContactResistanceDatasetSplit,
    lag_model: FirstOrderTemperatureLag,
    *,
    dense_sampling_interval: float = 0.1,
) -> ContactResistanceDatasetSplit:
    """Apply sensor lag on dense truth before output downsampling."""

    if not isinstance(lag_model, FirstOrderTemperatureLag):
        raise ValueError("lag model must be first-order temperature lag")

    def lag_dataset(
        dataset: ContactResistanceRegimeDataset,
    ) -> ObservationDataset:
        output_interval = dataset.observations.sampling_interval
        if dense_sampling_interval > output_interval:
            raise ValueError(
                "dense lag interval must not exceed output interval"
            )
        if all(
            lag_model.time_constant_for(sensor.name) == 0.0
            for sensor in dataset.observations.sensors
        ):
            return apply_first_order_temperature_lag(
                dataset.observations,
                lag_model,
            ).dataset
        dense = simulate_contact_resistance_observations(
            dataset.regime,
            cold_contact_resistance=REFERENCE_COLD_CONTACT_RESISTANCE,
            sampling_interval=dense_sampling_interval,
        )
        lagged = apply_first_order_temperature_lag(
            dense,
            lag_model,
        ).dataset
        return downsample_observation_dataset(
            lagged,
            output_interval,
        )

    return map_contact_resistance_dataset_split(
        ideal_datasets,
        lag_dataset,
    )


def restrict_observation_dataset(
    dataset: ObservationDataset,
    sensor_names: Sequence[str],
) -> ObservationDataset:
    """Return a dataset containing only the selected sensor definitions."""

    try:
        selected_names = tuple(sensor_names)
    except TypeError as error:
        raise ValueError("sensor names must be a collection") from error
    if not selected_names:
        raise ValueError("at least one sensor must remain available")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("restricted sensor names must be unique")
    known_names = {sensor.name for sensor in dataset.sensors}
    unknown_names = set(selected_names) - known_names
    if unknown_names:
        joined_names = ", ".join(sorted(unknown_names))
        raise ValueError(f"restricted sensors are unknown: {joined_names}")
    selected_name_set = set(selected_names)
    sensors = tuple(
        sensor
        for sensor in dataset.sensors
        if sensor.name in selected_name_set
    )
    observations = tuple(
        observation
        for observation in dataset.observations
        if observation.sensor_name in selected_name_set
    )
    return replace(
        dataset,
        sensors=sensors,
        observations=observations,
        provenance=(
            dataset.provenance
            if len(sensors) == len(dataset.sensors)
            else append_observation_process_step(
                dataset.provenance,
                ObservationProcessStep(
                    "sensor_restriction",
                    tuple(
                        MetadataSetting(
                            f"retained_sensor_{index}",
                            sensor.name,
                        )
                        for index, sensor in enumerate(sensors)
                    ),
                ),
            )
        ),
    )


def match_observation_schema(
    reference: ObservationDataset,
    schema: ObservationDataset,
) -> ObservationDataset:
    """Select ideal records matching another dataset's sensors and times."""

    schema_sensor_names = tuple(sensor.name for sensor in schema.sensors)
    restricted_reference = restrict_observation_dataset(
        reference,
        schema_sensor_names,
    )
    schema_keys = {
        (observation.time, observation.sensor_name)
        for observation in schema.observations
    }
    observations = tuple(
        observation
        for observation in restricted_reference.observations
        if (observation.time, observation.sensor_name) in schema_keys
    )
    returned_keys = {
        (observation.time, observation.sensor_name)
        for observation in observations
    }
    if returned_keys != schema_keys:
        raise ValueError("reference cannot match the requested schema")
    return replace(
        restricted_reference,
        observations=observations,
        sampling_interval=schema.sampling_interval,
        provenance=append_observation_process_step(
            restricted_reference.provenance,
            ObservationProcessStep(
                "matched_observation_schema",
                (
                    MetadataSetting(
                        "retained_observation_count",
                        len(observations),
                    ),
                    MetadataSetting(
                        "schema_sampling_interval_s",
                        schema.sampling_interval,
                    ),
                ),
            ),
        ),
    )


def match_split_schema(
    reference: ContactResistanceDatasetSplit,
    schema: ContactResistanceDatasetSplit,
) -> ContactResistanceDatasetSplit:
    """Match an ideal split to another split's available records."""

    reference_by_name = {
        dataset.regime.name: dataset
        for group in (
            reference.train,
            reference.validation,
            reference.test,
        )
        for dataset in group
    }

    def match(dataset: ContactResistanceRegimeDataset) -> ObservationDataset:
        if dataset.regime.name not in reference_by_name:
            raise ValueError("schema regime is absent from the reference")
        return match_observation_schema(
            reference_by_name[dataset.regime.name].observations,
            dataset.observations,
        )

    return map_contact_resistance_dataset_split(schema, match)


def count_selected_observations(
    datasets: Sequence[ContactResistanceRegimeDataset],
    sensor_names: Sequence[str],
) -> int:
    """Count available readings from the selected sensors."""

    selected_names = set(sensor_names)
    return sum(
        observation.sensor_name in selected_names
        for dataset in datasets
        for observation in dataset.observations.observations
    )


def contact_resistance_information_curvature(
    training_datasets: Sequence[ContactResistanceRegimeDataset],
    *,
    fitted_sensor_names: Sequence[str],
    center: float = REFERENCE_COLD_CONTACT_RESISTANCE,
    parameter_step: float = 0.01,
) -> float:
    """Return local training-SSE curvature versus contact resistance."""

    if not math.isfinite(parameter_step) or parameter_step <= 0.0:
        raise ValueError("parameter step must be finite and positive")
    observation_count = count_selected_observations(
        training_datasets,
        fitted_sensor_names,
    )
    if observation_count == 0:
        raise ValueError("selected sensors have no training observations")
    losses = tuple(
        contact_resistance_training_loss(
            candidate,
            training_datasets,
            fitted_sensor_names=fitted_sensor_names,
        )
        for candidate in (
            center - parameter_step,
            center,
            center + parameter_step,
        )
    )
    return (
        observation_count
        * (losses[0] - 2.0 * losses[1] + losses[2])
        / (parameter_step * parameter_step)
    )


def _single_regime_rmse(
    cold_contact_resistance: float,
    datasets: Sequence[ContactResistanceRegimeDataset],
    sensor_names: Sequence[str],
) -> float:
    if len(datasets) != 1:
        raise ValueError("the frozen studies expect one regime per split")
    return contact_resistance_observation_rmse(
        cold_contact_resistance,
        datasets[0],
        sensor_names=sensor_names,
    )


def run_contact_resistance_robustness_case(
    name: str,
    observation_datasets: ContactResistanceDatasetSplit,
    truth_datasets: ContactResistanceDatasetSplit,
    *,
    fitted_sensor_names: Sequence[str],
    search: ContactResistanceSearchConfig = ROBUSTNESS_SEARCH_CONFIG,
) -> ContactResistanceRobustnessCaseResult:
    """Fit and score one deterministic measurement-design case."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("case name must be nonempty")
    fitted_sensor_names = tuple(fitted_sensor_names)
    fit = fit_cold_contact_resistance(
        observation_datasets.train,
        search,
        fitted_sensor_names=fitted_sensor_names,
    )
    inferred = fit.inferred_cold_contact_resistance
    signed_error = inferred - REFERENCE_COLD_CONTACT_RESISTANCE
    return ContactResistanceRobustnessCaseResult(
        name=name.strip(),
        fitted_sensor_names=fitted_sensor_names,
        inferred_cold_contact_resistance=inferred,
        signed_parameter_error=signed_error,
        absolute_parameter_error=abs(signed_error),
        training_observation_count=count_selected_observations(
            observation_datasets.train,
            fitted_sensor_names,
        ),
        training_information_curvature=(
            contact_resistance_information_curvature(
                observation_datasets.train,
                fitted_sensor_names=fitted_sensor_names,
            )
        ),
        search_evaluations=len(fit.evaluations),
        training_observation_rmse=_single_regime_rmse(
            inferred,
            observation_datasets.train,
            fitted_sensor_names,
        ),
        validation_observation_rmse=_single_regime_rmse(
            inferred,
            observation_datasets.validation,
            fitted_sensor_names,
        ),
        test_observation_rmse=_single_regime_rmse(
            inferred,
            observation_datasets.test,
            fitted_sensor_names,
        ),
        training_truth_rmse=_single_regime_rmse(
            inferred,
            truth_datasets.train,
            fitted_sensor_names,
        ),
        validation_truth_rmse=_single_regime_rmse(
            inferred,
            truth_datasets.validation,
            fitted_sensor_names,
        ),
        test_truth_rmse=_single_regime_rmse(
            inferred,
            truth_datasets.test,
            fitted_sensor_names,
        ),
    )


def format_contact_resistance_robustness_cases(
    title: str,
    cases: Sequence[ContactResistanceRobustnessCaseResult],
) -> str:
    """Format compact parameter and validation metrics for several cases."""

    lines = [title]
    for case in cases:
        sensor_names = ",".join(case.fitted_sensor_names)
        lines.append(
            f"{case.name}: R={case.inferred_cold_contact_resistance:.9f} K/W, "
            f"error={case.signed_parameter_error:+.9f} K/W, "
            f"records={case.training_observation_count}, "
            f"curvature={case.training_information_curvature:.6e}, "
            f"sensors={sensor_names}"
        )
        lines.append(
            "  observation RMSE train/validation/test: "
            f"{case.training_observation_rmse:.6f} / "
            f"{case.validation_observation_rmse:.6f} / "
            f"{case.test_observation_rmse:.6f} K"
        )
        lines.append(
            "  truth RMSE train/validation/test: "
            f"{case.training_truth_rmse:.6f} / "
            f"{case.validation_truth_rmse:.6f} / "
            f"{case.test_truth_rmse:.6f} K"
        )
    return "\n".join(lines)


def validate_sensor_names(sensor_names: Sequence[str]) -> Tuple[str, ...]:
    """Validate a public robustness-study sensor selection."""

    try:
        sensor_names = tuple(sensor_names)
    except TypeError as error:
        raise ValueError("sensor names must be a collection") from error
    if not sensor_names:
        raise ValueError("at least one sensor name is required")
    if len(set(sensor_names)) != len(sensor_names):
        raise ValueError("sensor names must be unique")
    unknown_names = set(sensor_names) - set(ALL_SENSOR_NAMES)
    if unknown_names:
        joined_names = ", ".join(sorted(unknown_names))
        raise ValueError(f"unsupported sensors: {joined_names}")
    return sensor_names
