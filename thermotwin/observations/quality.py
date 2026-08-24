"""Compact quality summaries for virtual ThermoTwin datasets."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .test_stand import ObservationDataset, regular_measurement_times


@dataclass(frozen=True)
class SensorDatasetQuality:
    """Completeness and range information for one named sensor."""

    sensor_name: str
    observation_count: int
    expected_observation_count: int
    missing_observation_count: int
    completeness_fraction: float
    minimum_temperature: Optional[float]
    maximum_temperature: Optional[float]


@dataclass(frozen=True)
class DatasetQualitySummary:
    """Schema, completeness, range, and provenance checks for one dataset."""

    experiment_name: str
    regime_name: str
    split: str
    observation_count: int
    expected_observation_count: int
    missing_observation_count: int
    completeness_fraction: float
    sensor_count: int
    observed_measurement_time_count: int
    expected_measurement_time_count: int
    start_time: float
    end_time: float
    minimum_temperature: float
    maximum_temperature: float
    minimum_current: float
    maximum_current: float
    provenance_available: bool
    ground_truth_available: bool
    observation_process_steps: Tuple[str, ...]
    sensors: Tuple[SensorDatasetQuality, ...]


@dataclass(frozen=True)
class DatasetCollectionQualitySummary:
    """Aggregate quality and whole-regime split checks."""

    datasets: Tuple[DatasetQualitySummary, ...]
    dataset_count: int
    total_observation_count: int
    total_expected_observation_count: int
    overall_completeness_fraction: float
    all_provenance_available: bool
    all_ground_truth_available: bool
    regime_names_unique: bool
    required_splits_present: bool
    train_regimes: Tuple[str, ...]
    validation_regimes: Tuple[str, ...]
    test_regimes: Tuple[str, ...]


def _expected_measurement_times(dataset: ObservationDataset) -> Tuple[float, ...]:
    if dataset.provenance is not None:
        duration = dataset.provenance.experiment.duration
    else:
        duration = dataset.measurement_times[-1]
    return regular_measurement_times(duration, dataset.sampling_interval)


def summarize_observation_dataset(
    dataset: ObservationDataset,
) -> DatasetQualitySummary:
    """Calculate deterministic quality statistics for one observation table."""

    if not isinstance(dataset, ObservationDataset):
        raise ValueError("dataset must be an observation dataset")
    expected_times = _expected_measurement_times(dataset)
    expected_per_sensor = len(expected_times)
    expected_total = expected_per_sensor * len(dataset.sensors)
    observed_total = len(dataset.observations)
    if observed_total > expected_total:
        raise ValueError(
            "dataset contains more observations than its regular schema"
        )

    sensor_summaries = []
    for sensor in dataset.sensors:
        observations = dataset.observations_for(sensor.name)
        observed_count = len(observations)
        if observed_count > expected_per_sensor:
            raise ValueError(
                "sensor contains more observations than its regular schema"
            )
        temperatures = tuple(item.temperature for item in observations)
        missing_count = expected_per_sensor - observed_count
        sensor_summaries.append(
            SensorDatasetQuality(
                sensor_name=sensor.name,
                observation_count=observed_count,
                expected_observation_count=expected_per_sensor,
                missing_observation_count=missing_count,
                completeness_fraction=observed_count / expected_per_sensor,
                minimum_temperature=(
                    None if not temperatures else min(temperatures)
                ),
                maximum_temperature=(
                    None if not temperatures else max(temperatures)
                ),
            )
        )

    temperatures = tuple(
        observation.temperature for observation in dataset.observations
    )
    currents = tuple(observation.current for observation in dataset.observations)
    provenance = dataset.provenance
    experiment = None if provenance is None else provenance.experiment
    missing_total = expected_total - observed_total
    return DatasetQualitySummary(
        experiment_name=(
            "unrecorded" if experiment is None else experiment.experiment_name
        ),
        regime_name="unrecorded" if experiment is None else experiment.regime_name,
        split="unsplit" if experiment is None else experiment.split,
        observation_count=observed_total,
        expected_observation_count=expected_total,
        missing_observation_count=missing_total,
        completeness_fraction=observed_total / expected_total,
        sensor_count=len(dataset.sensors),
        observed_measurement_time_count=len(dataset.measurement_times),
        expected_measurement_time_count=expected_per_sensor,
        start_time=dataset.measurement_times[0],
        end_time=dataset.measurement_times[-1],
        minimum_temperature=min(temperatures),
        maximum_temperature=max(temperatures),
        minimum_current=min(currents),
        maximum_current=max(currents),
        provenance_available=provenance is not None,
        ground_truth_available=experiment is not None,
        observation_process_steps=(
            ()
            if provenance is None
            else tuple(step.name for step in provenance.observation_steps)
        ),
        sensors=tuple(sensor_summaries),
    )


def summarize_dataset_collection(
    datasets: Sequence[ObservationDataset],
) -> DatasetCollectionQualitySummary:
    """Summarize multiple datasets and verify whole-regime split integrity."""

    try:
        datasets = tuple(datasets)
    except TypeError as error:
        raise ValueError("datasets must be observation datasets") from error
    if not datasets or not all(
        isinstance(dataset, ObservationDataset) for dataset in datasets
    ):
        raise ValueError("datasets must be a nonempty observation collection")

    summaries = tuple(summarize_observation_dataset(item) for item in datasets)
    total_observed = sum(item.observation_count for item in summaries)
    total_expected = sum(
        item.expected_observation_count for item in summaries
    )
    regimes_by_split = {
        split: tuple(
            item.regime_name for item in summaries if item.split == split
        )
        for split in ("train", "validation", "test")
    }
    regime_names = tuple(
        item.regime_name
        for item in summaries
        if item.regime_name != "unrecorded"
    )
    return DatasetCollectionQualitySummary(
        datasets=summaries,
        dataset_count=len(summaries),
        total_observation_count=total_observed,
        total_expected_observation_count=total_expected,
        overall_completeness_fraction=total_observed / total_expected,
        all_provenance_available=all(
            item.provenance_available for item in summaries
        ),
        all_ground_truth_available=all(
            item.ground_truth_available for item in summaries
        ),
        regime_names_unique=len(regime_names) == len(set(regime_names)),
        required_splits_present=all(regimes_by_split.values()),
        train_regimes=regimes_by_split["train"],
        validation_regimes=regimes_by_split["validation"],
        test_regimes=regimes_by_split["test"],
    )


def _pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def format_dataset_quality_report(
    summary: DatasetCollectionQualitySummary,
) -> str:
    """Format a compact, human-readable collection-quality report."""

    if not isinstance(summary, DatasetCollectionQualitySummary):
        raise ValueError("summary must be a dataset collection summary")
    lines = [
        "ThermoTwin dataset quality report",
        f"datasets: {summary.dataset_count}",
        (
            "observations: "
            f"{summary.total_observation_count}/"
            f"{summary.total_expected_observation_count} "
            f"({100.0 * summary.overall_completeness_fraction:.2f}%)"
        ),
        "provenance recorded: "
        f"{_pass_fail(summary.all_provenance_available)}",
        "ground truth recorded: "
        f"{_pass_fail(summary.all_ground_truth_available)}",
        "whole-regime names unique: "
        f"{_pass_fail(summary.regime_names_unique)}",
        "train/validation/test present: "
        f"{_pass_fail(summary.required_splits_present)}",
        "",
        "regime | split | records | expected | complete | temperature range (K) | current range (A)",
        "--- | --- | ---: | ---: | ---: | ---: | ---:",
    ]
    for dataset in summary.datasets:
        lines.append(
            f"{dataset.regime_name} | {dataset.split} | "
            f"{dataset.observation_count} | "
            f"{dataset.expected_observation_count} | "
            f"{100.0 * dataset.completeness_fraction:.2f}% | "
            f"{dataset.minimum_temperature:.6f} to "
            f"{dataset.maximum_temperature:.6f} | "
            f"{dataset.minimum_current:.6f} to "
            f"{dataset.maximum_current:.6f}"
        )
    return "\n".join(lines)
