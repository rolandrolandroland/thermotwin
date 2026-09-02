"""Nonlinear validation of the lumped next-experiment recommendation."""

from dataclasses import dataclass, replace
import math
import random
from statistics import fmean
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

from ..core.controls import PiecewiseConstantCurrent
from ..inference.experiment_selection import (
    ExperimentCandidateScore,
    ExperimentSelectionConfig,
    candidate_current,
    run_next_experiment_selection,
)
from ..inference.joint_thermal_parameters import (
    JointThermalFitConfig,
    JointThermalFitResult,
    JointThermalIdentifiability,
    JointThermalTruth,
    analyze_joint_thermal_identifiability,
    fit_joint_thermal_parameters,
    generate_joint_thermal_observations,
)
from ..inference.sparse_sensors import (
    simulate_accessible_observations,
    sparse_withheld_current,
)
from ..numerics.matrices import inverse_and_determinant


@dataclass(frozen=True)
class NonlinearExperimentSelectionConfig:
    """Frozen truth variation, trial count, profiles, and fit settings."""

    trial_count: int = 20
    first_seed: int = 52001
    truth_log_standard_deviations: Tuple[float, float, float] = (0.10, 0.10, 0.10)
    truth_bias_standard_deviation: float = 0.05
    profile_log_offsets: Tuple[float, ...] = (
        -0.30,
        -0.20,
        -0.10,
        0.0,
        0.10,
        0.20,
        0.30,
    )
    fit: JointThermalFitConfig = JointThermalFitConfig()
    selection: ExperimentSelectionConfig = ExperimentSelectionConfig(
        monte_carlo_trials=250
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.trial_count, int)
            or isinstance(self.trial_count, bool)
            or self.trial_count <= 0
        ):
            raise ValueError("trial count must be a positive integer")
        if (
            not isinstance(self.first_seed, int)
            or isinstance(self.first_seed, bool)
            or self.first_seed < 0
        ):
            raise ValueError("first seed must be a nonnegative integer")
        if len(self.truth_log_standard_deviations) != 3 or any(
            not math.isfinite(value) or value < 0.0
            for value in self.truth_log_standard_deviations
        ):
            raise ValueError("three finite nonnegative truth spreads are required")
        if (
            not math.isfinite(self.truth_bias_standard_deviation)
            or self.truth_bias_standard_deviation < 0.0
        ):
            raise ValueError("truth bias spread must be finite and nonnegative")
        if not self.profile_log_offsets or any(
            not math.isfinite(value) for value in self.profile_log_offsets
        ):
            raise ValueError("profile offsets must be finite and nonempty")
        if not isinstance(self.fit, JointThermalFitConfig):
            raise ValueError("fit must be a joint thermal fit configuration")
        if not isinstance(self.selection, ExperimentSelectionConfig):
            raise ValueError("selection must be an experiment-selection configuration")


class NonlinearExperimentDefinition(NamedTuple):
    role: str
    candidate: ExperimentCandidateScore


class NonlinearExperimentTrialResult(NamedTuple):
    trial_index: int
    truth_seed: int
    noise_seed: int
    role: str
    candidate_name: str
    truth: JointThermalTruth
    fit: JointThermalFitResult
    physical_log_rmse: float
    all_physical_intervals_cover: bool
    physical_interval_coverage_count: int
    physical_uncertainty_volume: float
    withheld_cold_face_rmse: float
    withheld_hot_face_rmse: float


class NonlinearExperimentSummary(NamedTuple):
    role: str
    candidate_name: str
    current_amplitude: float
    pulse_duration: float
    electrical_energy: float
    information_gain_nats: float
    trial_count: int
    mean_physical_log_rmse: float
    median_physical_log_rmse: float
    worst_physical_log_rmse: float
    individual_parameter_95_coverage: float
    simultaneous_physical_95_coverage: float
    mean_physical_uncertainty_volume: float
    mean_absolute_resistance_capacitance_correlation: float
    mean_absolute_resistance_lag_correlation: float
    mean_absolute_capacitance_lag_correlation: float
    mean_withheld_cold_face_rmse: float
    mean_withheld_hot_face_rmse: float
    search_bound_hits: int


class JointProfilePoint(NamedTuple):
    role: str
    parameter_index: int
    fixed_log_multiplier: float
    normalized_mean_squared_error: float
    fitted_log_multipliers: Tuple[float, float, float]


class NonlinearExperimentSelectionResult(NamedTuple):
    config: NonlinearExperimentSelectionConfig
    definitions: Tuple[NonlinearExperimentDefinition, ...]
    selected_identifiability: JointThermalIdentifiability
    zero_current_identifiability: JointThermalIdentifiability
    trials: Tuple[NonlinearExperimentTrialResult, ...]
    summaries: Tuple[NonlinearExperimentSummary, ...]
    profiles: Tuple[JointProfilePoint, ...]
    selected_rmse_reduction_vs_naive_percent: float
    selected_rmse_reduction_vs_resource_control_percent: float
    selected_interval_volume_reduction_vs_naive_percent: float
    selected_interval_volume_reduction_vs_resource_control_percent: float


def nonlinear_experiment_definitions(
    config: NonlinearExperimentSelectionConfig,
) -> Tuple[NonlinearExperimentDefinition, ...]:
    selection = run_next_experiment_selection(config.selection)
    alternatives = tuple(
        candidate
        for candidate in selection.candidates
        if candidate.feasible and candidate.name != selection.selected.name
    )
    resource_control = min(
        alternatives,
        key=lambda candidate: (
            abs(candidate.electrical_energy - selection.selected.electrical_energy),
            -candidate.electrical_energy,
        ),
    )
    return (
        NonlinearExperimentDefinition("selected", selection.selected),
        NonlinearExperimentDefinition("naive", selection.naive),
        NonlinearExperimentDefinition("resource_control", resource_control),
    )


def nonlinear_experiment_trial_seeds(
    first_seed: int,
    trial_index: int,
) -> Tuple[int, int]:
    if first_seed < 0 or trial_index < 0:
        raise ValueError("seeds and trial indices must be nonnegative")
    return first_seed + 2 * trial_index, first_seed + 2 * trial_index + 1


def _trial_truth(
    config: NonlinearExperimentSelectionConfig,
    seed: int,
) -> JointThermalTruth:
    random_source = random.Random(seed)
    physical = tuple(
        nominal * math.exp(random_source.gauss(0.0, spread))
        for nominal, spread in zip(
            config.fit.nominal_values,
            config.truth_log_standard_deviations,
        )
    )
    return JointThermalTruth(
        cold_contact_resistance=physical[0],
        cold_face_thermal_capacitance=physical[1],
        sensor_time_constant=physical[2],
        cold_sensor_bias=random_source.gauss(
            0.0, config.truth_bias_standard_deviation
        ),
        hot_sensor_bias=random_source.gauss(
            0.0, config.truth_bias_standard_deviation
        ),
    )


def _rmse(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("RMSE inputs must be nonempty and aligned")
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)
    )


def _physical_uncertainty_volume(fit: JointThermalFitResult) -> float:
    """Return sqrt(det(covariance)) in the three log-parameter coordinates."""

    physical = tuple(tuple(row[:3]) for row in fit.covariance[:3])
    try:
        _, determinant = inverse_and_determinant(physical)
    except ValueError:
        return math.inf
    return math.sqrt(max(0.0, determinant))


def _score_trial(
    *,
    trial_index: int,
    truth_seed: int,
    noise_seed: int,
    definition: NonlinearExperimentDefinition,
    truth: JointThermalTruth,
    fit_config: JointThermalFitConfig,
) -> NonlinearExperimentTrialResult:
    current = candidate_current(
        definition.candidate.current_amplitude,
        definition.candidate.pulse_duration,
    )
    observations = generate_joint_thermal_observations(
        current,
        truth,
        fit_config,
        noise_seed=noise_seed,
    )
    fit = fit_joint_thermal_parameters(current, observations, fit_config)
    log_errors = tuple(
        math.log(estimate / expected)
        for estimate, expected in zip(fit.physical_values, truth.physical_values)
    )
    intervals = fit.intervals[:3]
    coverage = tuple(
        interval.lower_95 <= expected <= interval.upper_95
        for interval, expected in zip(intervals, truth.physical_values)
    )
    withheld = sparse_withheld_current()
    _, truth_trajectory = simulate_accessible_observations(
        withheld,
        cold_contact_resistance=truth.cold_contact_resistance,
        cold_face_thermal_capacitance=truth.cold_face_thermal_capacitance,
        sensor_time_constant=truth.sensor_time_constant,
        sampling_interval=fit_config.sampling_interval,
        dense_time_step=fit_config.dense_time_step,
    )
    _, predicted_trajectory = simulate_accessible_observations(
        withheld,
        cold_contact_resistance=fit.physical_values[0],
        cold_face_thermal_capacitance=fit.physical_values[1],
        sensor_time_constant=fit.physical_values[2],
        sampling_interval=fit_config.sampling_interval,
        dense_time_step=fit_config.dense_time_step,
    )
    return NonlinearExperimentTrialResult(
        trial_index=trial_index,
        truth_seed=truth_seed,
        noise_seed=noise_seed,
        role=definition.role,
        candidate_name=definition.candidate.name,
        truth=truth,
        fit=fit,
        physical_log_rmse=math.sqrt(
            sum(value * value for value in log_errors) / len(log_errors)
        ),
        all_physical_intervals_cover=all(coverage),
        physical_interval_coverage_count=sum(coverage),
        physical_uncertainty_volume=_physical_uncertainty_volume(fit),
        withheld_cold_face_rmse=_rmse(
            predicted_trajectory.cold_face, truth_trajectory.cold_face
        ),
        withheld_hot_face_rmse=_rmse(
            predicted_trajectory.hot_face, truth_trajectory.hot_face
        ),
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def summarize_nonlinear_experiment_trials(
    trials: Sequence[NonlinearExperimentTrialResult],
    definitions: Sequence[NonlinearExperimentDefinition],
) -> Tuple[NonlinearExperimentSummary, ...]:
    trials = tuple(trials)
    summaries = []
    for definition in definitions:
        selected = tuple(trial for trial in trials if trial.role == definition.role)
        if not selected:
            raise ValueError("each experiment role needs at least one nonlinear trial")
        errors = tuple(trial.physical_log_rmse for trial in selected)
        summaries.append(
            NonlinearExperimentSummary(
                role=definition.role,
                candidate_name=definition.candidate.name,
                current_amplitude=definition.candidate.current_amplitude,
                pulse_duration=definition.candidate.pulse_duration,
                electrical_energy=definition.candidate.electrical_energy,
                information_gain_nats=definition.candidate.information_gain_nats,
                trial_count=len(selected),
                mean_physical_log_rmse=fmean(errors),
                median_physical_log_rmse=_median(errors),
                worst_physical_log_rmse=max(errors),
                individual_parameter_95_coverage=(
                    sum(trial.physical_interval_coverage_count for trial in selected)
                    / (3 * len(selected))
                ),
                simultaneous_physical_95_coverage=(
                    sum(trial.all_physical_intervals_cover for trial in selected)
                    / len(selected)
                ),
                mean_physical_uncertainty_volume=fmean(
                    trial.physical_uncertainty_volume for trial in selected
                ),
                mean_absolute_resistance_capacitance_correlation=fmean(
                    abs(trial.fit.correlation[0][1]) for trial in selected
                ),
                mean_absolute_resistance_lag_correlation=fmean(
                    abs(trial.fit.correlation[0][2]) for trial in selected
                ),
                mean_absolute_capacitance_lag_correlation=fmean(
                    abs(trial.fit.correlation[1][2]) for trial in selected
                ),
                mean_withheld_cold_face_rmse=fmean(
                    trial.withheld_cold_face_rmse for trial in selected
                ),
                mean_withheld_hot_face_rmse=fmean(
                    trial.withheld_hot_face_rmse for trial in selected
                ),
                search_bound_hits=sum(trial.fit.reached_bound for trial in selected),
            )
        )
    return tuple(summaries)


def _profiles(
    config: NonlinearExperimentSelectionConfig,
    definitions: Sequence[NonlinearExperimentDefinition],
    truth: JointThermalTruth,
    noise_seed: int,
) -> Tuple[JointProfilePoint, ...]:
    points = []
    for definition in definitions:
        if definition.role == "resource_control":
            continue
        current = candidate_current(
            definition.candidate.current_amplitude,
            definition.candidate.pulse_duration,
        )
        observations = generate_joint_thermal_observations(
            current,
            truth,
            config.fit,
            noise_seed=noise_seed,
        )
        for parameter_index in range(3):
            for fixed in config.profile_log_offsets:
                fixed_values = [None, None, None]
                fixed_values[parameter_index] = fixed
                starts = tuple(
                    tuple(
                        fixed if index == parameter_index else value
                        for index, value in enumerate(start)
                    )
                    for start in config.fit.initial_log_multipliers
                )
                profile_config = replace(
                    config.fit,
                    fixed_log_multipliers=tuple(fixed_values),
                    initial_log_multipliers=starts,
                )
                fit = fit_joint_thermal_parameters(
                    current, observations, profile_config
                )
                points.append(
                    JointProfilePoint(
                        role=definition.role,
                        parameter_index=parameter_index,
                        fixed_log_multiplier=fixed,
                        normalized_mean_squared_error=(
                            fit.normalized_mean_squared_error
                        ),
                        fitted_log_multipliers=fit.log_multipliers,
                    )
                )
    return tuple(points)


def _reduction(selected: float, comparison: float) -> float:
    return 100.0 * (1.0 - selected / comparison)


def run_nonlinear_experiment_selection_study(
    config: NonlinearExperimentSelectionConfig = (
        NonlinearExperimentSelectionConfig()
    ),
    *,
    include_profiles: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> NonlinearExperimentSelectionResult:
    definitions = nonlinear_experiment_definitions(config)
    trials = []
    first_truth = None
    first_noise_seed = None
    for trial_index in range(config.trial_count):
        if progress is not None:
            progress(f"nonlinear trial {trial_index + 1}/{config.trial_count}")
        truth_seed, noise_seed = nonlinear_experiment_trial_seeds(
            config.first_seed, trial_index
        )
        truth = _trial_truth(config, truth_seed)
        if first_truth is None:
            first_truth = truth
            first_noise_seed = noise_seed
        for definition in definitions:
            trials.append(
                _score_trial(
                    trial_index=trial_index,
                    truth_seed=truth_seed,
                    noise_seed=noise_seed,
                    definition=definition,
                    truth=truth,
                    fit_config=config.fit,
                )
            )
    summaries = summarize_nonlinear_experiment_trials(trials, definitions)
    by_role = {summary.role: summary for summary in summaries}
    selected = by_role["selected"]
    naive = by_role["naive"]
    resource = by_role["resource_control"]
    profiles = (
        _profiles(config, definitions, first_truth, first_noise_seed)
        if include_profiles
        else ()
    )
    if progress is not None and include_profiles:
        progress("representative selected/naive profiles complete")
    return NonlinearExperimentSelectionResult(
        config=config,
        definitions=definitions,
        selected_identifiability=analyze_joint_thermal_identifiability(
            candidate_current(
                definitions[0].candidate.current_amplitude,
                definitions[0].candidate.pulse_duration,
            ),
            config.fit,
        ),
        zero_current_identifiability=analyze_joint_thermal_identifiability(
            PiecewiseConstantCurrent.constant(0.0),
            config.fit,
        ),
        trials=tuple(trials),
        summaries=summaries,
        profiles=profiles,
        selected_rmse_reduction_vs_naive_percent=_reduction(
            selected.mean_physical_log_rmse, naive.mean_physical_log_rmse
        ),
        selected_rmse_reduction_vs_resource_control_percent=_reduction(
            selected.mean_physical_log_rmse, resource.mean_physical_log_rmse
        ),
        selected_interval_volume_reduction_vs_naive_percent=_reduction(
            selected.mean_physical_uncertainty_volume,
            naive.mean_physical_uncertainty_volume,
        ),
        selected_interval_volume_reduction_vs_resource_control_percent=_reduction(
            selected.mean_physical_uncertainty_volume,
            resource.mean_physical_uncertainty_volume,
        ),
    )
