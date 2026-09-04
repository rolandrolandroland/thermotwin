"""Sequential experiment selection with an explicit model-mismatch control.

Three campaign policies share a cumulative energy budget: posterior-aware
adaptive selection, a precommitted greedy D-optimal batch, and a plausible
engineer heuristic.  The same four-node model plans and fits every campaign.
Observations come either from that model or from an independent five-state
truth with an unmodeled cold-interface thermal mass.
"""

from dataclasses import dataclass, replace
import math
import random
from statistics import fmean, median
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

from ..core.controls import PiecewiseConstantCurrent
from ..inference.experiment_selection import (
    ExperimentCandidateScore,
    ExperimentSelectionConfig,
    candidate_current,
    score_experiment_candidate,
)
from ..inference.joint_thermal_parameters import (
    JointThermalFitConfig,
    JointThermalTruth,
)
from ..inference.sparse_sensors import (
    ACCESSIBLE_SENSOR_NAMES,
    simulate_accessible_observations,
    sparse_withheld_current,
)
from ..numerics.matrices import (
    gram_matrix,
    inverse_and_determinant,
    matrix_add,
)
from ..observations.bias import FixedTemperatureBias, apply_fixed_temperature_bias
from ..observations.noise import GaussianTemperatureNoise, apply_gaussian_temperature_noise
from ..observations.test_stand import ObservationDataset
from ..simulation.four_node_experiments import constant_current_contact_reference_experiment
from ..simulation.interface_mass_mismatch import (
    InterfaceMassMismatch,
    simulate_interface_mass_observations,
)


TRUTH_CONDITIONS = ("matched_model", "extra_interface_mass")
CAMPAIGN_STRATEGIES = ("adaptive", "static_d_optimal", "engineer_heuristic")


@dataclass(frozen=True)
class AdaptiveCampaignConfig:
    """Frozen campaign resources, truth variation, and decision gates."""

    trial_count: int = 20
    experiment_count: int = 4
    total_energy_budget: float = 65.0
    first_seed: int = 73_001
    truth_log_standard_deviations: Tuple[float, float, float] = (
        0.18,
        0.18,
        0.18,
    )
    run_bias_standard_deviation: float = 0.05
    prior_log_standard_deviations: Tuple[float, float, float] = (
        0.20,
        0.50,
        0.30,
    )
    prediction_rmse_threshold: float = 0.04
    hidden_face_rmse_threshold: float = 0.05
    switch_bias_threshold: float = 0.05
    confidence_log_standard_error: float = 0.10
    parameter_log_rmse_threshold: float = 0.10
    mismatch: InterfaceMassMismatch = InterfaceMassMismatch()
    fit: JointThermalFitConfig = JointThermalFitConfig(
        dense_time_step=0.5,
        gauss_newton_iterations=5,
        initial_log_multipliers=((0.0, 0.0, 0.0),),
    )
    selection: ExperimentSelectionConfig = ExperimentSelectionConfig(
        dense_time_step=0.5,
        monte_carlo_trials=1,
    )
    heuristic_candidate_names: Tuple[str, ...] = (
        "0.4A_20s",
        "0.6A_20s",
        "0.8A_15s",
        "1.0A_10s",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.trial_count, int) or self.trial_count <= 0:
            raise ValueError("trial count must be a positive integer")
        if not isinstance(self.experiment_count, int) or self.experiment_count <= 0:
            raise ValueError("experiment count must be a positive integer")
        for name, value in (
            ("total energy budget", self.total_energy_budget),
            ("run bias standard deviation", self.run_bias_standard_deviation),
            ("prediction RMSE threshold", self.prediction_rmse_threshold),
            ("hidden-face RMSE threshold", self.hidden_face_rmse_threshold),
            ("switch bias threshold", self.switch_bias_threshold),
            ("confidence standard error", self.confidence_log_standard_error),
            ("parameter log-RMSE threshold", self.parameter_log_rmse_threshold),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.first_seed, int) or self.first_seed < 0:
            raise ValueError("first seed must be a nonnegative integer")
        for name, values in (
            ("truth log standard deviations", self.truth_log_standard_deviations),
            ("prior log standard deviations", self.prior_log_standard_deviations),
        ):
            if len(values) != 3 or any(
                not math.isfinite(value) or value <= 0.0 for value in values
            ):
                raise ValueError(f"{name} must contain three positive values")
        if len(self.heuristic_candidate_names) != self.experiment_count:
            raise ValueError("heuristic plan must match the experiment count")
        if len(set(self.heuristic_candidate_names)) != len(
            self.heuristic_candidate_names
        ):
            raise ValueError("heuristic candidate names must be distinct")


class CampaignObservation(NamedTuple):
    candidate_name: str
    current: PiecewiseConstantCurrent
    observations: ObservationDataset


class SequentialFitResult(NamedTuple):
    log_multipliers: Tuple[float, float, float]
    physical_values: Tuple[float, float, float]
    observation_rmse: float
    normalized_mean_squared_error: float
    covariance: Tuple[Tuple[float, ...], ...]
    log_standard_errors: Tuple[float, float, float]
    uncertainty_volume: float
    reached_bound: bool
    evaluation_count: int


class CampaignStepResult(NamedTuple):
    truth_condition: str
    strategy: str
    trial_index: int
    step_index: int
    candidate_name: str
    candidate_energy: float
    cumulative_energy: float
    physical_values: Tuple[float, float, float]
    parameter_log_rmse: float
    observation_rmse: float
    uncertainty_volume: float
    maximum_log_standard_error: float
    heldout_prediction_rmse: float
    heldout_hidden_face_rmse: float
    heldout_switch_bias: float
    parameter_recovery_passed: bool
    prediction_passed: bool
    confident: bool
    false_confidence: bool
    reached_bound: bool


class CampaignStepSummary(NamedTuple):
    truth_condition: str
    strategy: str
    step_index: int
    trial_count: int
    mean_cumulative_energy: float
    mean_parameter_log_rmse: float
    mean_observation_rmse: float
    median_uncertainty_volume: float
    mean_heldout_prediction_rmse: float
    mean_heldout_hidden_face_rmse: float
    parameter_recovery_pass_rate: float
    prediction_pass_rate: float
    confidence_rate: float
    false_confidence_rate: float
    bound_hit_rate: float


class CampaignStrategyOutcome(NamedTuple):
    truth_condition: str
    strategy: str
    median_first_passing_step: float
    median_energy_to_prediction_gate: float
    never_passed_rate: float
    final_prediction_pass_rate: float
    final_false_confidence_rate: float


class AdaptiveExperimentCampaignResult(NamedTuple):
    config: AdaptiveCampaignConfig
    candidates: Tuple[ExperimentCandidateScore, ...]
    static_plan: Tuple[str, ...]
    heuristic_plan: Tuple[str, ...]
    steps: Tuple[CampaignStepResult, ...]
    summaries: Tuple[CampaignStepSummary, ...]
    outcomes: Tuple[CampaignStrategyOutcome, ...]


def _values_from_log_multipliers(
    offsets: Sequence[float],
    nominal_values: Sequence[float],
) -> Tuple[float, float, float]:
    values = tuple(
        nominal * math.exp(offset)
        for nominal, offset in zip(nominal_values, offsets)
    )
    if len(values) != 3:
        raise ValueError("three physical values are required")
    return values  # type: ignore[return-value]


def _prediction_map(dataset: ObservationDataset) -> Dict[Tuple[str, float], float]:
    return {
        (item.sensor_name, item.time): item.temperature
        for item in dataset.observations
    }


def _profiled_residuals(
    observed: ObservationDataset,
    predicted: ObservationDataset,
    *,
    scale: float,
) -> Tuple[float, ...]:
    prediction = _prediction_map(predicted)
    differences = {name: [] for name in ACCESSIBLE_SENSOR_NAMES}
    for item in observed.observations:
        key = (item.sensor_name, item.time)
        if key not in prediction:
            raise ValueError("prediction is missing an observed sensor-time pair")
        differences[item.sensor_name].append(item.temperature - prediction[key])
    if any(not values for values in differences.values()):
        raise ValueError("each experiment needs both accessible sensors")
    biases = {
        name: fmean(values)
        for name, values in differences.items()
    }
    return tuple(
        (
            prediction[(item.sensor_name, item.time)]
            + biases[item.sensor_name]
            - item.temperature
        )
        / scale
        for item in observed.observations
    )


def fit_sequential_campaign(
    runs: Sequence[CampaignObservation],
    config: AdaptiveCampaignConfig,
    *,
    initial_log_multipliers: Optional[Tuple[float, float, float]] = None,
) -> SequentialFitResult:
    """Jointly refit shared physics while profiling per-run sensor offsets."""

    runs = tuple(runs)
    if not runs:
        raise ValueError("at least one campaign observation is required")
    fit_config = config.fit
    lower_logs, upper_logs = zip(*fit_config.log_bounds)
    cache: Dict[Tuple[float, float, float], Tuple[float, ...]] = {}
    observation_count = sum(len(run.observations.observations) for run in runs)
    evaluation_count = 0

    def bounded(values: Sequence[float]) -> Tuple[float, float, float]:
        return tuple(
            min(upper_logs[index], max(lower_logs[index], float(value)))
            for index, value in enumerate(values)
        )  # type: ignore[return-value]

    def residuals(offsets: Sequence[float]) -> Tuple[float, ...]:
        nonlocal evaluation_count
        key = bounded(offsets)
        if key not in cache:
            values = _values_from_log_multipliers(key, fit_config.nominal_values)
            combined = []
            for run in runs:
                prediction, _ = simulate_accessible_observations(
                    run.current,
                    cold_contact_resistance=values[0],
                    cold_face_thermal_capacitance=values[1],
                    sensor_time_constant=values[2],
                    sampling_interval=fit_config.sampling_interval,
                    dense_time_step=fit_config.dense_time_step,
                )
                combined.extend(
                    _profiled_residuals(
                        run.observations,
                        prediction,
                        scale=fit_config.noise_standard_deviation,
                    )
                )
            combined.extend(
                offset / prior_scale
                for offset, prior_scale in zip(
                    key, config.prior_log_standard_deviations
                )
            )
            cache[key] = tuple(combined)
            evaluation_count += 1
        return cache[key]

    def objective(offsets: Sequence[float]) -> float:
        values = residuals(offsets)
        return sum(value * value for value in values) / len(values)

    def optimize(start: Sequence[float]) -> Tuple[float, float, float]:
        values = list(bounded(start))
        damping = fit_config.initial_damping
        objective(values)
        for _ in range(fit_config.gauss_newton_iterations):
            error = residuals(values)
            columns = []
            for index in range(3):
                minus = list(values)
                plus = list(values)
                minus[index] = max(
                    lower_logs[index],
                    minus[index] - fit_config.finite_difference_step,
                )
                plus[index] = min(
                    upper_logs[index],
                    plus[index] + fit_config.finite_difference_step,
                )
                denominator = plus[index] - minus[index]
                left = residuals(minus)
                right = residuals(plus)
                columns.append(
                    tuple(
                        (right_value - left_value) / denominator
                        for left_value, right_value in zip(left, right)
                    )
                )
            normal = tuple(
                tuple(
                    sum(a * b for a, b in zip(left, right))
                    for right in columns
                )
                for left in columns
            )
            gradient = tuple(
                sum(value * item for value, item in zip(column, error))
                for column in columns
            )
            damped = tuple(
                tuple(
                    item
                    + (
                        damping * max(1.0, normal[row][row])
                        if row == column
                        else 0.0
                    )
                    for column, item in enumerate(matrix_row)
                )
                for row, matrix_row in enumerate(normal)
            )
            try:
                inverse, _ = inverse_and_determinant(damped)
            except ValueError:
                damping *= 10.0
                continue
            update = tuple(
                -sum(item * component for item, component in zip(row, gradient))
                for row in inverse
            )
            starting_loss = objective(values)
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.1, 0.03):
                candidate = [
                    min(
                        upper_logs[index],
                        max(lower_logs[index], value + fraction * update[index]),
                    )
                    for index, value in enumerate(values)
                ]
                if objective(candidate) < starting_loss:
                    values = list(bounded(candidate))
                    damping = max(fit_config.initial_damping * 1.0e-3, damping * 0.3)
                    accepted = True
                    break
            if not accepted:
                damping *= 10.0
        return bounded(values)

    if initial_log_multipliers is None:
        starts = fit_config.initial_log_multipliers
    else:
        starts = (initial_log_multipliers, (0.0, 0.0, 0.0))
    unique_starts = tuple(dict.fromkeys(bounded(start) for start in starts))
    optima = tuple(optimize(start) for start in unique_starts)
    best = min(optima, key=objective)
    best_residuals = residuals(best)

    columns = []
    for index in range(3):
        minus = list(best)
        plus = list(best)
        minus[index] = max(
            lower_logs[index], minus[index] - fit_config.finite_difference_step
        )
        plus[index] = min(
            upper_logs[index], plus[index] + fit_config.finite_difference_step
        )
        denominator = plus[index] - minus[index]
        left = residuals(minus)
        right = residuals(plus)
        columns.append(
            tuple(
                (right_value - left_value) / denominator
                for left_value, right_value in zip(left, right)
            )
        )
    information = tuple(
        tuple(
            sum(a * b for a, b in zip(left, right))
            for right in columns
        )
        for left in columns
    )
    covariance, information_determinant = inverse_and_determinant(information)
    standard_errors = tuple(
        math.sqrt(max(0.0, covariance[index][index])) for index in range(3)
    )
    covariance_determinant = 1.0 / information_determinant
    observation_residuals = best_residuals[:observation_count]
    return SequentialFitResult(
        log_multipliers=best,
        physical_values=_values_from_log_multipliers(
            best, fit_config.nominal_values
        ),
        observation_rmse=(
            fit_config.noise_standard_deviation
            * math.sqrt(
                sum(value * value for value in observation_residuals)
                / len(observation_residuals)
            )
        ),
        normalized_mean_squared_error=objective(best),
        covariance=covariance,
        log_standard_errors=standard_errors,  # type: ignore[arg-type]
        uncertainty_volume=math.sqrt(max(0.0, covariance_determinant)),
        reached_bound=any(
            abs(best[index] - lower_logs[index]) <= 1.0e-5
            or abs(upper_logs[index] - best[index]) <= 1.0e-5
            for index in range(3)
        ),
        evaluation_count=evaluation_count,
    )


def _candidate_information(
    candidate: ExperimentCandidateScore,
    log_multipliers: Tuple[float, float, float],
    config: AdaptiveCampaignConfig,
) -> Tuple[Tuple[float, ...], ...]:
    fit_config = config.fit
    columns = []
    for parameter_index in range(3):
        minus = list(log_multipliers)
        plus = list(log_multipliers)
        minus[parameter_index] -= fit_config.finite_difference_step
        plus[parameter_index] += fit_config.finite_difference_step
        minus_values = _values_from_log_multipliers(
            minus, fit_config.nominal_values
        )
        plus_values = _values_from_log_multipliers(
            plus, fit_config.nominal_values
        )
        current = candidate_current(
            candidate.current_amplitude, candidate.pulse_duration
        )
        left, _ = simulate_accessible_observations(
            current,
            cold_contact_resistance=minus_values[0],
            cold_face_thermal_capacitance=minus_values[1],
            sensor_time_constant=minus_values[2],
            sampling_interval=fit_config.sampling_interval,
            dense_time_step=fit_config.dense_time_step,
        )
        right, _ = simulate_accessible_observations(
            current,
            cold_contact_resistance=plus_values[0],
            cold_face_thermal_capacitance=plus_values[1],
            sensor_time_constant=plus_values[2],
            sampling_interval=fit_config.sampling_interval,
            dense_time_step=fit_config.dense_time_step,
        )
        left_map = _prediction_map(left)
        right_map = _prediction_map(right)
        derivatives = {name: [] for name in ACCESSIBLE_SENSOR_NAMES}
        for item in left.observations:
            key = (item.sensor_name, item.time)
            derivatives[item.sensor_name].append(
                (right_map[key] - left_map[key])
                / (2.0 * fit_config.finite_difference_step)
                / fit_config.noise_standard_deviation
            )
        centered = []
        for name in ACCESSIBLE_SENSOR_NAMES:
            mean_value = fmean(derivatives[name])
            centered.extend(value - mean_value for value in derivatives[name])
        columns.append(tuple(centered))
    jacobian = tuple(
        tuple(column[row] for column in columns)
        for row in range(len(columns[0]))
    )
    return gram_matrix(jacobian)


def _prior_information(
    config: AdaptiveCampaignConfig,
) -> Tuple[Tuple[float, ...], ...]:
    return tuple(
        tuple(
            1.0 / config.prior_log_standard_deviations[row] ** 2
            if row == column
            else 0.0
            for column in range(3)
        )
        for row in range(3)
    )


def _minimum_reserve(
    candidates: Sequence[ExperimentCandidateScore],
    excluded_names: Sequence[str],
    count: int,
) -> float:
    available = sorted(
        candidate.electrical_energy
        for candidate in candidates
        if candidate.name not in excluded_names
    )
    if len(available) < count:
        return math.inf
    return sum(available[:count])


def _choose_candidate(
    candidates: Sequence[ExperimentCandidateScore],
    *,
    used_names: Sequence[str],
    information: Sequence[Sequence[float]],
    log_multipliers: Tuple[float, float, float],
    remaining_energy: float,
    remaining_steps_after_choice: int,
    config: AdaptiveCampaignConfig,
) -> Tuple[ExperimentCandidateScore, Tuple[Tuple[float, ...], ...]]:
    _, old_determinant = inverse_and_determinant(information)
    scored = []
    for candidate in candidates:
        if candidate.name in used_names or candidate.electrical_energy > remaining_energy:
            continue
        reserve = _minimum_reserve(
            candidates,
            (*used_names, candidate.name),
            remaining_steps_after_choice,
        )
        if candidate.electrical_energy + reserve > remaining_energy + 1.0e-9:
            continue
        candidate_information = _candidate_information(
            candidate, log_multipliers, config
        )
        updated = matrix_add(information, candidate_information)
        _, updated_determinant = inverse_and_determinant(updated)
        gain = 0.5 * math.log(updated_determinant / old_determinant)
        scored.append((gain, -candidate.electrical_energy, candidate, updated))
    if not scored:
        raise ValueError("campaign budget cannot support the remaining experiments")
    _, _, selected, updated = max(scored, key=lambda item: (item[0], item[1]))
    return selected, updated


def _candidate_grid(
    config: AdaptiveCampaignConfig,
) -> Tuple[ExperimentCandidateScore, ...]:
    candidates = []
    for amplitude in config.selection.current_amplitudes:
        for duration in config.selection.pulse_durations:
            score, _ = score_experiment_candidate(
                amplitude, duration, config.selection
            )
            if score.feasible:
                candidates.append(score)
    return tuple(candidates)


def precommit_static_plan(
    candidates: Sequence[ExperimentCandidateScore],
    config: AdaptiveCampaignConfig,
) -> Tuple[ExperimentCandidateScore, ...]:
    """Greedily maximize cumulative nominal information before seeing data."""

    information = _prior_information(config)
    used = []
    selected = []
    spent = 0.0
    for step in range(config.experiment_count):
        candidate, information = _choose_candidate(
            candidates,
            used_names=used,
            information=information,
            log_multipliers=(0.0, 0.0, 0.0),
            remaining_energy=config.total_energy_budget - spent,
            remaining_steps_after_choice=config.experiment_count - step - 1,
            config=config,
        )
        selected.append(candidate)
        used.append(candidate.name)
        spent += candidate.electrical_energy
    return tuple(selected)


def _trial_truth(config: AdaptiveCampaignConfig, trial_index: int) -> JointThermalTruth:
    random_source = random.Random(config.first_seed + 10_000 * trial_index)
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
        cold_sensor_bias=0.0,
        hot_sensor_bias=0.0,
    )


def _ideal_truth_observations(
    truth_condition: str,
    current: PiecewiseConstantCurrent,
    truth: JointThermalTruth,
    config: AdaptiveCampaignConfig,
) -> ObservationDataset:
    if truth_condition == "matched_model":
        dataset, _ = simulate_accessible_observations(
            current,
            cold_contact_resistance=truth.cold_contact_resistance,
            cold_face_thermal_capacitance=truth.cold_face_thermal_capacitance,
            sensor_time_constant=truth.sensor_time_constant,
            sampling_interval=config.fit.sampling_interval,
            dense_time_step=config.fit.dense_time_step,
        )
        return dataset
    if truth_condition != "extra_interface_mass":
        raise ValueError(f"unknown truth condition: {truth_condition}")
    reference = constant_current_contact_reference_experiment()
    thermal = replace(
        reference.thermal_parameters,
        cold_contact_resistance=truth.cold_contact_resistance,
        cold_face_thermal_capacitance=truth.cold_face_thermal_capacitance,
    )
    dataset, _ = simulate_interface_mass_observations(
        current,
        thermoelectric_parameters=reference.thermoelectric_parameters,
        thermal_parameters=thermal,
        mismatch=config.mismatch,
        sensor_time_constant=truth.sensor_time_constant,
        sampling_interval=config.fit.sampling_interval,
        dense_time_step=config.fit.dense_time_step,
    )
    return dataset


def _candidate_observations(
    truth_condition: str,
    candidate: ExperimentCandidateScore,
    truth: JointThermalTruth,
    config: AdaptiveCampaignConfig,
    *,
    trial_index: int,
    candidate_index: int,
) -> CampaignObservation:
    current = candidate_current(candidate.current_amplitude, candidate.pulse_duration)
    ideal = _ideal_truth_observations(
        truth_condition, current, truth, config
    )
    condition_index = TRUTH_CONDITIONS.index(truth_condition)
    seed = (
        config.first_seed
        + condition_index * 1_000_000
        + trial_index * 1_000
        + candidate_index * 10
    )
    random_source = random.Random(seed)
    biased = apply_fixed_temperature_bias(
        ideal,
        FixedTemperatureBias(
            sensor_biases=(
                (
                    ACCESSIBLE_SENSOR_NAMES[0],
                    random_source.gauss(0.0, config.run_bias_standard_deviation),
                ),
                (
                    ACCESSIBLE_SENSOR_NAMES[1],
                    random_source.gauss(0.0, config.run_bias_standard_deviation),
                ),
            )
        ),
    ).dataset
    noisy = apply_gaussian_temperature_noise(
        biased,
        GaussianTemperatureNoise(
            default_standard_deviation=config.fit.noise_standard_deviation,
            random_seed=seed + 1,
        ),
    ).dataset
    return CampaignObservation(candidate.name, current, noisy)


def _rmse(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("RMSE inputs must be nonempty and aligned")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _withheld_prediction_metrics(
    truth_condition: str,
    truth: JointThermalTruth,
    fit: SequentialFitResult,
    config: AdaptiveCampaignConfig,
) -> Tuple[float, float, float]:
    current = sparse_withheld_current()
    if truth_condition == "matched_model":
        truth_dataset, truth_trajectory = simulate_accessible_observations(
            current,
            cold_contact_resistance=truth.cold_contact_resistance,
            cold_face_thermal_capacitance=truth.cold_face_thermal_capacitance,
            sensor_time_constant=truth.sensor_time_constant,
            sampling_interval=config.fit.sampling_interval,
            dense_time_step=config.fit.dense_time_step,
        )
    elif truth_condition == "extra_interface_mass":
        reference = constant_current_contact_reference_experiment()
        thermal = replace(
            reference.thermal_parameters,
            cold_contact_resistance=truth.cold_contact_resistance,
            cold_face_thermal_capacitance=truth.cold_face_thermal_capacitance,
        )
        truth_dataset, truth_trajectory = simulate_interface_mass_observations(
            current,
            thermoelectric_parameters=reference.thermoelectric_parameters,
            thermal_parameters=thermal,
            mismatch=config.mismatch,
            sensor_time_constant=truth.sensor_time_constant,
            sampling_interval=config.fit.sampling_interval,
            dense_time_step=config.fit.dense_time_step,
        )
    else:
        raise ValueError(f"unknown truth condition: {truth_condition}")
    prediction, prediction_trajectory = simulate_accessible_observations(
        current,
        cold_contact_resistance=fit.physical_values[0],
        cold_face_thermal_capacitance=fit.physical_values[1],
        sensor_time_constant=fit.physical_values[2],
        sampling_interval=config.fit.sampling_interval,
        dense_time_step=config.fit.dense_time_step,
    )
    truth_map = _prediction_map(truth_dataset)
    prediction_map = _prediction_map(prediction)
    residual_by_key = {
        key: prediction_map[key] - truth_map[key]
        for key in truth_map
    }
    rmse = _rmse(tuple(residual_by_key.values()), (0.0,) * len(residual_by_key))
    hidden_face_rmse = _rmse(
        prediction_trajectory.cold_face,
        truth_trajectory.cold_face,
    )
    window_biases = []
    for transition in current.transition_times:
        for sensor_name in ACCESSIBLE_SENSOR_NAMES:
            values = tuple(
                residual
                for (name, time), residual in residual_by_key.items()
                if name == sensor_name and transition <= time <= transition + 3.0
            )
            if values:
                window_biases.append(abs(fmean(values)))
    return rmse, hidden_face_rmse, max(window_biases, default=0.0)


def _step_result(
    *,
    truth_condition: str,
    strategy: str,
    trial_index: int,
    step_index: int,
    candidate: ExperimentCandidateScore,
    cumulative_energy: float,
    truth: JointThermalTruth,
    fit: SequentialFitResult,
    config: AdaptiveCampaignConfig,
) -> CampaignStepResult:
    log_errors = tuple(
        math.log(estimate / expected)
        for estimate, expected in zip(fit.physical_values, truth.physical_values)
    )
    heldout_rmse, hidden_face_rmse, switch_bias = _withheld_prediction_metrics(
        truth_condition, truth, fit, config
    )
    parameter_log_rmse = math.sqrt(
        sum(value * value for value in log_errors) / len(log_errors)
    )
    parameter_recovery_passed = (
        parameter_log_rmse <= config.parameter_log_rmse_threshold
    )
    prediction_passed = (
        heldout_rmse <= config.prediction_rmse_threshold
        and hidden_face_rmse <= config.hidden_face_rmse_threshold
        and switch_bias <= config.switch_bias_threshold
    )
    confident = max(fit.log_standard_errors) <= config.confidence_log_standard_error
    return CampaignStepResult(
        truth_condition=truth_condition,
        strategy=strategy,
        trial_index=trial_index,
        step_index=step_index,
        candidate_name=candidate.name,
        candidate_energy=candidate.electrical_energy,
        cumulative_energy=cumulative_energy,
        physical_values=fit.physical_values,
        parameter_log_rmse=parameter_log_rmse,
        observation_rmse=fit.observation_rmse,
        uncertainty_volume=fit.uncertainty_volume,
        maximum_log_standard_error=max(fit.log_standard_errors),
        heldout_prediction_rmse=heldout_rmse,
        heldout_hidden_face_rmse=hidden_face_rmse,
        heldout_switch_bias=switch_bias,
        parameter_recovery_passed=parameter_recovery_passed,
        prediction_passed=prediction_passed,
        confident=confident,
        false_confidence=(
            confident and (not prediction_passed or not parameter_recovery_passed)
        ),
        reached_bound=fit.reached_bound,
    )


def _run_fixed_strategy(
    *,
    truth_condition: str,
    strategy: str,
    trial_index: int,
    truth: JointThermalTruth,
    plan: Sequence[ExperimentCandidateScore],
    observations_for: Callable[[ExperimentCandidateScore], CampaignObservation],
    fit_for: Callable[
        [Sequence[CampaignObservation], Optional[SequentialFitResult]],
        SequentialFitResult,
    ],
    config: AdaptiveCampaignConfig,
) -> Tuple[CampaignStepResult, ...]:
    runs = []
    results = []
    cumulative_energy = 0.0
    previous_fit = None
    for step_index, candidate in enumerate(plan, start=1):
        runs.append(observations_for(candidate))
        fit = fit_for(runs, previous_fit)
        cumulative_energy += candidate.electrical_energy
        results.append(
            _step_result(
                truth_condition=truth_condition,
                strategy=strategy,
                trial_index=trial_index,
                step_index=step_index,
                candidate=candidate,
                cumulative_energy=cumulative_energy,
                truth=truth,
                fit=fit,
                config=config,
            )
        )
        previous_fit = fit
    return tuple(results)


def _run_adaptive_strategy(
    *,
    truth_condition: str,
    trial_index: int,
    truth: JointThermalTruth,
    candidates: Sequence[ExperimentCandidateScore],
    observations_for: Callable[[ExperimentCandidateScore], CampaignObservation],
    fit_for: Callable[
        [Sequence[CampaignObservation], Optional[SequentialFitResult]],
        SequentialFitResult,
    ],
    config: AdaptiveCampaignConfig,
) -> Tuple[CampaignStepResult, ...]:
    runs = []
    results = []
    used = []
    cumulative_energy = 0.0
    information = _prior_information(config)
    log_multipliers = (0.0, 0.0, 0.0)
    previous_fit = None
    for step_index in range(1, config.experiment_count + 1):
        candidate, _ = _choose_candidate(
            candidates,
            used_names=used,
            information=information,
            log_multipliers=log_multipliers,
            remaining_energy=config.total_energy_budget - cumulative_energy,
            remaining_steps_after_choice=config.experiment_count - step_index,
            config=config,
        )
        runs.append(observations_for(candidate))
        fit = fit_for(runs, previous_fit)
        cumulative_energy += candidate.electrical_energy
        results.append(
            _step_result(
                truth_condition=truth_condition,
                strategy="adaptive",
                trial_index=trial_index,
                step_index=step_index,
                candidate=candidate,
                cumulative_energy=cumulative_energy,
                truth=truth,
                fit=fit,
                config=config,
            )
        )
        used.append(candidate.name)
        log_multipliers = fit.log_multipliers
        information, _ = inverse_and_determinant(fit.covariance)
        previous_fit = fit
    return tuple(results)


def summarize_campaign_steps(
    steps: Sequence[CampaignStepResult],
    config: AdaptiveCampaignConfig,
) -> Tuple[CampaignStepSummary, ...]:
    summaries = []
    for condition in TRUTH_CONDITIONS:
        for strategy in CAMPAIGN_STRATEGIES:
            for step_index in range(1, config.experiment_count + 1):
                selected = tuple(
                    item
                    for item in steps
                    if item.truth_condition == condition
                    and item.strategy == strategy
                    and item.step_index == step_index
                )
                if len(selected) != config.trial_count:
                    raise ValueError("each campaign summary needs every paired trial")
                summaries.append(
                    CampaignStepSummary(
                        truth_condition=condition,
                        strategy=strategy,
                        step_index=step_index,
                        trial_count=len(selected),
                        mean_cumulative_energy=fmean(
                            item.cumulative_energy for item in selected
                        ),
                        mean_parameter_log_rmse=fmean(
                            item.parameter_log_rmse for item in selected
                        ),
                        mean_observation_rmse=fmean(
                            item.observation_rmse for item in selected
                        ),
                        median_uncertainty_volume=median(
                            item.uncertainty_volume for item in selected
                        ),
                        mean_heldout_prediction_rmse=fmean(
                            item.heldout_prediction_rmse for item in selected
                        ),
                        mean_heldout_hidden_face_rmse=fmean(
                            item.heldout_hidden_face_rmse for item in selected
                        ),
                        parameter_recovery_pass_rate=fmean(
                            float(item.parameter_recovery_passed)
                            for item in selected
                        ),
                        prediction_pass_rate=fmean(
                            float(item.prediction_passed) for item in selected
                        ),
                        confidence_rate=fmean(
                            float(item.confident) for item in selected
                        ),
                        false_confidence_rate=fmean(
                            float(item.false_confidence) for item in selected
                        ),
                        bound_hit_rate=fmean(
                            float(item.reached_bound) for item in selected
                        ),
                    )
                )
    return tuple(summaries)


def summarize_strategy_outcomes(
    steps: Sequence[CampaignStepResult],
    config: AdaptiveCampaignConfig,
) -> Tuple[CampaignStrategyOutcome, ...]:
    outcomes = []
    for condition in TRUTH_CONDITIONS:
        for strategy in CAMPAIGN_STRATEGIES:
            first_steps = []
            first_energies = []
            never_passed = 0
            final_passed = 0
            final_false_confidence = 0
            for trial_index in range(config.trial_count):
                trial = tuple(
                    item
                    for item in steps
                    if item.truth_condition == condition
                    and item.strategy == strategy
                    and item.trial_index == trial_index
                )
                passed = tuple(item for item in trial if item.prediction_passed)
                if passed:
                    first = min(passed, key=lambda item: item.step_index)
                    first_steps.append(float(first.step_index))
                    first_energies.append(first.cumulative_energy)
                else:
                    never_passed += 1
                final = max(trial, key=lambda item: item.step_index)
                final_passed += int(final.prediction_passed)
                final_false_confidence += int(final.false_confidence)
            outcomes.append(
                CampaignStrategyOutcome(
                    truth_condition=condition,
                    strategy=strategy,
                    median_first_passing_step=(
                        median(first_steps) if first_steps else math.inf
                    ),
                    median_energy_to_prediction_gate=(
                        median(first_energies) if first_energies else math.inf
                    ),
                    never_passed_rate=never_passed / config.trial_count,
                    final_prediction_pass_rate=final_passed / config.trial_count,
                    final_false_confidence_rate=(
                        final_false_confidence / config.trial_count
                    ),
                )
            )
    return tuple(outcomes)


def run_adaptive_experiment_campaign(
    config: AdaptiveCampaignConfig = AdaptiveCampaignConfig(),
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> AdaptiveExperimentCampaignResult:
    """Run every paired strategy under both model-adequacy conditions."""

    candidates = _candidate_grid(config)
    if len(candidates) < config.experiment_count:
        raise ValueError("candidate grid is smaller than the requested campaign")
    by_name = {candidate.name: candidate for candidate in candidates}
    missing_heuristic = set(config.heuristic_candidate_names) - set(by_name)
    if missing_heuristic:
        raise ValueError(
            "heuristic plan references unavailable candidates: "
            + ", ".join(sorted(missing_heuristic))
        )
    heuristic_plan = tuple(
        by_name[name] for name in config.heuristic_candidate_names
    )
    if sum(item.electrical_energy for item in heuristic_plan) > config.total_energy_budget:
        raise ValueError("heuristic plan exceeds the campaign energy budget")
    static_plan = precommit_static_plan(candidates, config)
    candidate_index = {
        candidate.name: index for index, candidate in enumerate(candidates)
    }

    all_steps = []
    for condition in TRUTH_CONDITIONS:
        for trial_index in range(config.trial_count):
            if progress is not None:
                progress(
                    f"{condition}: paired trial {trial_index + 1}/{config.trial_count}"
                )
            truth = _trial_truth(config, trial_index)
            observation_cache = {}
            fit_cache = {}

            def observations_for(candidate: ExperimentCandidateScore):
                if candidate.name not in observation_cache:
                    observation_cache[candidate.name] = _candidate_observations(
                        condition,
                        candidate,
                        truth,
                        config,
                        trial_index=trial_index,
                        candidate_index=candidate_index[candidate.name],
                    )
                return observation_cache[candidate.name]

            def fit_for(
                runs: Sequence[CampaignObservation],
                previous_fit: Optional[SequentialFitResult],
            ) -> SequentialFitResult:
                key = tuple(sorted(run.candidate_name for run in runs))
                if key not in fit_cache:
                    fit_cache[key] = fit_sequential_campaign(
                        runs,
                        config,
                        initial_log_multipliers=(
                            None
                            if previous_fit is None
                            else previous_fit.log_multipliers
                        ),
                    )
                return fit_cache[key]

            all_steps.extend(
                _run_adaptive_strategy(
                    truth_condition=condition,
                    trial_index=trial_index,
                    truth=truth,
                    candidates=candidates,
                    observations_for=observations_for,
                    fit_for=fit_for,
                    config=config,
                )
            )
            all_steps.extend(
                _run_fixed_strategy(
                    truth_condition=condition,
                    strategy="static_d_optimal",
                    trial_index=trial_index,
                    truth=truth,
                    plan=static_plan,
                    observations_for=observations_for,
                    fit_for=fit_for,
                    config=config,
                )
            )
            all_steps.extend(
                _run_fixed_strategy(
                    truth_condition=condition,
                    strategy="engineer_heuristic",
                    trial_index=trial_index,
                    truth=truth,
                    plan=heuristic_plan,
                    observations_for=observations_for,
                    fit_for=fit_for,
                    config=config,
                )
            )
    steps = tuple(all_steps)
    return AdaptiveExperimentCampaignResult(
        config=config,
        candidates=candidates,
        static_plan=tuple(item.name for item in static_plan),
        heuristic_plan=config.heuristic_candidate_names,
        steps=steps,
        summaries=summarize_campaign_steps(steps, config),
        outcomes=summarize_strategy_outcomes(steps, config),
    )


__all__ = [
    "AdaptiveCampaignConfig",
    "AdaptiveExperimentCampaignResult",
    "CampaignObservation",
    "CampaignStepResult",
    "CampaignStepSummary",
    "CampaignStrategyOutcome",
    "SequentialFitResult",
    "fit_sequential_campaign",
    "precommit_static_plan",
    "run_adaptive_experiment_campaign",
    "summarize_campaign_steps",
    "summarize_strategy_outcomes",
]
