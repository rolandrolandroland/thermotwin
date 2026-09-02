"""Inverse contact-resistance PINN under noise, missingness, and bias."""

from dataclasses import dataclass
import math
from statistics import fmean
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

from ..inference.contact_resistance import ContactResistanceRegimeDataset
from ..observations.bias import FixedTemperatureBias, apply_fixed_temperature_bias
from ..observations.missingness import apply_deterministic_temperature_missingness
from ..observations.noise import GaussianTemperatureNoise, apply_gaussian_temperature_noise
from ..pinn.inverse_contact_resistance import cold_contact_observations_from_dataset
from ..pinn.inverse_piecewise_contact_resistance import (
    PiecewiseInverseContactResistanceConfig,
    PiecewiseInverseContactResistanceValidation,
    ideal_piecewise_inverse_contact_problem,
    train_piecewise_inverse_contact_resistance,
    validate_piecewise_inverse_contact_resistance,
)
from .contact_resistance.missingness import (
    ContactResistanceMissingnessCase,
    missingness_for_contact_resistance_case,
)


@dataclass(frozen=True)
class ImperfectInversePINNCriteria:
    minimum_loss_reduction_fraction: float = 0.80
    maximum_absolute_parameter_error: float = 0.03
    maximum_withheld_all_sensor_rmse: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_loss_reduction_fraction < 1.0:
            raise ValueError("loss-reduction fraction must lie strictly from zero to one")
        for name, value in (
            ("maximum parameter error", self.maximum_absolute_parameter_error),
            ("maximum withheld RMSE", self.maximum_withheld_all_sensor_rmse),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


class ImperfectInversePINNCase(NamedTuple):
    name: str
    noise_standard_deviation: float
    turnoff_half_width: float | None
    cold_face_bias: float
    expected_recovery: bool


@dataclass(frozen=True)
class ImperfectInversePINNConfig:
    trial_count_per_case: int = 3
    first_seed: int = 61001
    initial_resistances: Tuple[float, ...] = (0.15, 0.50, 0.80)
    criteria: ImperfectInversePINNCriteria = ImperfectInversePINNCriteria()
    training_epochs: int = 2_500
    hidden_width: int = 16
    hidden_layers: int = 2
    collocation_points: int = 48
    network_learning_rate: float = 2.0e-3
    parameter_learning_rate: float = 1.0e-2
    observation_weight: float = 20.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.trial_count_per_case, int)
            or isinstance(self.trial_count_per_case, bool)
            or self.trial_count_per_case <= 0
        ):
            raise ValueError("trial count must be a positive integer")
        if (
            not isinstance(self.first_seed, int)
            or isinstance(self.first_seed, bool)
            or self.first_seed < 0
        ):
            raise ValueError("first seed must be a nonnegative integer")
        if len(self.initial_resistances) < self.trial_count_per_case or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.initial_resistances
        ):
            raise ValueError("one finite positive initial resistance is required per trial")
        for name, value in (
            ("training epochs", self.training_epochs),
            ("hidden width", self.hidden_width),
            ("hidden layers", self.hidden_layers),
            ("collocation points", self.collocation_points),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("network learning rate", self.network_learning_rate),
            ("parameter learning rate", self.parameter_learning_rate),
            ("observation weight", self.observation_weight),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


class ImperfectInversePINNTrial(NamedTuple):
    case_name: str
    trial_index: int
    observation_seed: int
    neural_seed: int
    initial_resistance: float
    validation: PiecewiseInverseContactResistanceValidation
    initial_total_loss: float
    final_total_loss: float
    loss_reduction_fraction: float
    operational_success: bool
    recovery_success: bool
    failure_reasons: Tuple[str, ...]


class ImperfectInversePINNSummary(NamedTuple):
    case_name: str
    expected_recovery: bool
    trial_count: int
    operational_success_count: int
    recovery_success_count: int
    mean_pinn_resistance: float
    mean_conventional_resistance: float
    pinn_parameter_rmse: float
    conventional_parameter_rmse: float
    worst_pinn_absolute_parameter_error: float
    mean_validation_all_sensor_rmse: float
    mean_test_all_sensor_rmse: float


class ImperfectInversePINNStudyResult(NamedTuple):
    config: ImperfectInversePINNConfig
    cases: Tuple[ImperfectInversePINNCase, ...]
    trials: Tuple[ImperfectInversePINNTrial, ...]
    summaries: Tuple[ImperfectInversePINNSummary, ...]


def imperfect_inverse_pinn_cases() -> Tuple[ImperfectInversePINNCase, ...]:
    """Return frozen recovery cases plus one intentional mismatch case."""

    return (
        ImperfectInversePINNCase("gaussian_noise", 0.02, None, 0.0, True),
        ImperfectInversePINNCase("turnoff_missingness", 0.0, 2.0, 0.0, True),
        ImperfectInversePINNCase(
            "noise_plus_turnoff_missingness", 0.02, 2.0, 0.0, True
        ),
        ImperfectInversePINNCase(
            "unmodeled_cold_face_bias", 0.02, None, 0.10, False
        ),
    )


def imperfect_inverse_pinn_seeds(
    first_seed: int,
    case_index: int,
    trial_index: int,
) -> Tuple[int, int]:
    if min(first_seed, case_index, trial_index) < 0:
        raise ValueError("seed coordinates must be nonnegative")
    base = first_seed + 10_000 * case_index + 2 * trial_index
    return base, base + 1


def build_imperfect_inverse_pinn_problem(
    case: ImperfectInversePINNCase,
    observation_seed: int,
):
    problem = ideal_piecewise_inverse_contact_problem(observation_interval=1.0)
    dataset = problem.dataset.observations
    if case.cold_face_bias != 0.0:
        dataset = apply_fixed_temperature_bias(
            dataset,
            FixedTemperatureBias(
                sensor_biases=(("cold_face_sensor", case.cold_face_bias),)
            ),
        ).dataset
    if case.noise_standard_deviation > 0.0:
        dataset = apply_gaussian_temperature_noise(
            dataset,
            GaussianTemperatureNoise(
                default_standard_deviation=case.noise_standard_deviation,
                random_seed=observation_seed,
            ),
        ).dataset
    transformed = ContactResistanceRegimeDataset(
        regime=problem.dataset.regime,
        observations=dataset,
    )
    if case.turnoff_half_width is not None:
        missingness = missingness_for_contact_resistance_case(
            ContactResistanceMissingnessCase(
                case.name,
                case.turnoff_half_width,
            ),
            transformed,
        )
        transformed = ContactResistanceRegimeDataset(
            regime=transformed.regime,
            observations=apply_deterministic_temperature_missingness(
                transformed.observations,
                missingness,
            ).dataset,
        )
    return problem._replace(
        dataset=transformed,
        observations=cold_contact_observations_from_dataset(transformed),
    )


def _loss_reduction(initial: float, final: float) -> float:
    if initial <= 0.0:
        return 0.0
    return 1.0 - final / initial


def _failure_reasons(
    validation: PiecewiseInverseContactResistanceValidation,
    loss_reduction: float,
    criteria: ImperfectInversePINNCriteria,
) -> Tuple[str, ...]:
    reasons = []
    if loss_reduction < criteria.minimum_loss_reduction_fraction:
        reasons.append("loss_reduction")
    if validation.absolute_parameter_error > criteria.maximum_absolute_parameter_error:
        reasons.append("parameter_error")
    if (
        validation.validation_regime_metrics.all_sensor_rmse
        > criteria.maximum_withheld_all_sensor_rmse
    ):
        reasons.append("validation_transfer")
    if (
        validation.test_regime_metrics.all_sensor_rmse
        > criteria.maximum_withheld_all_sensor_rmse
    ):
        reasons.append("test_transfer")
    return tuple(reasons)


def run_imperfect_inverse_pinn_study(
    config: ImperfectInversePINNConfig = ImperfectInversePINNConfig(),
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> ImperfectInversePINNStudyResult:
    cases = imperfect_inverse_pinn_cases()
    trials = []
    for case_index, case in enumerate(cases):
        for trial_index in range(config.trial_count_per_case):
            if progress is not None:
                progress(
                    f"{case.name}: trial {trial_index + 1}/"
                    f"{config.trial_count_per_case}"
                )
            observation_seed, neural_seed = imperfect_inverse_pinn_seeds(
                config.first_seed, case_index, trial_index
            )
            problem = build_imperfect_inverse_pinn_problem(case, observation_seed)
            training = train_piecewise_inverse_contact_resistance(
                problem,
                PiecewiseInverseContactResistanceConfig(
                    hidden_width=config.hidden_width,
                    hidden_layers=config.hidden_layers,
                    collocation_points=config.collocation_points,
                    epochs=config.training_epochs,
                    network_learning_rate=config.network_learning_rate,
                    parameter_learning_rate=config.parameter_learning_rate,
                    initial_cold_contact_resistance=(
                        config.initial_resistances[trial_index]
                    ),
                    observation_weight=config.observation_weight,
                    seed=neural_seed,
                    device="cpu",
                ),
            )
            validation = validate_piecewise_inverse_contact_resistance(
                training, problem
            )
            reduction = _loss_reduction(
                training.history.total_loss[0],
                training.history.total_loss[-1],
            )
            reasons = _failure_reasons(validation, reduction, config.criteria)
            operational_success = reduction >= config.criteria.minimum_loss_reduction_fraction
            trials.append(
                ImperfectInversePINNTrial(
                    case_name=case.name,
                    trial_index=trial_index,
                    observation_seed=observation_seed,
                    neural_seed=neural_seed,
                    initial_resistance=config.initial_resistances[trial_index],
                    validation=validation,
                    initial_total_loss=training.history.total_loss[0],
                    final_total_loss=training.history.total_loss[-1],
                    loss_reduction_fraction=reduction,
                    operational_success=operational_success,
                    recovery_success=not reasons,
                    failure_reasons=reasons,
                )
            )
    summaries = summarize_imperfect_inverse_pinn_trials(trials, cases)
    return ImperfectInversePINNStudyResult(
        config=config,
        cases=cases,
        trials=tuple(trials),
        summaries=summaries,
    )


def summarize_imperfect_inverse_pinn_trials(
    trials: Sequence[ImperfectInversePINNTrial],
    cases: Sequence[ImperfectInversePINNCase],
) -> Tuple[ImperfectInversePINNSummary, ...]:
    trials = tuple(trials)
    summaries = []
    truth = 0.25
    for case in cases:
        selected = tuple(trial for trial in trials if trial.case_name == case.name)
        if not selected:
            raise ValueError("every imperfect-data case needs at least one trial")
        pinn_values = tuple(
            trial.validation.inferred_cold_contact_resistance for trial in selected
        )
        conventional_values = tuple(
            trial.validation.conventional_cold_contact_resistance for trial in selected
        )
        summaries.append(
            ImperfectInversePINNSummary(
                case_name=case.name,
                expected_recovery=case.expected_recovery,
                trial_count=len(selected),
                operational_success_count=sum(
                    trial.operational_success for trial in selected
                ),
                recovery_success_count=sum(
                    trial.recovery_success for trial in selected
                ),
                mean_pinn_resistance=fmean(pinn_values),
                mean_conventional_resistance=fmean(conventional_values),
                pinn_parameter_rmse=math.sqrt(
                    fmean((value - truth) ** 2 for value in pinn_values)
                ),
                conventional_parameter_rmse=math.sqrt(
                    fmean((value - truth) ** 2 for value in conventional_values)
                ),
                worst_pinn_absolute_parameter_error=max(
                    abs(value - truth) for value in pinn_values
                ),
                mean_validation_all_sensor_rmse=fmean(
                    trial.validation.validation_regime_metrics.all_sensor_rmse
                    for trial in selected
                ),
                mean_test_all_sensor_rmse=fmean(
                    trial.validation.test_regime_metrics.all_sensor_rmse
                    for trial in selected
                ),
            )
        )
    return tuple(summaries)
