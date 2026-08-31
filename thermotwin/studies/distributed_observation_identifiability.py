"""Observation ablations that must reject unsupported resistivity curves."""

from dataclasses import dataclass, replace
import math
from statistics import fmean
from typing import NamedTuple, Optional, Sequence, Tuple

from ..inference.distributed_identifiability import (
    DistributedIdentifiabilityAssessment,
    DistributedIdentifiabilityConfig,
    DistributedIdentifiabilityGateConfig,
    DistributedIdentifiabilityResult,
    DistributedPropertyCoefficient,
    analyze_distributed_identifiability,
    assess_distributed_identifiability,
)
from ..inference.distributed_properties import (
    DistributedPropertyFitConfig,
    fit_distributed_property,
)
from ..observations.distributed import (
    DistributedObservationChannels,
    DistributedObservationSet,
    add_distributed_gaussian_noise,
)
from ..physics.distributed import PiecewiseLinearProperty
from ..simulation.distributed import (
    DistributedLegExperiment,
    distributed_inverse_constant_experiments,
    run_distributed_leg_experiment,
)
from ..simulation.distributed_independent import (
    PolynomialTemperatureProperty,
    observe_independent_distributed_result,
    run_independent_distributed_experiment,
)
from .distributed_independent_validation import (
    DistributedMismatchPredictionMetrics,
    evaluate_independent_prediction,
    smooth_resistivity_truth,
)


@dataclass(frozen=True)
class DistributedObservationIdentifiabilityConfig:
    """Frozen truth, sensor, fitting, and practical-rank assumptions."""

    truth_node_count: int = 25
    truth_time_step: float = 2.5e-4
    observation_interval: float = 0.08
    temperature_standard_deviation: float = 0.01
    voltage_standard_deviation: float = 1.0e-5
    first_noise_seed: int = 48_001
    neural_seed: int = 48_101
    inverse_pinn_epochs: int = 500
    log_multiplier_bounds: Tuple[float, float] = (-0.3, 0.3)
    initial_log_multiplier_sets: Tuple[Tuple[float, ...], ...] = (
        (math.log(0.8),) * 3,
        (0.0,) * 3,
        (math.log(1.2),) * 3,
    )
    gate: DistributedIdentifiabilityGateConfig = (
        DistributedIdentifiabilityGateConfig()
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("truth node count", self.truth_node_count),
            ("first noise seed", self.first_noise_seed),
            ("neural seed", self.neural_seed),
            ("inverse PINN epochs", self.inverse_pinn_epochs),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.truth_node_count < 5 or self.inverse_pinn_epochs <= 0:
            raise ValueError("truth nodes must be at least five and epochs positive")
        if self.first_noise_seed < 0 or self.neural_seed < 0:
            raise ValueError("seeds must be nonnegative")
        for name, value in (
            ("truth time step", self.truth_time_step),
            ("observation interval", self.observation_interval),
            (
                "temperature standard deviation",
                self.temperature_standard_deviation,
            ),
            ("voltage standard deviation", self.voltage_standard_deviation),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        lower, upper = self.log_multiplier_bounds
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("log-multiplier bounds must be finite and ordered")
        if len(self.initial_log_multiplier_sets) < 2:
            raise ValueError("at least two initial curves are required")
        for values in self.initial_log_multiplier_sets:
            if len(values) != 3:
                raise ValueError("each initial curve must have three coefficients")
            if any(
                not math.isfinite(value) or value < lower or value > upper
                for value in values
            ):
                raise ValueError("initial coefficients must be finite and in bounds")
        if not isinstance(self.gate, DistributedIdentifiabilityGateConfig):
            raise ValueError("gate must be a distributed identifiability gate")


class DistributedObservationCaseDefinition(NamedTuple):
    name: str
    experiment_indices: Tuple[int, ...]
    channels: DistributedObservationChannels
    scientific_question: str


class DistributedObservationFitResult(NamedTuple):
    estimator: str
    start_index: int
    initial_multipliers: Tuple[float, ...]
    fitted_multipliers: Optional[Tuple[float, ...]]
    normalized_observation_loss: float
    property_relative_rmse: float
    property_maximum_relative_error: float
    holdout_internal_temperature_rmse: float
    holdout_voltage_rmse: float
    permitted_by_prefit_gate: bool


class DistributedObservationCaseResult(NamedTuple):
    definition: DistributedObservationCaseDefinition
    identifiability: DistributedIdentifiabilityResult
    assessment: DistributedIdentifiabilityAssessment
    fit_policy: str
    fits: Tuple[DistributedObservationFitResult, ...]


class DistributedObservationIdentifiabilityStudyResult(NamedTuple):
    config: DistributedObservationIdentifiabilityConfig
    truth_property: PolynomialTemperatureProperty
    truth_knot_multipliers: Tuple[float, ...]
    holdout_name: str
    cases: Tuple[DistributedObservationCaseResult, ...]


def distributed_observation_case_definitions(
) -> Tuple[DistributedObservationCaseDefinition, ...]:
    """Return the predeclared full, structural, and weak observation cases."""

    full = DistributedObservationChannels()
    temperature_only = DistributedObservationChannels(voltage=False)
    return (
        DistributedObservationCaseDefinition(
            "full_bidirectional",
            (0, 1, 2),
            full,
            "Do zero, positive, and negative current with face temperatures and "
            "voltage support all three resistivity coefficients?",
        ),
        DistributedObservationCaseDefinition(
            "zero_current_only",
            (0,),
            full,
            "Does a passive relaxation contain any electrical-resistivity information?",
        ),
        DistributedObservationCaseDefinition(
            "positive_temperature_only",
            (1,),
            temperature_only,
            "Can Joule-heating changes in two face temperatures resolve the curve?",
        ),
        DistributedObservationCaseDefinition(
            "positive_temperature_voltage",
            (1,),
            full,
            "Does voltage identify curve shape, or mainly an average resistance?",
        ),
    )


def _truth_experiment(
    experiment: DistributedLegExperiment,
    truth_property: PolynomialTemperatureProperty,
) -> DistributedLegExperiment:
    return replace(
        experiment,
        material=replace(
            experiment.material,
            electrical_resistivity=truth_property,
        ),
    )


def _select_channels(
    observations: DistributedObservationSet,
    channels: DistributedObservationChannels,
) -> DistributedObservationSet:
    selected = set(channels.names())
    return DistributedObservationSet(
        tuple(item for item in observations.observations if item.channel in selected)
    )


def _property_errors(
    baseline: PiecewiseLinearProperty,
    multipliers: Sequence[float],
    truth: PolynomialTemperatureProperty,
) -> Tuple[float, float]:
    fitted = baseline.with_values(
        tuple(value * multiplier for value, multiplier in zip(baseline.values, multipliers))
    )
    temperatures = tuple(285.0 + 0.5 * index for index in range(61))
    errors = tuple(
        (fitted.value(temperature) - truth.value(temperature))
        / truth.value(temperature)
        for temperature in temperatures
    )
    return (
        math.sqrt(fmean(value * value for value in errors)),
        max(abs(value) for value in errors),
    )


def _holdout_metrics(
    baseline: PiecewiseLinearProperty,
    multipliers: Sequence[float],
    holdout: DistributedLegExperiment,
    truth_experiment: DistributedLegExperiment,
    truth_result,
) -> DistributedMismatchPredictionMetrics:
    fitted_property = baseline.with_values(
        tuple(value * multiplier for value, multiplier in zip(baseline.values, multipliers))
    )
    fitted_experiment = replace(
        holdout,
        material=replace(holdout.material, electrical_resistivity=fitted_property),
    )
    return evaluate_independent_prediction(
        fitted_experiment,
        run_distributed_leg_experiment(fitted_experiment),
        truth_experiment,
        truth_result,
    )


def _fit_result(
    *,
    estimator: str,
    start_index: int,
    initial_logs: Sequence[float],
    fitted_multipliers: Optional[Sequence[float]],
    loss: float,
    accepted: bool,
    baseline_property: PiecewiseLinearProperty,
    truth_property: PolynomialTemperatureProperty,
    holdout: DistributedLegExperiment,
    holdout_truth_experiment: DistributedLegExperiment,
    holdout_truth_result,
) -> DistributedObservationFitResult:
    initial_multipliers = tuple(math.exp(value) for value in initial_logs)
    if fitted_multipliers is None:
        return DistributedObservationFitResult(
            estimator,
            start_index,
            initial_multipliers,
            None,
            math.inf,
            math.inf,
            math.inf,
            math.inf,
            math.inf,
            False,
        )
    multipliers = tuple(float(value) for value in fitted_multipliers)
    property_rmse, property_maximum = _property_errors(
        baseline_property, multipliers, truth_property
    )
    metrics = _holdout_metrics(
        baseline_property,
        multipliers,
        holdout,
        holdout_truth_experiment,
        holdout_truth_result,
    )
    return DistributedObservationFitResult(
        estimator=estimator,
        start_index=start_index,
        initial_multipliers=initial_multipliers,
        fitted_multipliers=multipliers,
        normalized_observation_loss=float(loss),
        property_relative_rmse=property_rmse,
        property_maximum_relative_error=property_maximum,
        holdout_internal_temperature_rmse=metrics.internal_temperature_rmse,
        holdout_voltage_rmse=metrics.voltage_rmse,
        permitted_by_prefit_gate=accepted,
    )


def _fit_conventional(
    experiments: Sequence[DistributedLegExperiment],
    observations: Sequence[DistributedObservationSet],
    channels: DistributedObservationChannels,
    initial_logs: Sequence[float],
    config: DistributedObservationIdentifiabilityConfig,
):
    return fit_distributed_property(
        experiments,
        observations,
        DistributedPropertyFitConfig(
            property_name="electrical_resistivity",
            observation_interval=config.observation_interval,
            channels=channels,
            initial_log_multipliers=tuple(initial_logs),
            log_multiplier_bounds=config.log_multiplier_bounds,
            coordinate_passes=2,
            golden_section_iterations=8,
            gauss_newton_iterations=6,
            temperature_standard_deviation=config.temperature_standard_deviation,
            voltage_standard_deviation=config.voltage_standard_deviation,
            smoothness_weight=0.0,
        ),
    )


def _fit_pinn(
    experiments: Sequence[DistributedLegExperiment],
    observations: Sequence[DistributedObservationSet],
    initial_logs: Sequence[float],
    config: DistributedObservationIdentifiabilityConfig,
    *,
    seed: int,
) -> Tuple[Optional[Tuple[float, ...]], float]:
    try:
        from ..pinn.distributed_inverse import (
            InverseDistributedPropertyConfig,
            train_multi_experiment_inverse_distributed_property_pinn,
        )

        training = train_multi_experiment_inverse_distributed_property_pinn(
            experiments,
            observations,
            InverseDistributedPropertyConfig(
                property_name="electrical_resistivity",
                hidden_width=20,
                hidden_layers=3,
                interior_space_points=7,
                time_points=18,
                voltage_space_points=16,
                epochs=config.inverse_pinn_epochs,
                network_learning_rate=2.0e-3,
                property_learning_rate=2.0e-3,
                initial_log_multipliers=tuple(initial_logs),
                smoothness_weight=0.0,
                temperature_standard_deviation=(
                    config.temperature_standard_deviation
                ),
                voltage_standard_deviation=config.voltage_standard_deviation,
                seed=seed,
                device="cpu",
            ),
            baseline_material=experiments[0].material,
        )
        baseline = experiments[0].material.electrical_resistivity
        if not isinstance(baseline, PiecewiseLinearProperty):
            raise TypeError("PINN baseline resistivity must be piecewise linear")
        multipliers = tuple(
            value / base
            for value, base in zip(
                training.history.property_values[-1], baseline.values
            )
        )
        return multipliers, training.history.observation_loss[-1]
    except (FloatingPointError, RuntimeError):
        return None, math.inf


def estimator_coefficient_spread(
    case: DistributedObservationCaseResult,
    estimator: str,
) -> float:
    """Return the largest multiplier range across successful multistart fits."""

    curves = tuple(
        fit.fitted_multipliers
        for fit in case.fits
        if fit.estimator == estimator and fit.fitted_multipliers is not None
    )
    if len(curves) < 2:
        return math.inf
    return max(max(values) - min(values) for values in zip(*curves))


def run_distributed_observation_identifiability_study(
    config: DistributedObservationIdentifiabilityConfig = (
        DistributedObservationIdentifiabilityConfig()
    ),
    *,
    fit_models: bool = True,
) -> DistributedObservationIdentifiabilityStudyResult:
    """Run observation ablations and refuse authoritative under-ranked fits."""

    baselines = distributed_inverse_constant_experiments()[:3]
    baseline_property = baselines[0].material.electrical_resistivity
    if not isinstance(baseline_property, PiecewiseLinearProperty):
        raise TypeError("baseline resistivity must be piecewise linear")
    truth_property = smooth_resistivity_truth()
    truths = tuple(_truth_experiment(item, truth_property) for item in baselines)
    clean = tuple(
        observe_independent_distributed_result(
            truth_experiment,
            run_independent_distributed_experiment(
                truth_experiment,
                node_count=config.truth_node_count,
                time_step=config.truth_time_step,
            ),
            observation_interval=config.observation_interval,
            channels=DistributedObservationChannels(),
        )
        for truth_experiment in truths
    )
    noisy = tuple(
        add_distributed_gaussian_noise(
            dataset,
            standard_deviations={
                "cold_face_temperature": config.temperature_standard_deviation,
                "hot_face_temperature": config.temperature_standard_deviation,
                "voltage": config.voltage_standard_deviation,
            },
            seed=config.first_noise_seed + index,
        )
        for index, dataset in enumerate(clean)
    )

    holdout = distributed_inverse_constant_experiments()[3]
    holdout_truth_experiment = _truth_experiment(holdout, truth_property)
    holdout_truth_result = run_independent_distributed_experiment(
        holdout_truth_experiment,
        node_count=config.truth_node_count,
        time_step=config.truth_time_step,
    )
    parameters = tuple(
        DistributedPropertyCoefficient("electrical_resistivity", index)
        for index in range(3)
    )
    cases = []
    for case_index, definition in enumerate(distributed_observation_case_definitions()):
        experiments = tuple(baselines[index] for index in definition.experiment_indices)
        observations = tuple(
            _select_channels(noisy[index], definition.channels)
            for index in definition.experiment_indices
        )
        identifiability = analyze_distributed_identifiability(
            experiments,
            parameters,
            DistributedIdentifiabilityConfig(
                observation_interval=config.observation_interval,
                channels=definition.channels,
                temperature_standard_deviation=(
                    config.temperature_standard_deviation
                ),
                voltage_standard_deviation=config.voltage_standard_deviation,
            ),
        )
        assessment = assess_distributed_identifiability(
            identifiability, config.gate
        )
        if assessment.status == "structurally_non_identifiable":
            fit_policy = "withheld_structural_non_identifiability"
        elif assessment.status == "supported":
            fit_policy = "eligible_for_postfit_validation"
        else:
            fit_policy = "diagnostic_only_not_an_estimate"

        fits = []
        if fit_models and assessment.status != "structurally_non_identifiable":
            for start_index, initial_logs in enumerate(
                config.initial_log_multiplier_sets
            ):
                conventional = _fit_conventional(
                    experiments,
                    observations,
                    definition.channels,
                    initial_logs,
                    config,
                )
                conventional_multipliers = tuple(
                    value / base
                    for value, base in zip(
                        conventional.fitted_values, baseline_property.values
                    )
                )
                fits.append(
                    _fit_result(
                        estimator="conventional",
                        start_index=start_index,
                        initial_logs=initial_logs,
                        fitted_multipliers=conventional_multipliers,
                        loss=conventional.mean_normalized_squared_error,
                        accepted=assessment.status == "supported",
                        baseline_property=baseline_property,
                        truth_property=truth_property,
                        holdout=holdout,
                        holdout_truth_experiment=holdout_truth_experiment,
                        holdout_truth_result=holdout_truth_result,
                    )
                )
                multipliers, loss = _fit_pinn(
                    experiments,
                    observations,
                    initial_logs,
                    config,
                    seed=config.neural_seed + case_index,
                )
                fits.append(
                    _fit_result(
                        estimator="pinn",
                        start_index=start_index,
                        initial_logs=initial_logs,
                        fitted_multipliers=multipliers,
                        loss=loss,
                        accepted=assessment.status == "supported",
                        baseline_property=baseline_property,
                        truth_property=truth_property,
                        holdout=holdout,
                        holdout_truth_experiment=holdout_truth_experiment,
                        holdout_truth_result=holdout_truth_result,
                    )
                )
        cases.append(
            DistributedObservationCaseResult(
                definition=definition,
                identifiability=identifiability,
                assessment=assessment,
                fit_policy=fit_policy,
                fits=tuple(fits),
            )
        )
    return DistributedObservationIdentifiabilityStudyResult(
        config=config,
        truth_property=truth_property,
        truth_knot_multipliers=tuple(
            truth_property.value(temperature) / baseline
            for temperature, baseline in zip(
                baseline_property.temperatures, baseline_property.values
            )
        ),
        holdout_name="positive_0.4A_20K_lift",
        cases=tuple(cases),
    )
