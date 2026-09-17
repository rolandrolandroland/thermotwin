"""Frozen resource costs for prospective operating-decision actions.

This module implements Step 3 of the prospective operating-decision
experiment.  It converts the cost-free Step-2 uncertainty estimates into the
strict Step-1 action evaluations.  All energy is a nominal, selection-time
proxy; no hidden truth, realized outcome, verification observation, or final
response enters the calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence, Tuple

from .operating_decision import (
    FIXED_FACE_TEMPERATURE,
    FIXED_VOLTAGE,
    POLICY_NAMES,
    RUN_DURATION_SECONDS,
    STOP_NOW,
    VERIFICATION,
    FixedPolicy,
    OperatingRegime,
    default_fixed_policies,
    initial_acquisition_regime,
)
from .operating_decision_prospective import (
    ProspectiveActionEvaluation,
    ProspectiveSelection,
    ProspectiveSelectorRule,
    prospective_action_catalog_digest,
    prospective_selector_protocol_digest,
    prospective_selector_rule_from_payload,
    prospective_selector_rule_payload,
    select_prospective_action,
)
from .operating_decision_prospective_uncertainty import (
    ProspectiveActionUncertainty,
    ProspectivePaddedActionUncertainty,
    ProspectivePaddedModelActionSummary,
    PROSPECTIVE_PADDED_SCORING_PROTOCOL,
    ProspectiveUncertaintyConfig,
    ProspectiveUncertaintyResult,
    prospective_uncertainty_protocol_digest,
    score_prospective_uncertainty_with_offsets,
    validate_prospective_uncertainty_result_integrity,
)
from .operating_decision_realism import (
    OperatingDecisionRealismConfig,
    RunInstrumentation,
)
from .operating_decision_replication import corrected_physical_protocol_digest
from .operating_decision_resources import (
    NOMINAL_SELECTION_COST_PROTOCOL,
    nominal_selection_cost_proxy,
)
from .sensor_model_discrimination import (
    ALL_CHANNELS,
    COLD_FACE,
    VOLTAGE,
)


PROSPECTIVE_COST_SCHEMA_VERSION = 2
PROSPECTIVE_COST_PROTOCOL_VERSION = "operating_decision_prospective_cost_v2"
PROSPECTIVE_COST_FORMULA = (
    "development_padded_width_reduction_over_normalized_resource_cost_v2"
)
PROSPECTIVE_COST_ENERGY_CONVENTION = (
    "full_diagnostic_plan_incremental_nominal_terminal_energy_vs_stop_v1"
)
PROSPECTIVE_COST_TIME_CONVENTION = (
    "added_schedule_plus_assumed_reset_per_added_run_v1"
)
PROSPECTIVE_COST_INSTRUMENTATION_CONVENTION = (
    "typed_added_instrument_charged_once_v1"
)
PROSPECTIVE_COST_REFERENCE_ENERGY = "fixed_voltage_incremental_nominal_energy"
PROSPECTIVE_COST_REFERENCE_TIME = "one_added_run_plus_assumed_reset"
PROSPECTIVE_COST_PRIMARY_SCENARIO_NAME = "balanced_face_equal"
PROSPECTIVE_RESET_SENSITIVITY_SECONDS = (0.0, RUN_DURATION_SECONDS, 240.0)
_PROTOCOL_DOMAIN = "thermotwin.prospective_cost_protocol"
_RESULT_DOMAIN = "thermotwin.prospective_costed_scorecard"
_WEIGHT_TOLERANCE = 1.0e-12


def _label(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} must be a nonempty, trimmed label")


def _finite(name: str, value: float, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or (nonnegative and converted < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return 0.0 if converted == 0.0 else converted


def _positive(name: str, value: float) -> float:
    converted = _finite(name, value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _sha256(name: str, value: str) -> None:
    _label(name, value)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _canonical_digest(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProspectiveCostScenario:
    """Dimensionless resource weights declared before any evaluation cohort."""

    name: str = PROSPECTIVE_COST_PRIMARY_SCENARIO_NAME
    energy_weight: float = 1.0 / 3.0
    bench_time_weight: float = 1.0 / 3.0
    instrumentation_weight: float = 1.0 / 3.0
    face_instrumentation_multiplier: float = 1.0
    assumed_reset_seconds_per_added_run: float = RUN_DURATION_SECONDS

    def __post_init__(self) -> None:
        _label("cost scenario name", self.name)
        for attribute, label, value in (
            ("energy_weight", "energy weight", self.energy_weight),
            ("bench_time_weight", "bench-time weight", self.bench_time_weight),
            (
                "instrumentation_weight",
                "instrumentation weight",
                self.instrumentation_weight,
            ),
            (
                "face_instrumentation_multiplier",
                "face instrumentation multiplier",
                self.face_instrumentation_multiplier,
            ),
            (
                "assumed_reset_seconds_per_added_run",
                "assumed reset seconds",
                self.assumed_reset_seconds_per_added_run,
            ),
        ):
            converted = _finite(label, value, nonnegative=True)
            object.__setattr__(self, attribute, converted)
        weights = (
            self.energy_weight,
            self.bench_time_weight,
            self.instrumentation_weight,
        )
        if not math.isclose(
            sum(weights),
            1.0,
            rel_tol=0.0,
            abs_tol=_WEIGHT_TOLERANCE,
        ):
            raise ValueError("prospective cost weights must sum to one")
        if self.energy_weight + self.bench_time_weight <= 0.0:
            raise ValueError("energy and bench-time weights cannot both be zero")


PRIMARY_PROSPECTIVE_COST_SCENARIO = ProspectiveCostScenario()
if (
    PROSPECTIVE_RESET_SENSITIVITY_SECONDS
    != tuple(sorted(set(PROSPECTIVE_RESET_SENSITIVITY_SECONDS)))
    or PRIMARY_PROSPECTIVE_COST_SCENARIO.assumed_reset_seconds_per_added_run
    not in PROSPECTIVE_RESET_SENSITIVITY_SECONDS
):
    raise RuntimeError("prospective reset sensitivity grid is invalid")


def _scenario_grid() -> Tuple[ProspectiveCostScenario, ...]:
    mixes = (
        ("balanced", 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ("energy", 0.70, 0.20, 0.10),
        ("bench_time", 0.20, 0.70, 0.10),
        ("instrumentation", 0.15, 0.15, 0.70),
    )
    face_levels = (
        ("face_quarter", 0.25),
        ("face_equal", 1.0),
        ("face_quadruple", 4.0),
    )
    return tuple(
        ProspectiveCostScenario(
            name=f"{mix_name}_{face_name}",
            energy_weight=energy,
            bench_time_weight=time,
            instrumentation_weight=instrumentation,
            face_instrumentation_multiplier=face_multiplier,
        )
        for mix_name, energy, time, instrumentation in mixes
        for face_name, face_multiplier in face_levels
    )


PROSPECTIVE_COST_SENSITIVITY_SCENARIOS = _scenario_grid()
if tuple(item.name for item in PROSPECTIVE_COST_SENSITIVITY_SCENARIOS).count(
    PROSPECTIVE_COST_PRIMARY_SCENARIO_NAME
) != 1:
    raise RuntimeError("primary prospective cost scenario must occur once")


@dataclass(frozen=True)
class ProspectiveRunResources:
    """Nominal selection-time energy for one diagnostic run."""

    regime_name: str
    phase: str
    temporary_face_sensor: bool
    nominal_terminal_energy_joules: float

    def __post_init__(self) -> None:
        _label("resource regime name", self.regime_name)
        _label("resource phase", self.phase)
        if not isinstance(self.temporary_face_sensor, bool):
            raise ValueError("temporary face-sensor state must be boolean")
        object.__setattr__(
            self,
            "nominal_terminal_energy_joules",
            _finite(
                "nominal run energy",
                self.nominal_terminal_energy_joules,
                nonnegative=True,
            ),
        )


@dataclass(frozen=True)
class ProspectiveActionResources:
    """Raw incremental resources for one complete diagnostic-plan choice."""

    policy_name: str
    plan_runs: Tuple[ProspectiveRunResources, ...]
    added_regime_names: Tuple[str, ...]
    added_run_count: int
    added_schedule_seconds: float
    assumed_reset_seconds: float
    incremental_bench_seconds: float
    added_instruments: Tuple[str, ...]
    nominal_total_plan_energy_joules: float
    stop_reference_plan_energy_joules: float
    incremental_nominal_energy_joules: float

    def __post_init__(self) -> None:
        if self.policy_name not in POLICY_NAMES:
            raise ValueError("action resources have an unknown policy")
        runs = tuple(self.plan_runs)
        regimes = tuple(self.added_regime_names)
        instruments = tuple(self.added_instruments)
        object.__setattr__(self, "plan_runs", runs)
        object.__setattr__(self, "added_regime_names", regimes)
        object.__setattr__(self, "added_instruments", instruments)
        if not runs or any(
            not isinstance(item, ProspectiveRunResources) for item in runs
        ):
            raise ValueError("action resources need nominal diagnostic runs")
        if len({item.regime_name for item in runs}) != len(runs):
            raise ValueError("action resource run names must be unique")
        if any(not isinstance(item, str) or not item for item in regimes):
            raise ValueError("added regime names must be labels")
        if instruments not in ((), (VOLTAGE,), (COLD_FACE,)):
            raise ValueError("added instruments must retain their channel identity")
        if (
            not isinstance(self.added_run_count, int)
            or isinstance(self.added_run_count, bool)
            or self.added_run_count < 0
            or self.added_run_count != len(regimes)
        ):
            raise ValueError("added run count is inconsistent")
        for attribute, label in (
            ("added_schedule_seconds", "added schedule seconds"),
            ("assumed_reset_seconds", "assumed reset seconds"),
            ("incremental_bench_seconds", "incremental bench seconds"),
            (
                "nominal_total_plan_energy_joules",
                "nominal total plan energy",
            ),
            (
                "stop_reference_plan_energy_joules",
                "stop reference plan energy",
            ),
            (
                "incremental_nominal_energy_joules",
                "incremental nominal energy",
            ),
        ):
            object.__setattr__(
                self,
                attribute,
                _finite(label, getattr(self, attribute), nonnegative=True),
            )
        if self.added_schedule_seconds != RUN_DURATION_SECONDS * self.added_run_count:
            raise ValueError("added schedule time does not match added runs")
        if self.incremental_bench_seconds != (
            self.added_schedule_seconds + self.assumed_reset_seconds
        ):
            raise ValueError("incremental bench time must include assumed resets")
        if self.nominal_total_plan_energy_joules != sum(
            item.nominal_terminal_energy_joules for item in runs
        ):
            raise ValueError("nominal total plan energy does not match its runs")
        expected_increment = max(
            0.0,
            self.nominal_total_plan_energy_joules
            - self.stop_reference_plan_energy_joules,
        )
        if self.incremental_nominal_energy_joules != expected_increment:
            raise ValueError("incremental nominal energy does not match stop")
        if self.policy_name == STOP_NOW:
            if any(
                (
                    regimes,
                    self.added_run_count,
                    self.added_schedule_seconds,
                    self.assumed_reset_seconds,
                    self.incremental_bench_seconds,
                    instruments,
                    self.incremental_nominal_energy_joules,
                )
            ):
                raise ValueError("stop resources must be the zero reference")
        elif (
            self.added_run_count == 0
            or self.incremental_bench_seconds <= 0.0
            or self.incremental_nominal_energy_joules <= 0.0
        ):
            raise ValueError("acquisition action resources must be positive")


@dataclass(frozen=True)
class ProspectiveActionCost:
    """Normalized resource components and scalar cost for one action."""

    policy_name: str
    normalized_energy: float
    normalized_bench_time: float
    normalized_instrumentation: float
    energy_component: float
    bench_time_component: float
    instrumentation_component: float
    declared_cost: float

    def __post_init__(self) -> None:
        if self.policy_name not in POLICY_NAMES:
            raise ValueError("action cost has an unknown policy")
        for attribute, label in (
            ("normalized_energy", "normalized energy"),
            ("normalized_bench_time", "normalized bench time"),
            ("normalized_instrumentation", "normalized instrumentation"),
            ("energy_component", "energy cost component"),
            ("bench_time_component", "bench-time cost component"),
            ("instrumentation_component", "instrumentation cost component"),
            ("declared_cost", "declared cost"),
        ):
            object.__setattr__(
                self,
                attribute,
                _finite(label, getattr(self, attribute), nonnegative=True),
            )
        if self.declared_cost != (
            self.energy_component
            + self.bench_time_component
            + self.instrumentation_component
        ):
            raise ValueError("declared cost does not match its components")
        if self.policy_name == STOP_NOW:
            if any(
                (
                    self.normalized_energy,
                    self.normalized_bench_time,
                    self.normalized_instrumentation,
                    self.declared_cost,
                )
            ):
                raise ValueError("stop cost must remain zero")
        elif self.declared_cost <= 0.0:
            raise ValueError("acquisition action cost must be positive")


def _scenario_payload(scenario: ProspectiveCostScenario) -> dict:
    return {
        "name": scenario.name,
        "energy_weight": scenario.energy_weight,
        "bench_time_weight": scenario.bench_time_weight,
        "instrumentation_weight": scenario.instrumentation_weight,
        "face_instrumentation_multiplier": (
            scenario.face_instrumentation_multiplier
        ),
        "assumed_reset_seconds_per_added_run": (
            scenario.assumed_reset_seconds_per_added_run
        ),
    }


def _run_resources_payload(resources: ProspectiveRunResources) -> dict:
    return {
        "regime_name": resources.regime_name,
        "phase": resources.phase,
        "temporary_face_sensor": resources.temporary_face_sensor,
        "nominal_terminal_energy_joules": (
            resources.nominal_terminal_energy_joules
        ),
    }


def _action_resources_payload(resources: ProspectiveActionResources) -> dict:
    return {
        "policy_name": resources.policy_name,
        "plan_runs": [
            _run_resources_payload(item) for item in resources.plan_runs
        ],
        "added_regime_names": list(resources.added_regime_names),
        "added_run_count": resources.added_run_count,
        "added_schedule_seconds": resources.added_schedule_seconds,
        "assumed_reset_seconds": resources.assumed_reset_seconds,
        "incremental_bench_seconds": resources.incremental_bench_seconds,
        "added_instruments": list(resources.added_instruments),
        "nominal_total_plan_energy_joules": (
            resources.nominal_total_plan_energy_joules
        ),
        "stop_reference_plan_energy_joules": (
            resources.stop_reference_plan_energy_joules
        ),
        "incremental_nominal_energy_joules": (
            resources.incremental_nominal_energy_joules
        ),
    }


def _action_cost_payload(cost: ProspectiveActionCost) -> dict:
    return {
        "policy_name": cost.policy_name,
        "normalized_energy": cost.normalized_energy,
        "normalized_bench_time": cost.normalized_bench_time,
        "normalized_instrumentation": cost.normalized_instrumentation,
        "energy_component": cost.energy_component,
        "bench_time_component": cost.bench_time_component,
        "instrumentation_component": cost.instrumentation_component,
        "declared_cost": cost.declared_cost,
    }


def _action_evaluation_payload(evaluation: ProspectiveActionEvaluation) -> dict:
    return {
        "policy_name": evaluation.policy_name,
        "eligible": evaluation.eligible,
        "failure_reason": evaluation.failure_reason,
        "uncertainty_before": evaluation.uncertainty_before,
        "expected_uncertainty_after": evaluation.expected_uncertainty_after,
        "raw_uncertainty_before": evaluation.raw_uncertainty_before,
        "raw_expected_uncertainty_after": (
            evaluation.raw_expected_uncertainty_after
        ),
        "raw_expected_uncertainty_reduction": (
            evaluation.raw_expected_uncertainty_reduction
        ),
        "development_offset": evaluation.development_offset,
        "expected_uncertainty_reduction": (
            evaluation.expected_uncertainty_reduction
        ),
        "declared_cost": evaluation.declared_cost,
        "utility_per_cost": evaluation.utility_per_cost,
        "prospective_draw_count": evaluation.prospective_draw_count,
    }


def _instrumentation_for_regime(regime: OperatingRegime) -> RunInstrumentation:
    return RunInstrumentation(temporary_face_sensor=COLD_FACE in regime.channels)


def _verification_regime(
    policy: FixedPolicy,
    physical_config: OperatingDecisionRealismConfig,
) -> OperatingRegime:
    acquisition_regimes = (initial_acquisition_regime(), *policy.additional_regimes)
    installed = {
        channel for regime in acquisition_regimes for channel in regime.channels
    }
    return OperatingRegime(
        name="fixed_verification",
        phase=VERIFICATION,
        current=physical_config.verification_current,
        channels=tuple(channel for channel in ALL_CHANNELS if channel in installed),
    )


def _plan_run_resources(
    policy: FixedPolicy,
    physical_config: OperatingDecisionRealismConfig,
) -> Tuple[ProspectiveRunResources, ...]:
    regimes = (
        initial_acquisition_regime(),
        *policy.additional_regimes,
        _verification_regime(policy, physical_config),
    )
    return tuple(
        ProspectiveRunResources(
            regime_name=regime.name,
            phase=regime.phase,
            temporary_face_sensor=(COLD_FACE in regime.channels),
            nominal_terminal_energy_joules=nominal_selection_cost_proxy(
                regime,
                _instrumentation_for_regime(regime),
                physical_config,
            ).terminal_energy,
        )
        for regime in regimes
    )


def _build_action_resources(
    physical_config: OperatingDecisionRealismConfig,
    scenario: ProspectiveCostScenario,
) -> Tuple[ProspectiveActionResources, ...]:
    if not isinstance(physical_config, OperatingDecisionRealismConfig):
        raise ValueError("prospective resources need a realism configuration")
    if not isinstance(scenario, ProspectiveCostScenario):
        raise ValueError("prospective resources need a cost scenario")
    policies = default_fixed_policies()
    plans = {
        policy.name: _plan_run_resources(policy, physical_config)
        for policy in policies
    }
    stop_energy = sum(
        item.nominal_terminal_energy_joules for item in plans[STOP_NOW]
    )
    if stop_energy <= 0.0:
        raise ValueError("stop reference plan energy must be positive")
    resources = []
    for policy in policies:
        plan = plans[policy.name]
        total_energy = sum(item.nominal_terminal_energy_joules for item in plan)
        added_count = len(policy.additional_regimes)
        added_schedule = RUN_DURATION_SECONDS * added_count
        reset_seconds = scenario.assumed_reset_seconds_per_added_run * added_count
        added_instruments = tuple(
            channel
            for channel in (VOLTAGE, COLD_FACE)
            if any(channel in regime.channels for regime in policy.additional_regimes)
        )
        if len(added_instruments) != policy.extra_sensor_count:
            raise ValueError("policy instrumentation does not match its catalog")
        resources.append(
            ProspectiveActionResources(
                policy_name=policy.name,
                plan_runs=plan,
                added_regime_names=tuple(
                    regime.name for regime in policy.additional_regimes
                ),
                added_run_count=added_count,
                added_schedule_seconds=added_schedule,
                assumed_reset_seconds=reset_seconds,
                incremental_bench_seconds=added_schedule + reset_seconds,
                added_instruments=added_instruments,
                nominal_total_plan_energy_joules=total_energy,
                stop_reference_plan_energy_joules=stop_energy,
                incremental_nominal_energy_joules=max(0.0, total_energy - stop_energy),
            )
        )
    return tuple(resources)


def prospective_action_resources(
    physical_config: OperatingDecisionRealismConfig,
    scenario: ProspectiveCostScenario = PRIMARY_PROSPECTIVE_COST_SCENARIO,
) -> Tuple[ProspectiveActionResources, ...]:
    """Return raw action resources available before a hidden device is revealed."""

    return _build_action_resources(physical_config, scenario)


def _build_action_costs(
    resources: Sequence[ProspectiveActionResources],
    scenario: ProspectiveCostScenario,
) -> Tuple[Tuple[ProspectiveActionCost, ...], float, float]:
    resources = tuple(resources)
    if tuple(item.policy_name for item in resources) != POLICY_NAMES:
        raise ValueError("prospective resources must use canonical policy order")
    by_policy = {item.policy_name: item for item in resources}
    energy_reference = _positive(
        "cost energy reference",
        by_policy[FIXED_VOLTAGE].incremental_nominal_energy_joules,
    )
    time_reference = _positive(
        "cost bench-time reference",
        RUN_DURATION_SECONDS + scenario.assumed_reset_seconds_per_added_run,
    )
    costs = []
    for item in resources:
        normalized_energy = item.incremental_nominal_energy_joules / energy_reference
        normalized_time = item.incremental_bench_seconds / time_reference
        normalized_instrumentation = (
            float(VOLTAGE in item.added_instruments)
            + scenario.face_instrumentation_multiplier
            * float(COLD_FACE in item.added_instruments)
        )
        energy_component = scenario.energy_weight * normalized_energy
        time_component = scenario.bench_time_weight * normalized_time
        instrumentation_component = (
            scenario.instrumentation_weight * normalized_instrumentation
        )
        costs.append(
            ProspectiveActionCost(
                policy_name=item.policy_name,
                normalized_energy=normalized_energy,
                normalized_bench_time=normalized_time,
                normalized_instrumentation=normalized_instrumentation,
                energy_component=energy_component,
                bench_time_component=time_component,
                instrumentation_component=instrumentation_component,
                declared_cost=(
                    energy_component + time_component + instrumentation_component
                ),
            )
        )
    voltage_cost = next(
        item.declared_cost for item in costs if item.policy_name == FIXED_VOLTAGE
    )
    if not math.isclose(voltage_cost, 1.0, rel_tol=0.0, abs_tol=_WEIGHT_TOLERANCE):
        raise ValueError("fixed voltage must remain the unit-cost anchor")
    return tuple(costs), energy_reference, time_reference


def _build_action_evaluations(
    uncertainties: Sequence[
        ProspectivePaddedActionUncertainty | ProspectiveActionUncertainty
    ],
    costs: Sequence[ProspectiveActionCost],
) -> Tuple[ProspectiveActionEvaluation, ...]:
    uncertainties = tuple(uncertainties)
    costs = tuple(costs)
    uncertainties = tuple(
        item
        if isinstance(item, ProspectivePaddedActionUncertainty)
        else ProspectivePaddedActionUncertainty(
            raw_action=item,
            development_offset=0.0,
            padded_uncertainty_before=item.uncertainty_before,
            padded_expected_uncertainty_after=item.expected_uncertainty_after,
            source_summaries=tuple(
                ProspectivePaddedModelActionSummary(
                    raw_summary=summary,
                    padded_mean_after_width=summary.mean_after_width,
                )
                for summary in item.source_summaries
            ),
        )
        for item in uncertainties
    )
    if tuple(item.policy_name for item in uncertainties) != POLICY_NAMES:
        raise ValueError("uncertainty actions must use canonical policy order")
    if tuple(item.policy_name for item in costs) != POLICY_NAMES:
        raise ValueError("action costs must use canonical policy order")
    evaluations = []
    for uncertainty, cost in zip(uncertainties, costs):
        if uncertainty.policy_name == STOP_NOW:
            evaluations.append(ProspectiveActionEvaluation(STOP_NOW, True))
        elif not uncertainty.eligible:
            evaluations.append(
                ProspectiveActionEvaluation(
                    policy_name=uncertainty.policy_name,
                    eligible=False,
                    failure_reason=uncertainty.failure_reason,
                )
            )
        else:
            evaluations.append(
                ProspectiveActionEvaluation(
                    policy_name=uncertainty.policy_name,
                    eligible=True,
                    uncertainty_before=uncertainty.padded_uncertainty_before,
                    expected_uncertainty_after=(
                        uncertainty.padded_expected_uncertainty_after
                    ),
                    declared_cost=cost.declared_cost,
                    prospective_draw_count=(
                        uncertainty.raw_action.prospective_draw_count
                    ),
                    raw_uncertainty_before=(
                        uncertainty.raw_action.uncertainty_before
                    ),
                    raw_expected_uncertainty_after=(
                        uncertainty.raw_action.expected_uncertainty_after
                    ),
                    development_offset=uncertainty.development_offset,
                )
            )
    return tuple(evaluations)


def prospective_cost_protocol_digest(
    physical_config: OperatingDecisionRealismConfig,
    uncertainty_config: ProspectiveUncertaintyConfig = ProspectiveUncertaintyConfig(),
    scenario: ProspectiveCostScenario = PRIMARY_PROSPECTIVE_COST_SCENARIO,
    selector_rule: ProspectiveSelectorRule = ProspectiveSelectorRule(),
) -> str:
    """Bind Step 3 resources, normalization, and upstream Step 2 protocol."""

    if not isinstance(physical_config, OperatingDecisionRealismConfig):
        raise ValueError("prospective cost protocol needs a realism configuration")
    if not isinstance(uncertainty_config, ProspectiveUncertaintyConfig):
        raise ValueError("prospective cost protocol needs an uncertainty configuration")
    if not isinstance(scenario, ProspectiveCostScenario):
        raise ValueError("prospective cost protocol needs a cost scenario")
    if not isinstance(selector_rule, ProspectiveSelectorRule):
        raise ValueError("prospective cost protocol needs a selector rule")
    resources = _build_action_resources(physical_config, scenario)
    costs, energy_reference, time_reference = _build_action_costs(
        resources,
        scenario,
    )
    return _canonical_digest(
        {
            "domain": _PROTOCOL_DOMAIN,
            "schema_version": PROSPECTIVE_COST_SCHEMA_VERSION,
            "protocol_version": PROSPECTIVE_COST_PROTOCOL_VERSION,
            "formula": PROSPECTIVE_COST_FORMULA,
            "padded_scoring_protocol": PROSPECTIVE_PADDED_SCORING_PROTOCOL,
            "energy_convention": PROSPECTIVE_COST_ENERGY_CONVENTION,
            "time_convention": PROSPECTIVE_COST_TIME_CONVENTION,
            "instrumentation_convention": (
                PROSPECTIVE_COST_INSTRUMENTATION_CONVENTION
            ),
            "energy_reference_convention": PROSPECTIVE_COST_REFERENCE_ENERGY,
            "time_reference_convention": PROSPECTIVE_COST_REFERENCE_TIME,
            "nominal_energy_protocol": NOMINAL_SELECTION_COST_PROTOCOL,
            "physical_protocol_digest": corrected_physical_protocol_digest(
                physical_config
            ),
            "uncertainty_protocol_digest": prospective_uncertainty_protocol_digest(
                physical_config,
                uncertainty_config,
            ),
            "action_catalog_digest": prospective_action_catalog_digest(),
            "selector_protocol_digest": prospective_selector_protocol_digest(
                selector_rule
            ),
            "selector_rule": prospective_selector_rule_payload(selector_rule),
            "scenario": _scenario_payload(scenario),
            "energy_reference_joules": energy_reference,
            "bench_time_reference_seconds": time_reference,
            "action_resources": [
                _action_resources_payload(item) for item in resources
            ],
            "action_costs": [_action_cost_payload(item) for item in costs],
        }
    )


def _result_digest(
    *,
    uncertainty_result: ProspectiveUncertaintyResult,
    scenario: ProspectiveCostScenario,
    selector_rule: ProspectiveSelectorRule,
    protocol_digest: str,
    energy_reference_joules: float,
    bench_time_reference_seconds: float,
    resources: Sequence[ProspectiveActionResources],
    costs: Sequence[ProspectiveActionCost],
    evaluations: Sequence[ProspectiveActionEvaluation],
) -> str:
    return _canonical_digest(
        {
            "domain": _RESULT_DOMAIN,
            "uncertainty_result_digest": uncertainty_result.result_digest,
            "acquisition_evidence_digest": (
                uncertainty_result.acquisition_evidence.evidence_digest
            ),
            "scenario": _scenario_payload(scenario),
            "selector_rule": prospective_selector_rule_payload(selector_rule),
            "protocol_digest": protocol_digest,
            "energy_reference_joules": energy_reference_joules,
            "bench_time_reference_seconds": bench_time_reference_seconds,
            "action_resources": [
                _action_resources_payload(item) for item in resources
            ],
            "action_costs": [_action_cost_payload(item) for item in costs],
            "action_evaluations": [
                _action_evaluation_payload(item) for item in evaluations
            ],
        }
    )


@dataclass(frozen=True)
class ProspectiveCostedScorecard:
    """Authenticated Step-3 costs and selector-ready action evaluations."""

    uncertainty_result: ProspectiveUncertaintyResult
    scenario: ProspectiveCostScenario
    selector_rule: ProspectiveSelectorRule
    protocol_digest: str
    energy_reference_joules: float
    bench_time_reference_seconds: float
    action_resources: Tuple[ProspectiveActionResources, ...]
    action_costs: Tuple[ProspectiveActionCost, ...]
    action_evaluations: Tuple[ProspectiveActionEvaluation, ...]
    result_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.uncertainty_result, ProspectiveUncertaintyResult):
            raise ValueError("costed scorecard needs a Step-2 uncertainty result")
        validate_prospective_uncertainty_result_integrity(
            self.uncertainty_result
        )
        if not isinstance(self.scenario, ProspectiveCostScenario):
            raise ValueError("costed scorecard needs a cost scenario")
        if not isinstance(self.selector_rule, ProspectiveSelectorRule):
            raise ValueError("costed scorecard needs its actual selector rule")
        resources = tuple(self.action_resources)
        costs = tuple(self.action_costs)
        evaluations = tuple(self.action_evaluations)
        object.__setattr__(self, "action_resources", resources)
        object.__setattr__(self, "action_costs", costs)
        object.__setattr__(self, "action_evaluations", evaluations)
        energy_reference = _positive(
            "cost energy reference",
            self.energy_reference_joules,
        )
        time_reference = _positive(
            "cost bench-time reference",
            self.bench_time_reference_seconds,
        )
        object.__setattr__(self, "energy_reference_joules", energy_reference)
        object.__setattr__(self, "bench_time_reference_seconds", time_reference)

        expected_resources = _build_action_resources(
            self.uncertainty_result.physical_config,
            self.scenario,
        )
        expected_costs, expected_energy_reference, expected_time_reference = (
            _build_action_costs(expected_resources, self.scenario)
        )
        padded_uncertainties = score_prospective_uncertainty_with_offsets(
            self.uncertainty_result,
            self.selector_rule.development_offsets,
        )
        expected_evaluations = _build_action_evaluations(
            padded_uncertainties,
            expected_costs,
        )
        if resources != expected_resources:
            raise ValueError("costed resources do not match the frozen policies")
        if costs != expected_costs:
            raise ValueError("action costs do not match the frozen formula")
        if energy_reference != expected_energy_reference:
            raise ValueError("cost energy reference is inconsistent")
        if time_reference != expected_time_reference:
            raise ValueError("cost bench-time reference is inconsistent")
        if evaluations != expected_evaluations:
            raise ValueError("action evaluations do not match uncertainty and cost")

        _sha256("prospective cost protocol digest", self.protocol_digest)
        expected_protocol = prospective_cost_protocol_digest(
            self.uncertainty_result.physical_config,
            self.uncertainty_result.config,
            self.scenario,
            self.selector_rule,
        )
        if self.protocol_digest != expected_protocol:
            raise ValueError("prospective cost protocol digest is invalid")
        _sha256("prospective cost result digest", self.result_digest)
        expected_result = _result_digest(
            uncertainty_result=self.uncertainty_result,
            scenario=self.scenario,
            selector_rule=self.selector_rule,
            protocol_digest=self.protocol_digest,
            energy_reference_joules=energy_reference,
            bench_time_reference_seconds=time_reference,
            resources=resources,
            costs=costs,
            evaluations=evaluations,
        )
        if self.result_digest != expected_result:
            raise ValueError("prospective cost result digest is invalid")


def cost_prospective_uncertainty(
    uncertainty_result: ProspectiveUncertaintyResult,
    scenario: ProspectiveCostScenario = PRIMARY_PROSPECTIVE_COST_SCENARIO,
    selector_rule: ProspectiveSelectorRule = ProspectiveSelectorRule(),
) -> ProspectiveCostedScorecard:
    """Attach costs and one actual padded selector rule to raw Step-2 evidence."""

    if not isinstance(uncertainty_result, ProspectiveUncertaintyResult):
        raise ValueError("prospective costing needs a Step-2 uncertainty result")
    validate_prospective_uncertainty_result_integrity(uncertainty_result)
    if not isinstance(scenario, ProspectiveCostScenario):
        raise ValueError("prospective costing needs a cost scenario")
    if not isinstance(selector_rule, ProspectiveSelectorRule):
        raise ValueError("prospective costing needs a selector rule")
    resources = _build_action_resources(uncertainty_result.physical_config, scenario)
    costs, energy_reference, time_reference = _build_action_costs(
        resources,
        scenario,
    )
    padded_uncertainties = score_prospective_uncertainty_with_offsets(
        uncertainty_result,
        selector_rule.development_offsets,
    )
    evaluations = _build_action_evaluations(
        padded_uncertainties,
        costs,
    )
    protocol_digest = prospective_cost_protocol_digest(
        uncertainty_result.physical_config,
        uncertainty_result.config,
        scenario,
        selector_rule,
    )
    result_digest = _result_digest(
        uncertainty_result=uncertainty_result,
        scenario=scenario,
        selector_rule=selector_rule,
        protocol_digest=protocol_digest,
        energy_reference_joules=energy_reference,
        bench_time_reference_seconds=time_reference,
        resources=resources,
        costs=costs,
        evaluations=evaluations,
    )
    return ProspectiveCostedScorecard(
        uncertainty_result=uncertainty_result,
        scenario=scenario,
        selector_rule=selector_rule,
        protocol_digest=protocol_digest,
        energy_reference_joules=energy_reference,
        bench_time_reference_seconds=time_reference,
        action_resources=resources,
        action_costs=costs,
        action_evaluations=evaluations,
        result_digest=result_digest,
    )


def select_costed_prospective_action(
    scorecard: ProspectiveCostedScorecard,
    rule: ProspectiveSelectorRule | None = None,
) -> ProspectiveSelection:
    """Select from the scorecard and its bound acquisition snapshot."""

    if not isinstance(scorecard, ProspectiveCostedScorecard):
        raise ValueError("costed selection needs a prospective scorecard")
    if rule is not None and not isinstance(rule, ProspectiveSelectorRule):
        raise ValueError("costed selection needs a prospective selector rule")
    selected_rule = scorecard.selector_rule if rule is None else rule
    if prospective_selector_protocol_digest(selected_rule) != (
        prospective_selector_protocol_digest(scorecard.selector_rule)
    ):
        raise ValueError("costed selection rule does not match its scorecard")
    return select_prospective_action(
        scorecard.uncertainty_result.acquisition_evidence.snapshot,
        scorecard.action_evaluations,
        selected_rule,
    )


def prospective_costed_scorecard_payload(
    scorecard: ProspectiveCostedScorecard,
) -> dict:
    """Return a strict JSON-ready Step-3 record referencing its Step-2 result."""

    if not isinstance(scorecard, ProspectiveCostedScorecard):
        raise ValueError("prospective cost payload needs a scorecard")
    expected_result = _result_digest(
        uncertainty_result=scorecard.uncertainty_result,
        scenario=scorecard.scenario,
        selector_rule=scorecard.selector_rule,
        protocol_digest=scorecard.protocol_digest,
        energy_reference_joules=scorecard.energy_reference_joules,
        bench_time_reference_seconds=scorecard.bench_time_reference_seconds,
        resources=scorecard.action_resources,
        costs=scorecard.action_costs,
        evaluations=scorecard.action_evaluations,
    )
    if scorecard.result_digest != expected_result:
        raise ValueError("prospective cost result digest is invalid")
    return {
        "schema_version": PROSPECTIVE_COST_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_COST_PROTOCOL_VERSION,
        "formula": PROSPECTIVE_COST_FORMULA,
        "padded_scoring_protocol": PROSPECTIVE_PADDED_SCORING_PROTOCOL,
        "energy_convention": PROSPECTIVE_COST_ENERGY_CONVENTION,
        "time_convention": PROSPECTIVE_COST_TIME_CONVENTION,
        "instrumentation_convention": (
            PROSPECTIVE_COST_INSTRUMENTATION_CONVENTION
        ),
        "nominal_energy_protocol": NOMINAL_SELECTION_COST_PROTOCOL,
        "physical_protocol_digest": (
            scorecard.uncertainty_result.acquisition_evidence.physical_protocol_digest
        ),
        "uncertainty_protocol_digest": scorecard.uncertainty_result.protocol_digest,
        "uncertainty_result_digest": scorecard.uncertainty_result.result_digest,
        "action_catalog_digest": prospective_action_catalog_digest(),
        "selector_protocol_digest": prospective_selector_protocol_digest(
            scorecard.selector_rule
        ),
        "selector_rule": prospective_selector_rule_payload(
            scorecard.selector_rule
        ),
        "acquisition_evidence_digest": (
            scorecard.uncertainty_result.acquisition_evidence.evidence_digest
        ),
        "protocol_digest": scorecard.protocol_digest,
        "result_digest": scorecard.result_digest,
        "scenario": _scenario_payload(scorecard.scenario),
        "energy_reference_joules": scorecard.energy_reference_joules,
        "bench_time_reference_seconds": scorecard.bench_time_reference_seconds,
        "action_resources": [
            _action_resources_payload(item) for item in scorecard.action_resources
        ],
        "action_costs": [
            _action_cost_payload(item) for item in scorecard.action_costs
        ],
        "action_evaluations": [
            _action_evaluation_payload(item)
            for item in scorecard.action_evaluations
        ],
    }


def _exact_keys(payload: object, expected: set, label: str) -> dict:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{label} is malformed")
    return payload


def _scenario_from_payload(payload: object) -> ProspectiveCostScenario:
    item = _exact_keys(
        payload,
        {
            "name",
            "energy_weight",
            "bench_time_weight",
            "instrumentation_weight",
            "face_instrumentation_multiplier",
            "assumed_reset_seconds_per_added_run",
        },
        "prospective cost scenario",
    )
    return ProspectiveCostScenario(**item)


def _run_resources_from_payload(payload: object) -> ProspectiveRunResources:
    item = _exact_keys(
        payload,
        {
            "regime_name",
            "phase",
            "temporary_face_sensor",
            "nominal_terminal_energy_joules",
        },
        "prospective run resources",
    )
    return ProspectiveRunResources(**item)


def _action_resources_from_payload(payload: object) -> ProspectiveActionResources:
    item = _exact_keys(
        payload,
        {
            "policy_name",
            "plan_runs",
            "added_regime_names",
            "added_run_count",
            "added_schedule_seconds",
            "assumed_reset_seconds",
            "incremental_bench_seconds",
            "added_instruments",
            "nominal_total_plan_energy_joules",
            "stop_reference_plan_energy_joules",
            "incremental_nominal_energy_joules",
        },
        "prospective action resources",
    )
    if not isinstance(item["plan_runs"], list):
        raise ValueError("prospective action resource runs must be a list")
    if not isinstance(item["added_regime_names"], list) or not isinstance(
        item["added_instruments"], list
    ):
        raise ValueError("prospective action resource labels must be lists")
    return ProspectiveActionResources(
        policy_name=item["policy_name"],
        plan_runs=tuple(
            _run_resources_from_payload(run) for run in item["plan_runs"]
        ),
        added_regime_names=tuple(item["added_regime_names"]),
        added_run_count=item["added_run_count"],
        added_schedule_seconds=item["added_schedule_seconds"],
        assumed_reset_seconds=item["assumed_reset_seconds"],
        incremental_bench_seconds=item["incremental_bench_seconds"],
        added_instruments=tuple(item["added_instruments"]),
        nominal_total_plan_energy_joules=(
            item["nominal_total_plan_energy_joules"]
        ),
        stop_reference_plan_energy_joules=(
            item["stop_reference_plan_energy_joules"]
        ),
        incremental_nominal_energy_joules=(
            item["incremental_nominal_energy_joules"]
        ),
    )


def _action_cost_from_payload(payload: object) -> ProspectiveActionCost:
    item = _exact_keys(
        payload,
        {
            "policy_name",
            "normalized_energy",
            "normalized_bench_time",
            "normalized_instrumentation",
            "energy_component",
            "bench_time_component",
            "instrumentation_component",
            "declared_cost",
        },
        "prospective action cost",
    )
    return ProspectiveActionCost(**item)


def _action_evaluation_from_payload(payload: object) -> ProspectiveActionEvaluation:
    item = _exact_keys(
        payload,
        {
            "policy_name",
            "eligible",
            "failure_reason",
            "uncertainty_before",
            "expected_uncertainty_after",
            "raw_uncertainty_before",
            "raw_expected_uncertainty_after",
            "raw_expected_uncertainty_reduction",
            "development_offset",
            "expected_uncertainty_reduction",
            "declared_cost",
            "utility_per_cost",
            "prospective_draw_count",
        },
        "prospective action evaluation",
    )
    evaluation = ProspectiveActionEvaluation(
        policy_name=item["policy_name"],
        eligible=item["eligible"],
        failure_reason=item["failure_reason"],
        uncertainty_before=item["uncertainty_before"],
        expected_uncertainty_after=item["expected_uncertainty_after"],
        declared_cost=item["declared_cost"],
        prospective_draw_count=item["prospective_draw_count"],
        raw_uncertainty_before=item["raw_uncertainty_before"],
        raw_expected_uncertainty_after=(
            item["raw_expected_uncertainty_after"]
        ),
        development_offset=item["development_offset"],
    )
    if (
        item["expected_uncertainty_reduction"]
        != evaluation.expected_uncertainty_reduction
        or item["raw_expected_uncertainty_reduction"]
        != evaluation.raw_expected_uncertainty_reduction
        or item["utility_per_cost"] != evaluation.utility_per_cost
    ):
        raise ValueError("prospective action evaluation derivatives are invalid")
    return evaluation


def prospective_costed_scorecard_from_payload(
    payload: dict,
    uncertainty_result: ProspectiveUncertaintyResult,
) -> ProspectiveCostedScorecard:
    """Load Step-3 evidence while rederiving every resource and cost field."""

    item = _exact_keys(
        payload,
        {
            "schema_version",
            "protocol_version",
            "formula",
            "padded_scoring_protocol",
            "energy_convention",
            "time_convention",
            "instrumentation_convention",
            "nominal_energy_protocol",
            "physical_protocol_digest",
            "uncertainty_protocol_digest",
            "uncertainty_result_digest",
            "action_catalog_digest",
            "selector_protocol_digest",
            "selector_rule",
            "acquisition_evidence_digest",
            "protocol_digest",
            "result_digest",
            "scenario",
            "energy_reference_joules",
            "bench_time_reference_seconds",
            "action_resources",
            "action_costs",
            "action_evaluations",
        },
        "prospective costed scorecard",
    )
    if (
        type(item["schema_version"]) is not int
        or item["schema_version"] != PROSPECTIVE_COST_SCHEMA_VERSION
    ):
        raise ValueError("prospective cost schema version is unsupported")
    fixed_values = {
        "protocol_version": PROSPECTIVE_COST_PROTOCOL_VERSION,
        "formula": PROSPECTIVE_COST_FORMULA,
        "padded_scoring_protocol": PROSPECTIVE_PADDED_SCORING_PROTOCOL,
        "energy_convention": PROSPECTIVE_COST_ENERGY_CONVENTION,
        "time_convention": PROSPECTIVE_COST_TIME_CONVENTION,
        "instrumentation_convention": (
            PROSPECTIVE_COST_INSTRUMENTATION_CONVENTION
        ),
        "nominal_energy_protocol": NOMINAL_SELECTION_COST_PROTOCOL,
    }
    if any(item[name] != value for name, value in fixed_values.items()):
        raise ValueError("prospective cost protocol identity is unsupported")
    if not isinstance(uncertainty_result, ProspectiveUncertaintyResult):
        raise ValueError("prospective cost payload needs its Step-2 result")
    expected_refs = {
        "physical_protocol_digest": (
            uncertainty_result.acquisition_evidence.physical_protocol_digest
        ),
        "uncertainty_protocol_digest": uncertainty_result.protocol_digest,
        "uncertainty_result_digest": uncertainty_result.result_digest,
        "acquisition_evidence_digest": (
            uncertainty_result.acquisition_evidence.evidence_digest
        ),
    }
    if any(item[name] != value for name, value in expected_refs.items()):
        raise ValueError("prospective cost payload does not match its Step-2 result")
    selector_rule = prospective_selector_rule_from_payload(item["selector_rule"])
    if (
        item["action_catalog_digest"] != prospective_action_catalog_digest()
        or item["selector_protocol_digest"]
        != prospective_selector_protocol_digest(selector_rule)
    ):
        raise ValueError("prospective cost payload selector provenance is invalid")
    for name in ("action_resources", "action_costs", "action_evaluations"):
        if not isinstance(item[name], list):
            raise ValueError(f"prospective cost {name} must be a list")
    return ProspectiveCostedScorecard(
        uncertainty_result=uncertainty_result,
        scenario=_scenario_from_payload(item["scenario"]),
        selector_rule=selector_rule,
        protocol_digest=item["protocol_digest"],
        energy_reference_joules=item["energy_reference_joules"],
        bench_time_reference_seconds=item["bench_time_reference_seconds"],
        action_resources=tuple(
            _action_resources_from_payload(value)
            for value in item["action_resources"]
        ),
        action_costs=tuple(
            _action_cost_from_payload(value) for value in item["action_costs"]
        ),
        action_evaluations=tuple(
            _action_evaluation_from_payload(value)
            for value in item["action_evaluations"]
        ),
        result_digest=item["result_digest"],
    )


__all__ = [
    "PRIMARY_PROSPECTIVE_COST_SCENARIO",
    "PROSPECTIVE_COST_ENERGY_CONVENTION",
    "PROSPECTIVE_COST_FORMULA",
    "PROSPECTIVE_COST_INSTRUMENTATION_CONVENTION",
    "PROSPECTIVE_COST_PRIMARY_SCENARIO_NAME",
    "PROSPECTIVE_COST_PROTOCOL_VERSION",
    "PROSPECTIVE_RESET_SENSITIVITY_SECONDS",
    "PROSPECTIVE_COST_SCHEMA_VERSION",
    "PROSPECTIVE_COST_SENSITIVITY_SCENARIOS",
    "PROSPECTIVE_COST_TIME_CONVENTION",
    "ProspectiveActionCost",
    "ProspectiveActionResources",
    "ProspectiveCostScenario",
    "ProspectiveCostedScorecard",
    "ProspectiveRunResources",
    "cost_prospective_uncertainty",
    "prospective_action_resources",
    "prospective_cost_protocol_digest",
    "prospective_costed_scorecard_from_payload",
    "prospective_costed_scorecard_payload",
    "select_costed_prospective_action",
]
