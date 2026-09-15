"""Resource accounting for realistic operating-decision campaigns.

Selection needs a cost known before a hidden device is revealed, while final
evaluation should report the terminal energy that device actually consumed.
This module keeps those two quantities explicit: the nominal selection proxy
uses the frozen Stage 3 nominal model, and realized records use the hidden
device parameters and the truth-family trajectory for every diagnostic run.
"""

from dataclasses import dataclass, field, replace
import math
from typing import Optional, Tuple

from ..design.control_comparison import piecewise_electrical_energy
from ..simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
)
from .operating_decision import (
    ACQUISITION,
    FINAL_EVALUATION,
    VERIFICATION,
    OperatingRegime,
    initial_acquisition_regime,
)
from .operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
    RealismTruth,
    RealisticBlindedOperatingCase,
    RunInstrumentation,
    Stage3CaseId,
    _case_id,
    _truth_prediction,
    nominal_realistic_schedule_energy,
    realism_truth_for_trial,
)
from .sensor_model_discrimination import COLD_FACE


REALIZED_TERMINAL_ENERGY_PROTOCOL = "realism_truth_terminal_v1"
NOMINAL_SELECTION_COST_PROTOCOL = "stage3_nominal_terminal_proxy_v1"


def _validate_trial_identity(truth_condition: str, trial_index: int) -> None:
    if truth_condition not in STAGE3_TRUTH_CONDITIONS:
        raise ValueError("realized energy needs a known truth condition")
    if (
        not isinstance(trial_index, int)
        or isinstance(trial_index, bool)
        or trial_index < 0
    ):
        raise ValueError("trial index must be a nonnegative integer")


def _validate_regime_instrumentation(
    regime: OperatingRegime,
    instrumentation: RunInstrumentation,
) -> None:
    if not isinstance(regime, OperatingRegime):
        raise ValueError("terminal energy needs a valid operating regime")
    if not isinstance(instrumentation, RunInstrumentation):
        raise ValueError("terminal energy needs valid run instrumentation")
    if instrumentation.temporary_face_sensor != (COLD_FACE in regime.channels):
        raise ValueError(
            "temporary face-probe state must match the regime channels"
        )


def _validate_energy(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")


@dataclass(frozen=True)
class RealizedRunEnergy:
    """Terminal energy for one truth-level run, in joules."""

    truth_condition: str
    trial_index: int
    regime: OperatingRegime
    instrumentation: RunInstrumentation
    terminal_energy: float
    protocol: str = REALIZED_TERMINAL_ENERGY_PROTOCOL

    def __post_init__(self) -> None:
        _validate_trial_identity(self.truth_condition, self.trial_index)
        _validate_regime_instrumentation(self.regime, self.instrumentation)
        _validate_energy(self.terminal_energy, "realized terminal energy")
        if self.protocol != REALIZED_TERMINAL_ENERGY_PROTOCOL:
            raise ValueError("realized terminal-energy protocol is unknown")


@dataclass(frozen=True)
class RealizedCaseEnergy:
    """Realized diagnostic energy split into acquisition and verification."""

    case_id: Stage3CaseId
    truth_condition: str
    trial_index: int
    acquisition_runs: Tuple[RealizedRunEnergy, ...]
    verification_run: RealizedRunEnergy
    acquisition_energy: float = field(init=False)
    verification_energy: float = field(init=False)
    total_diagnostic_energy: float = field(init=False)

    def __post_init__(self) -> None:
        _validate_trial_identity(self.truth_condition, self.trial_index)
        acquisition_runs = tuple(self.acquisition_runs)
        object.__setattr__(self, "acquisition_runs", acquisition_runs)
        if (
            not isinstance(self.case_id, Stage3CaseId)
            or self.case_id.trial_index != self.trial_index
        ):
            raise ValueError("realized case energy has an invalid case identity")
        if not acquisition_runs or any(
            run.regime.phase != ACQUISITION for run in acquisition_runs
        ):
            raise ValueError("realized case energy needs acquisition runs")
        if self.verification_run.regime.phase != VERIFICATION:
            raise ValueError("realized case energy needs one verification run")
        all_runs = (*acquisition_runs, self.verification_run)
        if any(
            run.truth_condition != self.truth_condition
            or run.trial_index != self.trial_index
            for run in all_runs
        ):
            raise ValueError("realized run energy does not match its case")
        if len({run.regime.name for run in all_runs}) != len(all_runs):
            raise ValueError("realized case regime names must be distinct")

        acquisition_energy = sum(
            run.terminal_energy for run in acquisition_runs
        )
        verification_energy = self.verification_run.terminal_energy
        total_energy = acquisition_energy + verification_energy
        _validate_energy(acquisition_energy, "realized acquisition energy")
        _validate_energy(verification_energy, "realized verification energy")
        _validate_energy(total_energy, "realized diagnostic energy")
        object.__setattr__(self, "acquisition_energy", acquisition_energy)
        object.__setattr__(self, "verification_energy", verification_energy)
        object.__setattr__(self, "total_diagnostic_energy", total_energy)


@dataclass(frozen=True)
class NominalSelectionCostProxy:
    """Predeclared, device-independent terminal-energy proxy for selection."""

    regime: OperatingRegime
    instrumentation: RunInstrumentation
    terminal_energy: float
    protocol: str = NOMINAL_SELECTION_COST_PROTOCOL

    def __post_init__(self) -> None:
        _validate_regime_instrumentation(self.regime, self.instrumentation)
        _validate_energy(self.terminal_energy, "nominal selection-cost proxy")
        if self.protocol != NOMINAL_SELECTION_COST_PROTOCOL:
            raise ValueError("nominal selection-cost protocol is unknown")


def _realized_energy_for_truth(
    truth_condition: str,
    trial_index: int,
    regime: OperatingRegime,
    instrumentation: RunInstrumentation,
    truth: RealismTruth,
    config: OperatingDecisionRealismConfig,
) -> RealizedRunEnergy:
    prediction = _truth_prediction(
        truth_condition,
        truth,
        regime,
        instrumentation,
        config,
    )
    reference = constant_current_contact_reference_experiment()
    thermoelectric = replace(
        reference.thermoelectric_parameters,
        electrical_resistance=(
            reference.thermoelectric_parameters.electrical_resistance
            + truth.series_resistance
        ),
    )
    energy = piecewise_electrical_energy(
        prediction.time,
        prediction.cold_face,
        prediction.hot_face,
        thermoelectric,
        regime.current,
        start_time=prediction.time[0],
        end_time=prediction.time[-1],
    )
    return RealizedRunEnergy(
        truth_condition=truth_condition,
        trial_index=trial_index,
        regime=regime,
        instrumentation=instrumentation,
        terminal_energy=energy,
    )


def realized_regime_terminal_energy(
    truth_condition: str,
    trial_index: int,
    regime: OperatingRegime,
    instrumentation: RunInstrumentation,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> RealizedRunEnergy:
    """Simulate and integrate one run for the actual hidden device."""

    _validate_trial_identity(truth_condition, trial_index)
    _validate_regime_instrumentation(regime, instrumentation)
    truth = realism_truth_for_trial(config, trial_index)
    return _realized_energy_for_truth(
        truth_condition,
        trial_index,
        regime,
        instrumentation,
        truth,
        config,
    )


def realized_regime_terminal_energy_from_truth(
    truth_condition: str,
    trial_index: int,
    regime: OperatingRegime,
    instrumentation: RunInstrumentation,
    truth: RealismTruth,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> RealizedRunEnergy:
    """Integrate one run using an explicitly supplied versioned truth draw."""

    _validate_trial_identity(truth_condition, trial_index)
    _validate_regime_instrumentation(regime, instrumentation)
    if not isinstance(truth, RealismTruth):
        raise ValueError("realized energy needs a valid realism truth")
    return _realized_energy_for_truth(
        truth_condition,
        trial_index,
        regime,
        instrumentation,
        truth,
        config,
    )


def _validate_blinded_case(
    truth_condition: str,
    trial_index: int,
    case: RealisticBlindedOperatingCase,
    config: OperatingDecisionRealismConfig,
    *,
    expected_case_id: Optional[Stage3CaseId],
) -> None:
    if not isinstance(case, RealisticBlindedOperatingCase):
        raise ValueError("realized energy needs a realistic blinded case")
    if (
        case.case_id.trial_index != trial_index
        or case.case_id.policy_name != case.policy.name
        or (expected_case_id is not None and case.case_id != expected_case_id)
    ):
        raise ValueError("blinded case identity does not match the requested device")
    expected_acquisition = (
        initial_acquisition_regime(),
        *case.policy.additional_regimes,
    )
    if tuple(run.regime for run in case.acquisition_runs) != expected_acquisition:
        raise ValueError("blinded case acquisition regimes do not match its policy")
    if any(
        run.regime.phase != ACQUISITION for run in case.acquisition_runs
    ):
        raise ValueError("blinded case has a non-acquisition diagnostic run")
    if (
        case.verification_run.regime.phase != VERIFICATION
        or case.verification_run.regime.current != config.verification_current
    ):
        raise ValueError("blinded case has an invalid verification regime")
    if (
        case.final_regime.phase != FINAL_EVALUATION
        or case.final_regime.current != config.final_current
        or case.final_regime.channels
        or case.final_instrumentation.temporary_face_sensor
    ):
        raise ValueError("blinded case has an invalid untouched final regime")
    for run in (*case.acquisition_runs, case.verification_run):
        _validate_regime_instrumentation(run.regime, run.instrumentation)


def realized_blinded_case_energy(
    truth_condition: str,
    trial_index: int,
    case: RealisticBlindedOperatingCase,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> RealizedCaseEnergy:
    """Return realized diagnostic energy for one validated blinded case.

    The untouched final operating schedule is deliberately excluded: this
    record accounts only for evidence-gathering acquisition and verification.
    """

    _validate_trial_identity(truth_condition, trial_index)
    expected_id = _case_id(
        config,
        truth_condition,
        trial_index,
        case.policy.name,
    )
    _validate_blinded_case(
        truth_condition,
        trial_index,
        case,
        config,
        expected_case_id=expected_id,
    )
    truth = realism_truth_for_trial(config, trial_index)
    return realized_blinded_case_energy_from_truth(
        truth_condition,
        trial_index,
        case,
        truth,
        config,
        expected_case_id=expected_id,
    )


def realized_blinded_case_energy_from_truth(
    truth_condition: str,
    trial_index: int,
    case: RealisticBlindedOperatingCase,
    truth: RealismTruth,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
    *,
    expected_case_id: Optional[Stage3CaseId] = None,
) -> RealizedCaseEnergy:
    """Score diagnostic energy using a supplied truth from a versioned generator.

    ``expected_case_id`` lets a caller bind a corrected campaign token without
    forcing this accounting module to know how that token was constructed.
    """

    _validate_trial_identity(truth_condition, trial_index)
    if not isinstance(truth, RealismTruth):
        raise ValueError("realized energy needs a valid realism truth")
    _validate_blinded_case(
        truth_condition,
        trial_index,
        case,
        config,
        expected_case_id=expected_case_id,
    )
    acquisition_runs = tuple(
        _realized_energy_for_truth(
            truth_condition,
            trial_index,
            run.regime,
            run.instrumentation,
            truth,
            config,
        )
        for run in case.acquisition_runs
    )
    verification_run = _realized_energy_for_truth(
        truth_condition,
        trial_index,
        case.verification_run.regime,
        case.verification_run.instrumentation,
        truth,
        config,
    )
    return RealizedCaseEnergy(
        case_id=case.case_id,
        truth_condition=truth_condition,
        trial_index=trial_index,
        acquisition_runs=acquisition_runs,
        verification_run=verification_run,
    )


def nominal_selection_cost_proxy(
    regime: OperatingRegime,
    instrumentation: RunInstrumentation,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> NominalSelectionCostProxy:
    """Return the frozen nominal energy proxy available before device reveal."""

    _validate_regime_instrumentation(regime, instrumentation)
    energy = nominal_realistic_schedule_energy(
        regime.current,
        config,
        temporary_face_sensor=instrumentation.temporary_face_sensor,
    )
    return NominalSelectionCostProxy(
        regime=regime,
        instrumentation=instrumentation,
        terminal_energy=energy,
    )


__all__ = [
    "NOMINAL_SELECTION_COST_PROTOCOL",
    "REALIZED_TERMINAL_ENERGY_PROTOCOL",
    "NominalSelectionCostProxy",
    "RealizedCaseEnergy",
    "RealizedRunEnergy",
    "nominal_selection_cost_proxy",
    "realized_blinded_case_energy",
    "realized_blinded_case_energy_from_truth",
    "realized_regime_terminal_energy",
    "realized_regime_terminal_energy_from_truth",
]
