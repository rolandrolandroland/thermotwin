"""Reproducible experiment definitions for the ThermoTwin forward model."""

from dataclasses import dataclass
from typing import NamedTuple

from ..core.controls import CurrentInput
from .two_node_diagnostics import TrajectoryDiagnostics, evaluate_trajectory
from ..physics.thermoelectric import ThermoelectricParameters
from ..physics.two_node import (
    TemperatureTrajectory,
    TwoNodeThermalParameters,
    integrate_two_node,
)


@dataclass(frozen=True)
class TwoNodeExperiment:
    """All fixed inputs required to reproduce one two-node simulation."""

    thermoelectric_parameters: ThermoelectricParameters
    thermal_parameters: TwoNodeThermalParameters
    initial_cold_temperature: float
    initial_hot_temperature: float
    duration: float
    time_step: float
    current: CurrentInput
    cold_reservoir_temperature: float
    hot_reservoir_temperature: float
    cold_external_heat: float = 0.0
    hot_external_heat: float = 0.0


class ExperimentResult(NamedTuple):
    """Temperature trajectory and aligned derived histories."""

    trajectory: TemperatureTrajectory
    diagnostics: TrajectoryDiagnostics


def run_two_node_experiment(
    experiment: TwoNodeExperiment,
) -> ExperimentResult:
    """Run an experiment with RK4 and evaluate all derived histories."""

    trajectory = integrate_two_node(
        experiment.thermoelectric_parameters,
        experiment.thermal_parameters,
        initial_cold_temperature=experiment.initial_cold_temperature,
        initial_hot_temperature=experiment.initial_hot_temperature,
        duration=experiment.duration,
        time_step=experiment.time_step,
        current=experiment.current,
        cold_reservoir_temperature=experiment.cold_reservoir_temperature,
        hot_reservoir_temperature=experiment.hot_reservoir_temperature,
        cold_external_heat=experiment.cold_external_heat,
        hot_external_heat=experiment.hot_external_heat,
    )
    diagnostics = evaluate_trajectory(
        experiment.thermoelectric_parameters,
        trajectory,
        current=experiment.current,
    )
    return ExperimentResult(
        trajectory=trajectory,
        diagnostics=diagnostics,
    )


def constant_current_reference_experiment() -> TwoNodeExperiment:
    """Return the agreed 1 A, 60 s forward-PINN reference case.

    Keeping these values in one named function prevents the conventional
    reference trajectory and the learned model from silently using different
    physical inputs.
    """

    return TwoNodeExperiment(
        thermoelectric_parameters=ThermoelectricParameters(
            seebeck_coefficient=0.05,
            electrical_resistance=2.0,
            thermal_conductance=0.5,
        ),
        thermal_parameters=TwoNodeThermalParameters(
            cold_thermal_capacitance=100.0,
            hot_thermal_capacitance=200.0,
            cold_reservoir_conductance=2.0,
            hot_reservoir_conductance=4.0,
        ),
        initial_cold_temperature=300.0,
        initial_hot_temperature=300.0,
        duration=60.0,
        time_step=0.1,
        current=1.0,
        cold_reservoir_temperature=300.0,
        hot_reservoir_temperature=300.0,
    )
