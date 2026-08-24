"""Reproducible experiments for the contact-aware four-node model."""

from dataclasses import dataclass
from typing import NamedTuple

from .four_node_diagnostics import (
    ContactTrajectoryDiagnostics,
    evaluate_contact_trajectory,
)
from ..physics.four_node import (
    FourNodeContactTemperatureTrajectory,
    FourNodeContactThermalParameters,
    integrate_four_node_contact,
)
from ..core.controls import CurrentInput
from ..physics.thermoelectric import ThermoelectricParameters


@dataclass(frozen=True)
class FourNodeContactExperiment:
    """All fixed inputs required for one contact-aware simulation."""

    thermoelectric_parameters: ThermoelectricParameters
    thermal_parameters: FourNodeContactThermalParameters
    initial_cold_face_temperature: float
    initial_hot_face_temperature: float
    initial_cold_exchanger_temperature: float
    initial_hot_exchanger_temperature: float
    duration: float
    time_step: float
    current: CurrentInput
    cold_reservoir_temperature: float
    hot_reservoir_temperature: float
    cold_external_heat: float = 0.0
    hot_external_heat: float = 0.0


class ContactExperimentResult(NamedTuple):
    """Four-node trajectory and aligned contact diagnostics."""

    trajectory: FourNodeContactTemperatureTrajectory
    diagnostics: ContactTrajectoryDiagnostics


def run_four_node_contact_experiment(
    experiment: FourNodeContactExperiment,
) -> ContactExperimentResult:
    """Run contact-aware RK4 and evaluate all derived histories."""

    trajectory = integrate_four_node_contact(
        experiment.thermoelectric_parameters,
        experiment.thermal_parameters,
        initial_cold_face_temperature=(
            experiment.initial_cold_face_temperature
        ),
        initial_hot_face_temperature=experiment.initial_hot_face_temperature,
        initial_cold_exchanger_temperature=(
            experiment.initial_cold_exchanger_temperature
        ),
        initial_hot_exchanger_temperature=(
            experiment.initial_hot_exchanger_temperature
        ),
        duration=experiment.duration,
        time_step=experiment.time_step,
        current=experiment.current,
        cold_reservoir_temperature=experiment.cold_reservoir_temperature,
        hot_reservoir_temperature=experiment.hot_reservoir_temperature,
        cold_external_heat=experiment.cold_external_heat,
        hot_external_heat=experiment.hot_external_heat,
    )
    diagnostics = evaluate_contact_trajectory(
        experiment.thermoelectric_parameters,
        experiment.thermal_parameters,
        trajectory,
        current=experiment.current,
        cold_reservoir_temperature=experiment.cold_reservoir_temperature,
        hot_reservoir_temperature=experiment.hot_reservoir_temperature,
        cold_external_heat=experiment.cold_external_heat,
        hot_external_heat=experiment.hot_external_heat,
    )
    return ContactExperimentResult(
        trajectory=trajectory,
        diagnostics=diagnostics,
    )


def constant_current_contact_reference_experiment(
) -> FourNodeContactExperiment:
    """Return the agreed generic 1 A, 60 s contact-aware reference case."""

    return FourNodeContactExperiment(
        thermoelectric_parameters=ThermoelectricParameters(
            seebeck_coefficient=0.05,
            electrical_resistance=2.0,
            thermal_conductance=0.5,
        ),
        thermal_parameters=FourNodeContactThermalParameters(
            cold_face_thermal_capacitance=50.0,
            hot_face_thermal_capacitance=100.0,
            cold_exchanger_thermal_capacitance=50.0,
            hot_exchanger_thermal_capacitance=100.0,
            cold_contact_resistance=0.25,
            hot_contact_resistance=0.25,
            cold_reservoir_conductance=2.0,
            hot_reservoir_conductance=4.0,
        ),
        initial_cold_face_temperature=300.0,
        initial_hot_face_temperature=300.0,
        initial_cold_exchanger_temperature=300.0,
        initial_hot_exchanger_temperature=300.0,
        duration=60.0,
        time_step=0.1,
        current=1.0,
        cold_reservoir_temperature=300.0,
        hot_reservoir_temperature=300.0,
    )
