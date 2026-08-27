"""Reference integration and reproducible experiments for the 1-D leg model."""

from dataclasses import dataclass, replace
import math
from typing import NamedTuple, Optional, Sequence, Tuple

from ..core.controls import CurrentInput, PiecewiseConstantCurrent, current_at
from ..numerics.integration import IntegrationDivergenceError
from ..physics.distributed import (
    ConstantProperty,
    DistributedFaceThermalParameters,
    DistributedLegGeometry,
    DistributedStateDiagnostics,
    DistributedThermoelectricMaterial,
    PiecewiseLinearProperty,
    distributed_leg_rhs,
    distributed_stored_energy,
    evaluate_distributed_state,
    linear_cell_temperatures,
    recommended_explicit_time_step,
)


class DistributedTemperatureTrajectory(NamedTuple):
    """Time-aligned face temperatures and internal cell profiles."""

    time: Tuple[float, ...]
    cold_face: Tuple[float, ...]
    cells: Tuple[Tuple[float, ...], ...]
    hot_face: Tuple[float, ...]


class DistributedTrajectoryDiagnostics(NamedTuple):
    """Time-aligned derived histories for a distributed trajectory."""

    current: Tuple[float, ...]
    voltage: Tuple[float, ...]
    electrical_power: Tuple[float, ...]
    cold_side_heat: Tuple[float, ...]
    hot_side_heat: Tuple[float, ...]
    stored_energy: Tuple[float, ...]
    instantaneous_energy_balance_residual: Tuple[float, ...]


class DistributedExperimentResult(NamedTuple):
    trajectory: DistributedTemperatureTrajectory
    diagnostics: DistributedTrajectoryDiagnostics


@dataclass(frozen=True)
class DistributedLegExperiment:
    """Complete inputs for one coupled face-node/distributed-leg experiment."""

    material: DistributedThermoelectricMaterial
    geometry: DistributedLegGeometry
    face_parameters: DistributedFaceThermalParameters
    cell_count: int
    initial_cold_face_temperature: float
    initial_hot_face_temperature: float
    duration: float
    time_step: float
    current: CurrentInput
    cold_reservoir_temperature: float
    hot_reservoir_temperature: float
    initial_cell_temperatures: Optional[Tuple[float, ...]] = None
    cold_external_heat: float = 0.0
    hot_external_heat: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cell_count, int)
            or isinstance(self.cell_count, bool)
            or self.cell_count < 2
        ):
            raise ValueError("cell count must be an integer of at least two")
        for name, value in (
            ("duration", self.duration),
            ("time step", self.time_step),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("initial cold-face temperature", self.initial_cold_face_temperature),
            ("initial hot-face temperature", self.initial_hot_face_temperature),
            ("cold reservoir temperature", self.cold_reservoir_temperature),
            ("hot reservoir temperature", self.hot_reservoir_temperature),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive kelvin")
        for name, value in (
            ("cold external heat", self.cold_external_heat),
            ("hot external heat", self.hot_external_heat),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        current_at(self.current, 0.0)
        if self.initial_cell_temperatures is not None:
            cells = tuple(float(value) for value in self.initial_cell_temperatures)
            object.__setattr__(self, "initial_cell_temperatures", cells)
            if len(cells) != self.cell_count:
                raise ValueError(
                    "initial cell-temperature count must match cell count"
                )
            if any(not math.isfinite(value) or value <= 0.0 for value in cells):
                raise ValueError(
                    "initial cell temperatures must be finite positive kelvin"
                )

    def initial_cells(self) -> Tuple[float, ...]:
        if self.initial_cell_temperatures is not None:
            return self.initial_cell_temperatures
        return linear_cell_temperatures(
            self.initial_cold_face_temperature,
            self.initial_hot_face_temperature,
            self.cell_count,
        )


def _flatten_rates(rates) -> Tuple[float, ...]:
    return (rates.cold_face,) + rates.cells + (rates.hot_face,)


def _validate_state(state: Sequence[float], *, time: float, stage: str) -> None:
    if any(not math.isfinite(value) for value in state):
        reason = "a temperature became nonfinite"
    elif any(value <= 0.0 for value in state):
        reason = "a temperature left the positive-kelvin domain"
    else:
        return
    raise IntegrationDivergenceError(
        f"distributed-leg integration diverged near t={time:.9g} s during "
        f"{stage}: {reason}; reduce the time step or revise the inputs"
    )


def _rates_for_state(
    experiment: DistributedLegExperiment,
    state: Tuple[float, ...],
    *,
    step_current: float,
    time: float,
    stage: str,
) -> Tuple[float, ...]:
    _validate_state(state, time=time, stage=stage)
    try:
        rates = distributed_leg_rhs(
            experiment.material,
            experiment.geometry,
            experiment.face_parameters,
            cold_face_temperature=state[0],
            cell_temperatures=state[1:-1],
            hot_face_temperature=state[-1],
            current=step_current,
            cold_reservoir_temperature=experiment.cold_reservoir_temperature,
            hot_reservoir_temperature=experiment.hot_reservoir_temperature,
            cold_external_heat=experiment.cold_external_heat,
            hot_external_heat=experiment.hot_external_heat,
        )
    except (OverflowError, ValueError) as error:
        if isinstance(error, ValueError) and "positive" not in str(error) and "finite" not in str(error):
            raise
        raise IntegrationDivergenceError(
            f"distributed-leg integration diverged near t={time:.9g} s "
            f"during {stage}: {error}; reduce the time step or revise the inputs"
        ) from error
    flat = _flatten_rates(rates)
    if any(not math.isfinite(value) for value in flat):
        raise IntegrationDivergenceError(
            f"distributed-leg integration diverged near t={time:.9g} s "
            f"during {stage}: a temperature rate became nonfinite; reduce the "
            "time step or revise the inputs"
        )
    return flat


def integrate_distributed_leg(
    experiment: DistributedLegExperiment,
) -> DistributedTemperatureTrajectory:
    """Integrate the semidiscrete energy balances with transition-split RK4."""

    states = [
        (
            experiment.initial_cold_face_temperature,
            *experiment.initial_cells(),
            experiment.initial_hot_face_temperature,
        )
    ]
    times = [0.0]
    while times[-1] < experiment.duration:
        start_time = times[-1]
        end_time = min(start_time + experiment.time_step, experiment.duration)
        if isinstance(experiment.current, PiecewiseConstantCurrent):
            transition = experiment.current.next_transition_after(start_time)
            if transition is not None:
                end_time = min(end_time, transition)
        step = end_time - start_time
        if step <= 0.0:
            raise RuntimeError("distributed integration time failed to advance")
        state = states[-1]
        step_current = current_at(experiment.current, start_time)

        k1 = _rates_for_state(
            experiment,
            state,
            step_current=step_current,
            time=start_time,
            stage="RK4 k1",
        )
        k2_state = tuple(value + 0.5 * step * rate for value, rate in zip(state, k1))
        k2 = _rates_for_state(
            experiment,
            k2_state,
            step_current=step_current,
            time=start_time + 0.5 * step,
            stage="RK4 k2",
        )
        k3_state = tuple(value + 0.5 * step * rate for value, rate in zip(state, k2))
        k3 = _rates_for_state(
            experiment,
            k3_state,
            step_current=step_current,
            time=start_time + 0.5 * step,
            stage="RK4 k3",
        )
        k4_state = tuple(value + step * rate for value, rate in zip(state, k3))
        k4 = _rates_for_state(
            experiment,
            k4_state,
            step_current=step_current,
            time=end_time,
            stage="RK4 k4",
        )
        next_state = tuple(
            value + step * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for value, a, b, c, d in zip(state, k1, k2, k3, k4)
        )
        _validate_state(next_state, time=end_time, stage="RK4 update")
        times.append(end_time)
        states.append(next_state)

    return DistributedTemperatureTrajectory(
        time=tuple(times),
        cold_face=tuple(state[0] for state in states),
        cells=tuple(tuple(state[1:-1]) for state in states),
        hot_face=tuple(state[-1] for state in states),
    )


def evaluate_distributed_trajectory(
    experiment: DistributedLegExperiment,
    trajectory: DistributedTemperatureTrajectory,
) -> DistributedTrajectoryDiagnostics:
    """Evaluate all derived histories without numerically differentiating data."""

    expected_length = len(trajectory.time)
    if not (
        len(trajectory.cold_face)
        == len(trajectory.cells)
        == len(trajectory.hot_face)
        == expected_length
    ):
        raise ValueError("trajectory histories must have equal lengths")
    currents = tuple(current_at(experiment.current, time) for time in trajectory.time)
    state_diagnostics: Tuple[DistributedStateDiagnostics, ...] = tuple(
        evaluate_distributed_state(
            experiment.material,
            experiment.geometry,
            experiment.face_parameters,
            cold_face_temperature=cold,
            cell_temperatures=cells,
            hot_face_temperature=hot,
            current=current,
            cold_reservoir_temperature=experiment.cold_reservoir_temperature,
            hot_reservoir_temperature=experiment.hot_reservoir_temperature,
            cold_external_heat=experiment.cold_external_heat,
            hot_external_heat=experiment.hot_external_heat,
        )
        for cold, cells, hot, current in zip(
            trajectory.cold_face,
            trajectory.cells,
            trajectory.hot_face,
            currents,
        )
    )
    return DistributedTrajectoryDiagnostics(
        current=currents,
        voltage=tuple(item.voltage for item in state_diagnostics),
        electrical_power=tuple(item.electrical_power for item in state_diagnostics),
        cold_side_heat=tuple(item.cold_side_heat for item in state_diagnostics),
        hot_side_heat=tuple(item.hot_side_heat for item in state_diagnostics),
        stored_energy=tuple(
            distributed_stored_energy(
                experiment.material,
                experiment.geometry,
                experiment.face_parameters,
                cold_face_temperature=cold,
                cell_temperatures=cells,
                hot_face_temperature=hot,
            )
            for cold, cells, hot in zip(
                trajectory.cold_face, trajectory.cells, trajectory.hot_face
            )
        ),
        instantaneous_energy_balance_residual=tuple(
            item.energy_balance_residual for item in state_diagnostics
        ),
    )


def run_distributed_leg_experiment(
    experiment: DistributedLegExperiment,
) -> DistributedExperimentResult:
    trajectory = integrate_distributed_leg(experiment)
    return DistributedExperimentResult(
        trajectory=trajectory,
        diagnostics=evaluate_distributed_trajectory(experiment, trajectory),
    )


def reference_distributed_material(
    *, temperature_dependent: bool = False
) -> DistributedThermoelectricMaterial:
    """Return a generic Bi2Te3-like synthetic material, not a catalog fit."""

    if temperature_dependent:
        temperatures = (285.0, 300.0, 315.0)
        alpha = PiecewiseLinearProperty(
            temperatures, (195.0e-6, 200.0e-6, 205.0e-6)
        )
        resistivity = PiecewiseLinearProperty(
            temperatures, (1.05e-5, 1.00e-5, 0.96e-5)
        )
        conductivity = PiecewiseLinearProperty(
            temperatures, (1.45, 1.50, 1.56)
        )
    else:
        alpha = ConstantProperty(200.0e-6)
        resistivity = ConstantProperty(1.00e-5)
        conductivity = ConstantProperty(1.50)
    return DistributedThermoelectricMaterial(
        seebeck_coefficient=alpha,
        electrical_resistivity=resistivity,
        thermal_conductivity=conductivity,
        mass_density=7700.0,
        specific_heat_capacity=150.0,
    )


def distributed_reference_experiment(
    *,
    current: CurrentInput = 0.8,
    temperature_dependent: bool = False,
    cold_reservoir_temperature: float = 295.0,
    hot_reservoir_temperature: float = 305.0,
    duration: float = 1.0,
    cell_count: int = 8,
    time_step: Optional[float] = None,
) -> DistributedLegExperiment:
    """Return the small CPU-first reference case for distributed development."""

    material = reference_distributed_material(
        temperature_dependent=temperature_dependent
    )
    geometry = DistributedLegGeometry(length=1.5e-3, area=2.25e-6)
    if time_step is None:
        time_step = min(
            0.002,
            recommended_explicit_time_step(
                material,
                geometry,
                cell_count=cell_count,
                temperature_range=(285.0, 315.0),
            ),
        )
    return DistributedLegExperiment(
        material=material,
        geometry=geometry,
        face_parameters=DistributedFaceThermalParameters(
            cold_thermal_capacitance=0.05,
            hot_thermal_capacitance=0.05,
            cold_reservoir_conductance=0.01,
            hot_reservoir_conductance=0.01,
        ),
        cell_count=cell_count,
        initial_cold_face_temperature=295.0,
        initial_hot_face_temperature=305.0,
        duration=duration,
        time_step=time_step,
        current=current,
        cold_reservoir_temperature=cold_reservoir_temperature,
        hot_reservoir_temperature=hot_reservoir_temperature,
    )


def distributed_identifiability_experiments() -> Tuple[DistributedLegExperiment, ...]:
    """Return relaxation, positive-pulse, negative-pulse, and lifted regimes.

    The suite is intentionally small. Its role is to span current parity and a
    useful temperature interval before any inverse neural network is trained.
    """

    base = distributed_reference_experiment(
        temperature_dependent=True,
        current=0.0,
        duration=0.8,
        cell_count=6,
        time_step=0.0015,
    )
    relaxation = replace(
        base,
        initial_cold_face_temperature=300.0,
        initial_hot_face_temperature=300.0,
        cold_reservoir_temperature=290.0,
        hot_reservoir_temperature=310.0,
    )
    positive_pulse = replace(
        base,
        current=PiecewiseConstantCurrent.pulse(
            start_time=0.10,
            end_time=0.60,
            pulse_current=0.8,
        ),
    )
    negative_pulse = replace(
        base,
        current=PiecewiseConstantCurrent.pulse(
            start_time=0.10,
            end_time=0.60,
            pulse_current=-0.8,
        ),
    )
    lifted = replace(
        base,
        current=0.4,
        initial_cold_face_temperature=290.0,
        initial_hot_face_temperature=310.0,
        cold_reservoir_temperature=290.0,
        hot_reservoir_temperature=310.0,
    )
    return relaxation, positive_pulse, negative_pulse, lifted


def distributed_inverse_constant_experiments() -> Tuple[DistributedLegExperiment, ...]:
    """Return constant regimes suitable for a shared-property inverse PINN."""

    base = distributed_reference_experiment(
        temperature_dependent=True,
        current=0.0,
        duration=0.4,
        cell_count=5,
        time_step=0.0015,
    )
    return (
        replace(
            base,
            current=0.0,
            initial_cold_face_temperature=300.0,
            initial_hot_face_temperature=300.0,
            cold_reservoir_temperature=290.0,
            hot_reservoir_temperature=310.0,
        ),
        replace(base, current=0.8),
        replace(base, current=-0.8),
        replace(
            base,
            current=0.4,
            initial_cold_face_temperature=290.0,
            initial_hot_face_temperature=310.0,
            cold_reservoir_temperature=290.0,
            hot_reservoir_temperature=310.0,
        ),
    )
