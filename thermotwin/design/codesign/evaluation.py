"""Module scaling, steady evaluation, and current selection for co-design."""

from dataclasses import replace
import math
from typing import Optional, Tuple

from ..materials import MaterialSample
from ...physics.four_node import (
    FourNodeContactSteadyState,
    FourNodeContactThermalParameters,
    four_node_contact_steady_state_from_current_moments,
)
from ...physics.thermoelectric import ThermoelectricParameters
from ..power_electronics import (
    AveragedThermoelectricRates,
    CurrentMoments,
    averaged_thermoelectric_rates,
    smoothed_pwm_current_moments,
)
from .models import (
    CURRENT_DENSITY_BINDING_UTILIZATION,
    ApplicationSpecification,
    DesignOperatingPoint,
    ModuleAssemblyAssumptions,
    ModuleElectricalResistanceComponents,
    ModuleGeometry,
    PropertyMultipliers,
    PrototypeDesign,
)


def module_electrical_resistance_components(
    p_material: MaterialSample,
    n_material: MaterialSample,
    geometry: ModuleGeometry,
    *,
    assembly: ModuleAssemblyAssumptions = ModuleAssemblyAssumptions(),
    multipliers: PropertyMultipliers = PropertyMultipliers(),
) -> ModuleElectricalResistanceComponents:
    """Return bulk and metal/thermoelectric interface resistance."""

    if p_material.carrier_type != "p" or n_material.carrier_type != "n":
        raise ValueError("module requires one p-type and one n-type material")
    p_resistivity = 1.0 / (
        p_material.electrical_conductivity
        * multipliers.p_electrical_conductivity
    )
    n_resistivity = 1.0 / (
        n_material.electrical_conductivity
        * multipliers.n_electrical_conductivity
    )
    bulk_leg_resistance = (
        geometry.couple_count
        * geometry.leg_length
        / geometry.leg_area
        * (p_resistivity + n_resistivity)
    )
    electrical_contact_resistance = (
        4.0
        * geometry.couple_count
        * assembly.specific_electrical_contact_resistivity
        / geometry.leg_area
    )
    total_resistance = bulk_leg_resistance + electrical_contact_resistance
    contact_fraction = (
        electrical_contact_resistance / total_resistance
        if total_resistance > 0.0
        else 0.0
    )
    return ModuleElectricalResistanceComponents(
        bulk_leg_resistance,
        electrical_contact_resistance,
        total_resistance,
        contact_fraction,
    )


def module_thermoelectric_parameters(
    p_material: MaterialSample,
    n_material: MaterialSample,
    geometry: ModuleGeometry,
    *,
    assembly: ModuleAssemblyAssumptions = ModuleAssemblyAssumptions(),
    multipliers: PropertyMultipliers = PropertyMultipliers(),
) -> ThermoelectricParameters:
    """Scale material properties into constant module alpha, R, and K."""

    alpha_pair = (
        p_material.seebeck_coefficient * multipliers.p_seebeck
        - n_material.seebeck_coefficient * multipliers.n_seebeck
    )
    resistance = module_electrical_resistance_components(
        p_material,
        n_material,
        geometry,
        assembly=assembly,
        multipliers=multipliers,
    )
    leg_conductance = (
        geometry.couple_count
        * geometry.leg_area
        / geometry.leg_length
        * (
            p_material.thermal_conductivity
            * multipliers.p_thermal_conductivity
            + n_material.thermal_conductivity
            * multipliers.n_thermal_conductivity
        )
    )
    return ThermoelectricParameters(
        seebeck_coefficient=geometry.couple_count * alpha_pair,
        electrical_resistance=resistance.total_resistance,
        thermal_conductance=(
            leg_conductance + assembly.parasitic_thermal_conductance
        ),
    )


def prototype_cost_index(design: PrototypeDesign) -> float:
    """Return an explicit relative build-burden proxy, not a dollar estimate."""

    baseline_volume = 2.0 * 120.0 * 1.6e-6 * 1.5e-3
    material_volume_term = design.geometry.active_material_volume / baseline_volume
    assembly_term = design.geometry.couple_count / 120.0
    cold_exchanger_term = design.cold_exchanger_conductance / 2.5
    hot_exchanger_term = design.hot_exchanger_conductance / 5.0
    return (
        0.40 * material_volume_term
        + 0.20 * assembly_term
        + 0.20 * cold_exchanger_term
        + 0.20 * hot_exchanger_term
    )


def averaged_contact_steady_state_for_parameters(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    current: CurrentMoments,
    *,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
) -> FourNodeContactSteadyState:
    """Delegate arbitrary module/current moments to the shared steady kernel."""

    return four_node_contact_steady_state_from_current_moments(
        thermoelectric_parameters,
        thermal_parameters,
        mean_current=current.mean_current,
        mean_square_current=current.mean_square_current,
        cold_reservoir_temperature=cold_reservoir_temperature,
        hot_reservoir_temperature=hot_reservoir_temperature,
    )


def _application_utility(
    application: ApplicationSpecification,
    *,
    cooling_rate: float,
    wall_cop: Optional[float],
    heat_flux: float,
    cost_index: float,
    supply_power: float,
    peak_voltage: float,
    maximum_peak_voltage: float,
) -> Tuple[bool, float]:
    cop_value = wall_cop if wall_cop is not None else 0.0
    violations = (
        max(0.0, application.minimum_cooling_rate - cooling_rate)
        / max(application.minimum_cooling_rate, 1.0),
        max(0.0, application.minimum_wall_cop - cop_value)
        / max(application.minimum_wall_cop, 1.0),
        max(0.0, supply_power - application.maximum_supply_power)
        / application.maximum_supply_power,
        max(0.0, peak_voltage - maximum_peak_voltage) / maximum_peak_voltage,
    )
    feasible = not any(value > 0.0 for value in violations)
    if not feasible:
        return False, -1.0 - sum(violations)
    if application.objective == "efficiency":
        utility = cop_value / math.sqrt(cost_index)
    elif application.objective == "balanced":
        utility = cooling_rate * cop_value / cost_index
    else:
        utility = heat_flux * math.sqrt(cop_value) / math.sqrt(cost_index)
    return True, utility


def evaluate_design_current(
    design: PrototypeDesign,
    application: ApplicationSpecification,
    mean_current: float,
    *,
    assembly: ModuleAssemblyAssumptions = ModuleAssemblyAssumptions(),
    multipliers: PropertyMultipliers = PropertyMultipliers(),
    electrical_contact_resistivity_multiplier: float = 1.0,
    contact_multiplier: float = 1.0,
    cold_exchanger_multiplier: float = 1.0,
    hot_exchanger_multiplier: float = 1.0,
    converter_efficiency: Optional[float] = None,
) -> DesignOperatingPoint:
    """Evaluate one fixed mean-current, smoothed-PWM operating point."""

    if not math.isfinite(mean_current) or mean_current <= 0.0:
        raise ValueError("mean current must be finite and positive")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (
            electrical_contact_resistivity_multiplier,
            contact_multiplier,
            cold_exchanger_multiplier,
            hot_exchanger_multiplier,
        )
    ):
        raise ValueError("interface multipliers must be finite and positive")
    efficiency = (
        assembly.converter_efficiency
        if converter_efficiency is None
        else converter_efficiency
    )
    if not math.isfinite(efficiency) or not 0.0 < efficiency <= 1.0:
        raise ValueError("converter efficiency must lie in (0, 1]")
    effective_assembly = replace(
        assembly,
        specific_electrical_contact_resistivity=(
            assembly.specific_electrical_contact_resistivity
            * electrical_contact_resistivity_multiplier
        ),
    )
    current = smoothed_pwm_current_moments(
        mean_current,
        effective_assembly.pwm_ripple_peak_to_peak_fraction,
    )
    resistance = module_electrical_resistance_components(
        design.p_material,
        design.n_material,
        design.geometry,
        assembly=effective_assembly,
        multipliers=multipliers,
    )
    parameters = module_thermoelectric_parameters(
        design.p_material,
        design.n_material,
        design.geometry,
        assembly=effective_assembly,
        multipliers=multipliers,
    )
    thermal = FourNodeContactThermalParameters(
        cold_face_thermal_capacitance=1.0,
        hot_face_thermal_capacitance=1.0,
        cold_exchanger_thermal_capacitance=1.0,
        hot_exchanger_thermal_capacitance=1.0,
        cold_contact_resistance=(
            design.symmetric_contact_resistance * contact_multiplier
        ),
        hot_contact_resistance=(
            design.symmetric_contact_resistance * contact_multiplier
        ),
        cold_reservoir_conductance=(
            design.cold_exchanger_conductance * cold_exchanger_multiplier
        ),
        hot_reservoir_conductance=(
            design.hot_exchanger_conductance * hot_exchanger_multiplier
        ),
    )
    half_lift = 0.5 * application.external_temperature_lift
    cold_reservoir = 300.0 - half_lift
    hot_reservoir = 300.0 + half_lift
    state = averaged_contact_steady_state_for_parameters(
        parameters,
        thermal,
        current,
        cold_reservoir_temperature=cold_reservoir,
        hot_reservoir_temperature=hot_reservoir,
    )
    rates: AveragedThermoelectricRates = averaged_thermoelectric_rates(
        seebeck_coefficient=parameters.seebeck_coefficient,
        electrical_resistance=parameters.electrical_resistance,
        thermal_conductance=parameters.thermal_conductance,
        current=current,
        hot_temperature=state.hot_face,
        cold_temperature=state.cold_face,
    )
    delivered_cooling = thermal.cold_reservoir_conductance * (
        cold_reservoir - state.cold_exchanger
    )
    delivered_heating = thermal.hot_reservoir_conductance * (
        state.hot_exchanger - hot_reservoir
    )
    if not math.isclose(delivered_cooling, rates.cold_heat, abs_tol=1e-9):
        raise RuntimeError("cold-side steady energy balance did not close")
    if not math.isclose(delivered_heating, rates.hot_heat, abs_tol=1e-9):
        raise RuntimeError("hot-side steady energy balance did not close")
    supply_power = (
        rates.module_electrical_power / efficiency
        + effective_assembly.fixed_converter_loss
    )
    wall_cop = (
        delivered_cooling / supply_power
        if delivered_cooling > 0.0 and supply_power > 0.0
        else None
    )
    footprint = design.geometry.estimated_footprint_area
    heat_flux = delivered_cooling / (footprint * 1.0e4)
    cost_index = prototype_cost_index(design)
    peak_voltage = (
        parameters.seebeck_coefficient * (state.hot_face - state.cold_face)
        + current.peak_current * parameters.electrical_resistance
    )
    peak_current_density = current.peak_current / design.geometry.leg_area
    current_density_utilization = (
        peak_current_density / effective_assembly.maximum_current_density
    )
    current_density_ok = current_density_utilization <= 1.0 + 1e-12
    current_density_binding = (
        current_density_utilization >= CURRENT_DENSITY_BINDING_UTILIZATION
    )
    feasible, utility = _application_utility(
        application,
        cooling_rate=delivered_cooling,
        wall_cop=wall_cop,
        heat_flux=heat_flux,
        cost_index=cost_index,
        supply_power=supply_power,
        peak_voltage=peak_voltage,
        maximum_peak_voltage=effective_assembly.maximum_peak_voltage,
    )
    if not current_density_ok:
        feasible = False
        utility = min(utility, -1.0 - (current_density_utilization - 1.0))
    return DesignOperatingPoint(
        design,
        application,
        parameters,
        resistance.bulk_leg_resistance,
        resistance.electrical_contact_resistance,
        current.mean_current,
        current.peak_current,
        state.cold_face,
        state.hot_face,
        state.cold_exchanger,
        state.hot_exchanger,
        delivered_cooling,
        delivered_heating,
        rates.module_electrical_power,
        supply_power,
        wall_cop,
        heat_flux,
        cost_index,
        peak_voltage,
        peak_current_density,
        current_density_utilization,
        current_density_binding,
        feasible,
        utility,
    )


def optimize_design_current(
    design: PrototypeDesign,
    application: ApplicationSpecification,
    *,
    grid_size: int = 28,
    assembly: ModuleAssemblyAssumptions = ModuleAssemblyAssumptions(),
) -> DesignOperatingPoint:
    """Select the best feasible grid current, or least-violating point."""

    if isinstance(grid_size, bool) or grid_size < 2:
        raise ValueError("current grid size must be at least two")
    maximum_mean_current = (
        assembly.maximum_current_density
        * design.geometry.leg_area
        / (1.0 + 0.5 * assembly.pwm_ripple_peak_to_peak_fraction)
    )
    minimum_current = min(0.05, 0.05 * maximum_mean_current)
    currents = tuple(
        minimum_current
        + (maximum_mean_current - minimum_current) * index / (grid_size - 1)
        for index in range(grid_size)
    )
    points = tuple(
        evaluate_design_current(
            design,
            application,
            current,
            assembly=assembly,
        )
        for current in currents
    )
    return max(points, key=lambda point: point.utility)
