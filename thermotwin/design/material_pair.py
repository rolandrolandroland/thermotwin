"""Explicit-material steady evaluation shared by opt-in design studies."""

from dataclasses import dataclass
import math
from typing import NamedTuple, Optional, Tuple

from ..physics.four_node import FourNodeContactThermalParameters
from .materials import MaterialSample
from .power_electronics import (
    CurrentMoments,
    averaged_thermoelectric_rates,
    smoothed_pwm_current_moments,
)
from .codesign.evaluation import (
    averaged_contact_steady_state_for_parameters,
    module_electrical_resistance_components,
    module_thermoelectric_parameters,
)
from .codesign.models import (
    ApplicationSpecification,
    ModuleAssemblyAssumptions,
    ModuleGeometry,
    PrototypeDesign,
)


@dataclass(frozen=True)
class MaterialPairDesign:
    """One geometry and assembly with explicit p- and n-material records."""

    design_id: str
    p_material: MaterialSample
    n_material: MaterialSample
    geometry: ModuleGeometry
    symmetric_thermal_contact_resistance: float
    cold_exchanger_conductance: float
    hot_exchanger_conductance: float

    def __post_init__(self) -> None:
        if not self.design_id:
            raise ValueError("design ID must be nonempty")
        if self.p_material.carrier_type != "p":
            raise ValueError("p material must be p-type")
        if self.n_material.carrier_type != "n":
            raise ValueError("n material must be n-type")
        for name, value in (
            ("thermal contact resistance", self.symmetric_thermal_contact_resistance),
            ("cold exchanger conductance", self.cold_exchanger_conductance),
            ("hot exchanger conductance", self.hot_exchanger_conductance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_prototype(
        cls,
        prototype: PrototypeDesign,
        *,
        n_material: Optional[MaterialSample] = None,
        suffix: str = "explicit",
    ) -> "MaterialPairDesign":
        return cls(
            f"{prototype.design_id}-{suffix}",
            prototype.p_material,
            prototype.n_material if n_material is None else n_material,
            prototype.geometry,
            prototype.symmetric_contact_resistance,
            prototype.cold_exchanger_conductance,
            prototype.hot_exchanger_conductance,
        )


class MaterialPairOperatingPoint(NamedTuple):
    design: MaterialPairDesign
    application: ApplicationSpecification
    mean_current: float
    peak_current: float
    cold_face_temperature: float
    hot_face_temperature: float
    cold_exchanger_temperature: float
    hot_exchanger_temperature: float
    delivered_cooling_rate: float
    delivered_heating_rate: float
    module_electrical_power: float
    supply_electrical_power: float
    wall_cooling_cop: Optional[float]
    wall_heating_cop: Optional[float]
    device_zt_300k: float
    bulk_leg_electrical_resistance: float
    electrical_contact_resistance: float
    electrical_contact_fraction: float
    peak_voltage: float
    peak_current_density: float
    current_density_utilization: float
    current_density_constraint_binding: bool
    prototype_cost_index: float
    feasible: bool
    infeasibility_reasons: Tuple[str, ...]
    utility: float


def _zero_safe_smoothed_current(
    mean_current: float,
    ripple_peak_to_peak_fraction: float,
) -> CurrentMoments:
    if mean_current == 0.0:
        return CurrentMoments("smoothed_pwm", 0.0, 0.0, 0.0, 0.0, None, 1.0)
    return smoothed_pwm_current_moments(
        mean_current,
        ripple_peak_to_peak_fraction,
    )


def material_pair_cost_index(design: MaterialPairDesign) -> float:
    """Match the frozen campaign's relative build-burden calculation."""

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
    current_density_ok: bool,
) -> tuple[bool, Tuple[str, ...], float]:
    cop_value = wall_cop if wall_cop is not None else 0.0
    reasons = []
    violations = []
    checks = (
        (
            "cooling_target",
            max(0.0, application.minimum_cooling_rate - cooling_rate)
            / max(application.minimum_cooling_rate, 1.0),
        ),
        (
            "cop_target",
            max(0.0, application.minimum_wall_cop - cop_value)
            / max(application.minimum_wall_cop, 1.0),
        ),
        (
            "supply_power_limit",
            max(0.0, supply_power - application.maximum_supply_power)
            / application.maximum_supply_power,
        ),
        (
            "voltage_limit",
            max(0.0, peak_voltage - maximum_peak_voltage)
            / maximum_peak_voltage,
        ),
    )
    for reason, violation in checks:
        violations.append(violation)
        if violation > 0.0:
            reasons.append(reason)
    if not current_density_ok:
        reasons.append("current_density_limit")
    if reasons:
        density_violation = 1.0 if not current_density_ok else 0.0
        return False, tuple(reasons), -1.0 - sum(violations) - density_violation
    if application.objective == "efficiency":
        utility = cop_value / math.sqrt(cost_index)
    elif application.objective == "balanced":
        utility = cooling_rate * cop_value / cost_index
    else:
        utility = heat_flux * math.sqrt(cop_value) / math.sqrt(cost_index)
    return True, (), utility


def evaluate_material_pair_current(
    design: MaterialPairDesign,
    application: ApplicationSpecification,
    mean_current: float,
    *,
    assembly: ModuleAssemblyAssumptions = ModuleAssemblyAssumptions(),
) -> MaterialPairOperatingPoint:
    """Evaluate one explicit material pair at one nonnegative mean current."""

    if not math.isfinite(mean_current) or mean_current < 0.0:
        raise ValueError("mean current must be finite and nonnegative")
    current = _zero_safe_smoothed_current(
        mean_current,
        assembly.pwm_ripple_peak_to_peak_fraction,
    )
    resistance = module_electrical_resistance_components(
        design.p_material,
        design.n_material,
        design.geometry,
        assembly=assembly,
    )
    parameters = module_thermoelectric_parameters(
        design.p_material,
        design.n_material,
        design.geometry,
        assembly=assembly,
    )
    thermal = FourNodeContactThermalParameters(
        1.0,
        1.0,
        1.0,
        1.0,
        design.symmetric_thermal_contact_resistance,
        design.symmetric_thermal_contact_resistance,
        design.cold_exchanger_conductance,
        design.hot_exchanger_conductance,
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
    rates = averaged_thermoelectric_rates(
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
    if not math.isclose(delivered_cooling, rates.cold_heat, abs_tol=1e-8):
        raise RuntimeError("cold-side steady energy balance did not close")
    if not math.isclose(delivered_heating, rates.hot_heat, abs_tol=1e-8):
        raise RuntimeError("hot-side steady energy balance did not close")
    supply_power = (
        rates.module_electrical_power / assembly.converter_efficiency
        + assembly.fixed_converter_loss
    )
    wall_cooling_cop = (
        delivered_cooling / supply_power
        if delivered_cooling > 0.0 and supply_power > 0.0
        else None
    )
    wall_heating_cop = (
        delivered_heating / supply_power
        if delivered_heating > 0.0 and supply_power > 0.0
        else None
    )
    device_zt = (
        parameters.seebeck_coefficient**2
        * 300.0
        / (parameters.electrical_resistance * parameters.thermal_conductance)
    )
    peak_voltage = (
        parameters.seebeck_coefficient * (state.hot_face - state.cold_face)
        + current.peak_current * parameters.electrical_resistance
    )
    peak_current_density = current.peak_current / design.geometry.leg_area
    density_utilization = peak_current_density / assembly.maximum_current_density
    density_ok = density_utilization <= 1.0 + 1e-12
    density_binding = density_utilization >= 0.995
    cost_index = material_pair_cost_index(design)
    heat_flux = delivered_cooling / (
        design.geometry.estimated_footprint_area * 1.0e4
    )
    feasible, reasons, utility = _application_utility(
        application,
        cooling_rate=delivered_cooling,
        wall_cop=wall_cooling_cop,
        heat_flux=heat_flux,
        cost_index=cost_index,
        supply_power=supply_power,
        peak_voltage=peak_voltage,
        maximum_peak_voltage=assembly.maximum_peak_voltage,
        current_density_ok=density_ok,
    )
    return MaterialPairOperatingPoint(
        design,
        application,
        mean_current,
        current.peak_current,
        state.cold_face,
        state.hot_face,
        state.cold_exchanger,
        state.hot_exchanger,
        delivered_cooling,
        delivered_heating,
        rates.module_electrical_power,
        supply_power,
        wall_cooling_cop,
        wall_heating_cop,
        device_zt,
        resistance.bulk_leg_resistance,
        resistance.electrical_contact_resistance,
        resistance.contact_fraction,
        peak_voltage,
        peak_current_density,
        density_utilization,
        density_binding,
        cost_index,
        feasible,
        reasons,
        utility,
    )


def optimize_material_pair_current(
    design: MaterialPairDesign,
    application: ApplicationSpecification,
    *,
    grid_size: int = 28,
    assembly: ModuleAssemblyAssumptions = ModuleAssemblyAssumptions(),
) -> MaterialPairOperatingPoint:
    """Match the frozen campaign's current grid for an explicit material pair."""

    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("current grid size must be an integer at least two")
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
    return max(
        (
            evaluate_material_pair_current(
                design,
                application,
                current,
                assembly=assembly,
            )
            for current in currents
        ),
        key=lambda point: point.utility,
    )


def match_required_cooling_current(
    design: MaterialPairDesign,
    application: ApplicationSpecification,
    *,
    assembly: ModuleAssemblyAssumptions,
    bracket_subdivisions: int = 24,
    current_tolerance: float = 1e-7,
) -> Optional[MaterialPairOperatingPoint]:
    """Return the first rising-branch point that meets required cooling."""

    if bracket_subdivisions < 2:
        raise ValueError("bracket subdivisions must be at least two")
    if not math.isfinite(current_tolerance) or current_tolerance <= 0.0:
        raise ValueError("current tolerance must be finite and positive")
    maximum_mean_current = (
        assembly.maximum_current_density
        * design.geometry.leg_area
        / (1.0 + 0.5 * assembly.pwm_ripple_peak_to_peak_fraction)
    )
    cache = {}

    def point_at(current: float) -> Optional[MaterialPairOperatingPoint]:
        if current not in cache:
            try:
                cache[current] = evaluate_material_pair_current(
                    design,
                    application,
                    current,
                    assembly=assembly,
                )
            except (ArithmeticError, ValueError):
                # High-current steady states can leave the positive-kelvin
                # domain. They are unusable operating points, not evidence
                # that no lower rising-branch crossing exists.
                cache[current] = None
        return cache[current]

    def cooling_at(current: float) -> float:
        point = point_at(current)
        return point.delivered_cooling_rate if point is not None else -1.0e300

    target = application.minimum_cooling_rate
    previous_current = 0.0
    previous_cooling = cooling_at(previous_current)
    bracket = None
    if previous_cooling >= target:
        bracket = (0.0, 0.0)
    else:
        # Quadratic spacing keeps the first rising crossing resolved when a
        # sensitivity case raises the current ceiling. A linear grid can step
        # entirely over a narrow low-current feasible interval.
        for index in range(1, bracket_subdivisions + 1):
            candidate = maximum_mean_current * (index / bracket_subdivisions) ** 2
            candidate_cooling = cooling_at(candidate)
            if previous_cooling < target <= candidate_cooling:
                bracket = (previous_current, candidate)
                break
            previous_current = candidate
            previous_cooling = candidate_cooling
    if bracket is None:
        return None
    lower, upper = bracket
    if lower == upper:
        return point_at(upper)
    while upper - lower > current_tolerance:
        candidate = 0.5 * (lower + upper)
        if cooling_at(candidate) < application.minimum_cooling_rate:
            lower = candidate
        else:
            upper = candidate
    # ``upper`` is the maintained at-or-above-target side of the bracket.
    # Returning the midpoint can fall a few floating-point ulps below a hard
    # application requirement and be misclassified as infeasible.
    return point_at(upper)


def cooling_target_reachable_without_current_density_limit(
    design: MaterialPairDesign,
    application: ApplicationSpecification,
    *,
    assembly: ModuleAssemblyAssumptions,
    scan_points: int = 33,
    refinement_iterations: int = 32,
    maximum_expansions: int = 4,
) -> bool:
    """Return whether the cooling target is physically reachable without the cap.

    This diagnostic removes only the selected current-density ceiling.  It does
    not send current to infinity.  Instead it brackets the finite cooling
    maximum using the constant-property Peltier/Joule current scale, expands
    the bracket if the best sampled point is still at its upper edge, and then
    refines the sampled maximum.  Invalid high-current steady states are
    treated as lying outside the usable physical branch.
    """

    if isinstance(scan_points, bool) or not isinstance(scan_points, int) or scan_points < 5:
        raise ValueError("scan points must be an integer at least five")
    if (
        isinstance(refinement_iterations, bool)
        or not isinstance(refinement_iterations, int)
        or refinement_iterations < 1
    ):
        raise ValueError("refinement iterations must be a positive integer")
    if (
        isinstance(maximum_expansions, bool)
        or not isinstance(maximum_expansions, int)
        or maximum_expansions < 0
    ):
        raise ValueError("maximum expansions must be a nonnegative integer")

    parameters = module_thermoelectric_parameters(
        design.p_material,
        design.n_material,
        design.geometry,
        assembly=assembly,
    )
    ripple = assembly.pwm_ripple_peak_to_peak_fraction
    ripple_mean_square_multiplier = 1.0 + ripple**2 / 12.0
    cold_reservoir_temperature = 300.0 - 0.5 * application.external_temperature_lift
    peltier_joule_scale = (
        abs(parameters.seebeck_coefficient)
        * max(cold_reservoir_temperature, 1.0)
        / (parameters.electrical_resistance * ripple_mean_square_multiplier)
    )
    capped_mean_current = (
        assembly.maximum_current_density
        * design.geometry.leg_area
        / (1.0 + 0.5 * ripple)
    )
    upper_current = max(2.0 * peltier_joule_scale, 2.0 * capped_mean_current, 0.1)
    target = application.minimum_cooling_rate
    cache = {}

    def cooling_at(current: float) -> float:
        if current not in cache:
            try:
                point = evaluate_material_pair_current(
                    design,
                    application,
                    current,
                    assembly=assembly,
                )
            except (ArithmeticError, ValueError):
                cache[current] = -math.inf
            else:
                cache[current] = point.delivered_cooling_rate
        return cache[current]

    sampled_currents = ()
    sampled_cooling = ()
    best_index = 0
    for expansion in range(maximum_expansions + 1):
        sampled_currents = tuple(
            upper_current * (index / (scan_points - 1)) ** 2
            for index in range(scan_points)
        )
        sampled_cooling = tuple(cooling_at(current) for current in sampled_currents)
        if max(sampled_cooling) >= target:
            return True
        best_index = max(range(scan_points), key=sampled_cooling.__getitem__)
        if best_index < scan_points - 1:
            break
        if expansion == maximum_expansions:
            raise RuntimeError(
                "current-density-unconstrained cooling search did not bracket a maximum"
            )
        upper_current *= 2.0

    if best_index == 0:
        return sampled_cooling[0] >= target

    lower = sampled_currents[best_index - 1]
    upper = sampled_currents[min(best_index + 1, scan_points - 1)]
    inverse_golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left_probe = upper - inverse_golden_ratio * (upper - lower)
    right_probe = lower + inverse_golden_ratio * (upper - lower)
    left_cooling = cooling_at(left_probe)
    right_cooling = cooling_at(right_probe)
    if max(left_cooling, right_cooling) >= target:
        return True

    for _ in range(refinement_iterations):
        if left_cooling < right_cooling:
            lower = left_probe
            left_probe = right_probe
            left_cooling = right_cooling
            right_probe = lower + inverse_golden_ratio * (upper - lower)
            right_cooling = cooling_at(right_probe)
        else:
            upper = right_probe
            right_probe = left_probe
            right_cooling = left_cooling
            left_probe = upper - inverse_golden_ratio * (upper - lower)
            left_cooling = cooling_at(left_probe)
        if max(left_cooling, right_cooling) >= target:
            return True
    return False


__all__ = [
    "MaterialPairDesign",
    "MaterialPairOperatingPoint",
    "cooling_target_reachable_without_current_density_limit",
    "evaluate_material_pair_current",
    "match_required_cooling_current",
    "material_pair_cost_index",
    "optimize_material_pair_current",
]
