"""Electrical-contact process window across geometry and application needs."""

from dataclasses import dataclass, replace
import math
from typing import NamedTuple, Optional, Sequence, Tuple

from .codesign.evaluation import module_electrical_resistance_components
from .codesign.models import (
    APPLICATION_SPECIFICATIONS,
    ApplicationSpecification,
    ModuleAssemblyAssumptions,
    ModuleGeometry,
)
from .literature_materials import (
    AG2SE_2026_OPTIMIZED,
    PUBLISHED_AG2SE_UNICOUPLE,
)
from .material_pair import (
    MaterialPairDesign,
    cooling_target_reachable_without_current_density_limit,
    evaluate_material_pair_current,
    match_required_cooling_current,
)
from .materials import N_TYPE_SAMPLES, P_TYPE_SAMPLES, MaterialSample, material_sample


def logarithmic_grid(
    minimum: float,
    maximum: float,
    count: int,
) -> Tuple[float, ...]:
    """Return an inclusive dependency-free logarithmic grid."""

    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0.0
        or maximum <= minimum
    ):
        raise ValueError("logarithmic grid bounds must be positive and ordered")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("logarithmic grid count must be an integer at least two")
    log_minimum = math.log(minimum)
    log_span = math.log(maximum) - log_minimum
    return tuple(
        math.exp(log_minimum + log_span * index / (count - 1))
        for index in range(count)
    )


class ProcessMaterialPair(NamedTuple):
    key: str
    label: str
    p_material: MaterialSample
    n_material: MaterialSample


def default_process_material_pairs() -> Tuple[ProcessMaterialPair, ...]:
    """Return one legacy reference and the Ag2Se/p-type pairing envelope."""

    reference = ProcessMaterialPair(
        "reference_9107_10562",
        "reference 9107 + 10562",
        material_sample(9107),
        material_sample(10562),
    )
    ag2se = AG2SE_2026_OPTIMIZED.material
    extensions = tuple(
        ProcessMaterialPair(
            f"ag2se_p_{sample.sample_id}",
            f"p {sample.sample_id} + optimized Ag2Se",
            sample,
            ag2se,
        )
        for sample in P_TYPE_SAMPLES
    )
    return (reference,) + extensions


@dataclass(frozen=True)
class ContactProcessWindowConfig:
    """Frozen grids and system assumptions for the process-window study."""

    leg_lengths: Tuple[float, ...] = logarithmic_grid(0.05e-3, 2.5e-3, 61)
    specific_contact_resistivities: Tuple[float, ...] = (0.0,) + logarithmic_grid(
        1.0e-11,
        5.0e-8,
        61,
    )
    current_density_limits: Tuple[float, ...] = (1.0e6, 3.0e6)
    material_pairs: Tuple[ProcessMaterialPair, ...] = default_process_material_pairs()
    applications: Tuple[ApplicationSpecification, ...] = APPLICATION_SPECIFICATIONS
    couple_count: int = 120
    leg_area: float = 1.6e-6
    symmetric_thermal_contact_resistance: float = 0.25
    cold_exchanger_conductance: float = 2.5
    hot_exchanger_conductance: float = 5.0
    bracket_subdivisions: int = 24
    current_tolerance: float = 1.0e-6
    base_assembly: ModuleAssemblyAssumptions = ModuleAssemblyAssumptions()

    def __post_init__(self) -> None:
        if not self.leg_lengths or any(
            not math.isfinite(value) or value <= 0.0 for value in self.leg_lengths
        ):
            raise ValueError("leg lengths must be finite and positive")
        if tuple(sorted(set(self.leg_lengths))) != self.leg_lengths:
            raise ValueError("leg lengths must be unique and ordered")
        if not self.specific_contact_resistivities or any(
            not math.isfinite(value) or value < 0.0
            for value in self.specific_contact_resistivities
        ):
            raise ValueError("contact resistivities must be finite and nonnegative")
        if (
            tuple(sorted(set(self.specific_contact_resistivities)))
            != self.specific_contact_resistivities
        ):
            raise ValueError("contact resistivities must be unique and ordered")
        if not self.current_density_limits or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.current_density_limits
        ):
            raise ValueError("current-density limits must be finite and positive")
        if tuple(sorted(set(self.current_density_limits))) != self.current_density_limits:
            raise ValueError("current-density limits must be unique and ordered")
        if not self.material_pairs or len({pair.key for pair in self.material_pairs}) != len(
            self.material_pairs
        ):
            raise ValueError("material-pair keys must be nonempty and unique")
        if not self.applications or len({item.name for item in self.applications}) != len(
            self.applications
        ):
            raise ValueError("applications must be nonempty and unique")
        if self.couple_count <= 0:
            raise ValueError("couple count must be positive")
        for name, value in (
            ("leg area", self.leg_area),
            ("thermal contact resistance", self.symmetric_thermal_contact_resistance),
            ("cold exchanger conductance", self.cold_exchanger_conductance),
            ("hot exchanger conductance", self.hot_exchanger_conductance),
            ("current tolerance", self.current_tolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.bracket_subdivisions < 2:
            raise ValueError("bracket subdivisions must be at least two")


class ProcessWindowPoint(NamedTuple):
    pair_key: str
    pair_label: str
    application_name: str
    application_label: str
    leg_length: float
    specific_contact_resistivity: float
    maximum_current_density: float
    matched_required_cooling: bool
    feasible: bool
    infeasibility_reasons: Tuple[str, ...]
    mean_current: float
    delivered_cooling_rate: float
    delivered_heating_rate: float
    wall_cooling_cop: Optional[float]
    wall_heating_cop: Optional[float]
    supply_electrical_power: float
    device_zt_300k: float
    electrical_contact_fraction: float
    current_density_utilization: float
    peak_voltage: float


class PublishedElectricalSweepPoint(NamedTuple):
    specific_contact_resistivity: float
    electrical_contact_fraction: float
    normalized_zt_retention: float


class ContactProcessWindowResult(NamedTuple):
    config: ContactProcessWindowConfig
    points: Tuple[ProcessWindowPoint, ...]
    published_electrical_sweep: Tuple[PublishedElectricalSweepPoint, ...]


def _process_design(
    config: ContactProcessWindowConfig,
    pair: ProcessMaterialPair,
    leg_length: float,
) -> MaterialPairDesign:
    return MaterialPairDesign(
        f"{pair.key}-L-{leg_length:.12g}",
        pair.p_material,
        pair.n_material,
        ModuleGeometry(config.couple_count, leg_length, config.leg_area),
        config.symmetric_thermal_contact_resistance,
        config.cold_exchanger_conductance,
        config.hot_exchanger_conductance,
    )


def _evaluate_process_point(
    config: ContactProcessWindowConfig,
    pair: ProcessMaterialPair,
    application: ApplicationSpecification,
    leg_length: float,
    specific_contact_resistivity: float,
    maximum_current_density: float,
    unconstrained_reachability_cache,
) -> ProcessWindowPoint:
    design = _process_design(config, pair, leg_length)
    assembly = replace(
        config.base_assembly,
        specific_electrical_contact_resistivity=specific_contact_resistivity,
        maximum_current_density=maximum_current_density,
    )
    matched = match_required_cooling_current(
        design,
        application,
        assembly=assembly,
        bracket_subdivisions=config.bracket_subdivisions,
        current_tolerance=config.current_tolerance,
    )
    if matched is None:
        operating = evaluate_material_pair_current(
            design,
            application,
            0.0,
            assembly=assembly,
        )
        matched_required = False
        feasible = False
        reachability_key = (
            pair.key,
            application.name,
            leg_length,
            specific_contact_resistivity,
        )
        if reachability_key not in unconstrained_reachability_cache:
            unconstrained_reachability_cache[reachability_key] = (
                cooling_target_reachable_without_current_density_limit(
                    design,
                    application,
                    assembly=assembly,
                )
            )
        reasons = (
            "cooling_target_current_density_limited"
            if unconstrained_reachability_cache[reachability_key]
            else "cooling_target_physics_limited",
        )
    else:
        operating = matched
        matched_required = True
        feasible = operating.feasible
        reasons = operating.infeasibility_reasons
    return ProcessWindowPoint(
        pair.key,
        pair.label,
        application.name,
        application.label,
        leg_length,
        specific_contact_resistivity,
        maximum_current_density,
        matched_required,
        feasible,
        reasons,
        operating.mean_current,
        operating.delivered_cooling_rate,
        operating.delivered_heating_rate,
        operating.wall_cooling_cop,
        operating.wall_heating_cop,
        operating.supply_electrical_power,
        operating.device_zt_300k,
        operating.electrical_contact_fraction,
        operating.current_density_utilization,
        operating.peak_voltage,
    )


def run_contact_process_window(
    config: ContactProcessWindowConfig = ContactProcessWindowConfig(),
) -> ContactProcessWindowResult:
    """Run the cost-free process-window study on the declared tensor grid."""

    points = []
    unconstrained_reachability_cache = {}
    for pair in config.material_pairs:
        for application in config.applications:
            for leg_length in config.leg_lengths:
                for specific_contact_resistivity in (
                    config.specific_contact_resistivities
                ):
                    reachability_key = (
                        pair.key,
                        application.name,
                        leg_length,
                        specific_contact_resistivity,
                    )
                    points_by_limit = {}
                    for maximum_current_density in reversed(
                        config.current_density_limits
                    ):
                        point = _evaluate_process_point(
                            config,
                            pair,
                            application,
                            leg_length,
                            specific_contact_resistivity,
                            maximum_current_density,
                            unconstrained_reachability_cache,
                        )
                        points_by_limit[maximum_current_density] = point
                        if point.matched_required_cooling:
                            unconstrained_reachability_cache[reachability_key] = True
                    points.extend(
                        points_by_limit[limit]
                        for limit in config.current_density_limits
                    )
    paper = PUBLISHED_AG2SE_UNICOUPLE
    paper_sweep = tuple(
        PublishedElectricalSweepPoint(
            value,
            paper.modeled_contact_share(value),
            1.0 - paper.modeled_contact_share(value),
        )
        for value in config.specific_contact_resistivities
    )
    return ContactProcessWindowResult(config, tuple(points), paper_sweep)


def select_process_points(
    result: ContactProcessWindowResult,
    *,
    pair_key: Optional[str] = None,
    application_name: Optional[str] = None,
    maximum_current_density: Optional[float] = None,
) -> Tuple[ProcessWindowPoint, ...]:
    """Filter result points without recomputing the experiment."""

    return tuple(
        point
        for point in result.points
        if (pair_key is None or point.pair_key == pair_key)
        and (application_name is None or point.application_name == application_name)
        and (
            maximum_current_density is None
            or point.maximum_current_density == maximum_current_density
        )
    )


def maximum_feasible_contact_resistivity(
    points: Sequence[ProcessWindowPoint],
) -> Optional[float]:
    feasible = tuple(point.specific_contact_resistivity for point in points if point.feasible)
    return max(feasible) if feasible else None


def format_contact_process_window_report(
    result: ContactProcessWindowResult,
) -> str:
    """Return a compact text report with paper and application landmarks."""

    paper = PUBLISHED_AG2SE_UNICOUPLE
    lines = [
        "ThermoTwin electrical-contact process window",
        (
            "published unicouple: 50% crossover="
            f"{paper.half_contact_share_resistivity:.6e} ohm m^2; "
            "inferred Rc*A="
            f"{paper.inferred_specific_contact_resistivity:.6e} ohm m^2"
        ),
        (
            "reported/model contact share at inferred point: "
            f"{100.0 * paper.reported_contact_share:.2f}% / "
            f"{100.0 * paper.modeled_contact_share(paper.inferred_specific_contact_resistivity):.2f}%"
        ),
        "cost index: excluded",
        "current-density cases: 1.0 A/mm^2 existing campaign constraint; "
        "3.0 A/mm^2 exploratory sensitivity",
    ]
    unmatched = tuple(
        point for point in result.points if not point.matched_required_cooling
    )
    if unmatched:
        cap_limited = sum(
            point.infeasibility_reasons
            == ("cooling_target_current_density_limited",)
            for point in unmatched
        )
        physics_limited = sum(
            point.infeasibility_reasons == ("cooling_target_physics_limited",)
            for point in unmatched
        )
        lines.append(
            "all-grid cooling-target misses: "
            f"current-density cap={cap_limited}/{len(unmatched)} "
            f"({100.0 * cap_limited / len(unmatched):.1f}%); "
            f"physical maximum={physics_limited}/{len(unmatched)} "
            f"({100.0 * physics_limited / len(unmatched):.1f}%)"
        )
    target_length = min(
        result.config.leg_lengths,
        key=lambda value: abs(value - paper.leg_length),
    )
    for pair_key in ("reference_9107_10562", "ag2se_p_9107"):
        lines.append("")
        lines.append(f"{pair_key} at nearest grid length {target_length * 1e3:.4f} mm:")
        for application in result.config.applications:
            for limit in result.config.current_density_limits:
                selected = tuple(
                    point
                    for point in result.points
                    if point.pair_key == pair_key
                    and point.application_name == application.name
                    and point.maximum_current_density == limit
                    and point.leg_length == target_length
                )
                maximum = maximum_feasible_contact_resistivity(selected)
                maximum_text = "none" if maximum is None else f"{maximum:.6e} ohm m^2"
                lines.append(
                    f"  {application.label}, {limit / 1e6:.1f} A/mm^2: "
                    f"maximum feasible rho_c={maximum_text}"
                )
    ag2se_pair_keys = tuple(
        pair.key
        for pair in result.config.material_pairs
        if pair.key.startswith("ag2se_p_")
    )
    if ag2se_pair_keys:
        lines.append("")
        lines.append(
            "optimized Ag2Se across all declared p-type records at "
            f"{target_length * 1e3:.4f} mm:"
        )
        for application in result.config.applications:
            for limit in result.config.current_density_limits:
                maxima = []
                for pair_key in ag2se_pair_keys:
                    selected = tuple(
                        point
                        for point in result.points
                        if point.pair_key == pair_key
                        and point.application_name == application.name
                        and point.maximum_current_density == limit
                        and point.leg_length == target_length
                    )
                    maximum = maximum_feasible_contact_resistivity(selected)
                    if maximum is not None:
                        maxima.append(maximum)
                range_text = (
                    "none"
                    if not maxima
                    else f"{min(maxima):.6e} to {max(maxima):.6e} ohm m^2"
                )
                lines.append(
                    f"  {application.label}, {limit / 1e6:.1f} A/mm^2: "
                    f"maximum-feasible-rho range={range_text}"
                )
    return "\n".join(lines)


__all__ = [
    "ContactProcessWindowConfig",
    "ContactProcessWindowResult",
    "ProcessMaterialPair",
    "ProcessWindowPoint",
    "PublishedElectricalSweepPoint",
    "default_process_material_pairs",
    "format_contact_process_window_report",
    "logarithmic_grid",
    "maximum_feasible_contact_resistivity",
    "run_contact_process_window",
    "select_process_points",
]
