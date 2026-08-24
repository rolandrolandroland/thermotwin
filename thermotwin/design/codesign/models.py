"""Typed configuration and result records for material/geometry co-design."""

from dataclasses import dataclass
import math
from typing import NamedTuple, Optional, Tuple

from ..materials import MaterialSample, N_TYPE_SAMPLES, P_TYPE_SAMPLES
from ...physics.thermoelectric import ThermoelectricParameters


CURRENT_DENSITY_BINDING_UTILIZATION = 0.995


@dataclass(frozen=True)
class ModuleGeometry:
    """Repeated p/n-leg geometry for one idealized thermoelectric module."""

    couple_count: int
    leg_length: float
    leg_area: float
    packing_fraction: float = 0.65

    def __post_init__(self) -> None:
        if isinstance(self.couple_count, bool) or self.couple_count <= 0:
            raise ValueError("couple count must be a positive integer")
        if int(self.couple_count) != self.couple_count:
            raise ValueError("couple count must be an integer")
        for name, value in (
            ("leg length", self.leg_length),
            ("leg area", self.leg_area),
            ("packing fraction", self.packing_fraction),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.packing_fraction > 1.0:
            raise ValueError("packing fraction cannot exceed one")

    @property
    def active_material_volume(self) -> float:
        """Return total p-plus-n leg volume in cubic metres."""

        return 2.0 * self.couple_count * self.leg_area * self.leg_length

    @property
    def estimated_footprint_area(self) -> float:
        """Return leg-area-based module footprint estimate in square metres."""

        return 2.0 * self.couple_count * self.leg_area / self.packing_fraction


@dataclass(frozen=True)
class ModuleAssemblyAssumptions:
    """Documented non-material contributions used in the virtual campaign."""

    specific_electrical_contact_resistivity: float = 2.0e-10
    parasitic_thermal_conductance: float = 0.04
    pwm_ripple_peak_to_peak_fraction: float = 0.10
    converter_efficiency: float = 0.95
    fixed_converter_loss: float = 0.05
    maximum_current_density: float = 1.0e6
    maximum_peak_voltage: float = 12.0

    def __post_init__(self) -> None:
        positive = (
            ("maximum current density", self.maximum_current_density),
            ("maximum peak voltage", self.maximum_peak_voltage),
        )
        for name, value in positive:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = (
            (
                "specific electrical contact resistivity",
                self.specific_electrical_contact_resistivity,
            ),
            ("parasitic thermal conductance", self.parasitic_thermal_conductance),
            ("PWM ripple fraction", self.pwm_ripple_peak_to_peak_fraction),
            ("fixed converter loss", self.fixed_converter_loss),
        )
        for name, value in nonnegative:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 < self.converter_efficiency <= 1.0:
            raise ValueError("converter efficiency must lie in (0, 1]")


@dataclass(frozen=True)
class PrototypeDesign:
    """One virtual material, geometry, contact, and exchanger design."""

    design_id: str
    p_sample_index: int
    n_sample_index: int
    geometry: ModuleGeometry
    symmetric_contact_resistance: float
    cold_exchanger_conductance: float
    hot_exchanger_conductance: float

    def __post_init__(self) -> None:
        if not self.design_id:
            raise ValueError("design ID must be nonempty")
        if not 0 <= self.p_sample_index < len(P_TYPE_SAMPLES):
            raise ValueError("p-sample index is outside the curated catalog")
        if not 0 <= self.n_sample_index < len(N_TYPE_SAMPLES):
            raise ValueError("n-sample index is outside the curated catalog")
        for name, value in (
            ("contact resistance", self.symmetric_contact_resistance),
            ("cold exchanger conductance", self.cold_exchanger_conductance),
            ("hot exchanger conductance", self.hot_exchanger_conductance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def p_material(self) -> MaterialSample:
        return P_TYPE_SAMPLES[self.p_sample_index]

    @property
    def n_material(self) -> MaterialSample:
        return N_TYPE_SAMPLES[self.n_sample_index]


@dataclass(frozen=True)
class PropertyMultipliers:
    """Dimensionless as-built multipliers used by the robustness experiment."""

    p_seebeck: float = 1.0
    n_seebeck: float = 1.0
    p_electrical_conductivity: float = 1.0
    n_electrical_conductivity: float = 1.0
    p_thermal_conductivity: float = 1.0
    n_thermal_conductivity: float = 1.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0.0 for value in self):
            raise ValueError("all material-property multipliers must be positive")

    def __iter__(self):
        return iter(
            (
                self.p_seebeck,
                self.n_seebeck,
                self.p_electrical_conductivity,
                self.n_electrical_conductivity,
                self.p_thermal_conductivity,
                self.n_thermal_conductivity,
            )
        )


@dataclass(frozen=True)
class ApplicationSpecification:
    """Cooling requirement and scalar decision objective for one use case."""

    name: str
    label: str
    external_temperature_lift: float
    minimum_cooling_rate: float
    minimum_wall_cop: float
    maximum_supply_power: float
    objective: str

    def __post_init__(self) -> None:
        if not self.name or not self.label:
            raise ValueError("application name and label must be nonempty")
        if self.objective not in {"efficiency", "balanced", "capacity"}:
            raise ValueError("unknown application objective")
        for name, value, allow_zero in (
            ("temperature lift", self.external_temperature_lift, True),
            ("minimum cooling rate", self.minimum_cooling_rate, True),
            ("minimum wall COP", self.minimum_wall_cop, True),
            ("maximum supply power", self.maximum_supply_power, False),
        ):
            if (
                not math.isfinite(value)
                or value < 0.0
                or (not allow_zero and value == 0.0)
            ):
                raise ValueError(f"{name} is invalid")


APPLICATION_SPECIFICATIONS: Tuple[ApplicationSpecification, ...] = (
    ApplicationSpecification(
        "low_lift_efficiency",
        "10 K efficiency-first",
        10.0,
        2.5,
        0.75,
        25.0,
        "efficiency",
    ),
    ApplicationSpecification(
        "high_lift_balanced",
        "25 K balanced",
        25.0,
        0.75,
        0.15,
        30.0,
        "balanced",
    ),
    ApplicationSpecification(
        "capacity_first",
        "10 K capacity-first",
        10.0,
        4.0,
        0.60,
        30.0,
        "capacity",
    ),
)


class DesignOperatingPoint(NamedTuple):
    """One design at its selected mean-current operating point."""

    design: PrototypeDesign
    application: ApplicationSpecification
    thermoelectric_parameters: ThermoelectricParameters
    bulk_leg_electrical_resistance: float
    electrical_contact_resistance: float
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
    heat_flux: float
    prototype_cost_index: float
    peak_voltage: float
    peak_current_density: float
    current_density_utilization: float
    current_density_constraint_binding: bool
    feasible: bool
    utility: float


class InitialDesignSummary(NamedTuple):
    application: ApplicationSpecification
    evaluations: Tuple[DesignOperatingPoint, ...]
    feasible_count: int
    best: DesignOperatingPoint


class BayesianOptimizationResult(NamedTuple):
    application: ApplicationSpecification
    initial_evaluations: Tuple[DesignOperatingPoint, ...]
    acquired_evaluations: Tuple[DesignOperatingPoint, ...]
    best_utility_history: Tuple[float, ...]
    random_median_history: Tuple[float, ...]
    random_lower_history: Tuple[float, ...]
    random_upper_history: Tuple[float, ...]
    selected: DesignOperatingPoint
    oracle_best: DesignOperatingPoint


class RobustnessResult(NamedTuple):
    application: ApplicationSpecification
    nominal: DesignOperatingPoint
    trial_count: int
    feasible_fraction: float
    cooling_rate_quantiles: Tuple[float, float, float]
    wall_cop_quantiles: Tuple[float, float, float]
    hot_face_temperature_quantiles: Tuple[float, float, float]


@dataclass(frozen=True)
class CodesignCampaignConfig:
    initial_design_count: int = 24
    candidate_design_count: int = 180
    bayesian_iterations: int = 12
    random_search_repetitions: int = 25
    robustness_trials: int = 300
    seed: int = 20260821
    current_grid_size: int = 28
    assembly: ModuleAssemblyAssumptions = ModuleAssemblyAssumptions()

    def __post_init__(self) -> None:
        for name, value in (
            ("initial design count", self.initial_design_count),
            ("candidate design count", self.candidate_design_count),
            ("Bayesian iterations", self.bayesian_iterations),
            ("random repetitions", self.random_search_repetitions),
            ("robustness trials", self.robustness_trials),
            ("current grid size", self.current_grid_size),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.bayesian_iterations > self.candidate_design_count:
            raise ValueError("Bayesian iterations exceed candidate count")


class CodesignCampaignResult(NamedTuple):
    config: CodesignCampaignConfig
    initial_designs: Tuple[PrototypeDesign, ...]
    candidate_designs: Tuple[PrototypeDesign, ...]
    initial_summaries: Tuple[InitialDesignSummary, ...]
    bayesian_results: Tuple[BayesianOptimizationResult, ...]
    robustness_results: Tuple[RobustnessResult, ...]


class ModuleElectricalResistanceComponents(NamedTuple):
    """Bulk-leg and areal-contact contributions to module resistance."""

    bulk_leg_resistance: float
    electrical_contact_resistance: float
    total_resistance: float
    contact_fraction: float
