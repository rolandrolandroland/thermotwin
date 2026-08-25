"""Matched-pair Ag2Se substitution across the frozen co-design pool."""

from dataclasses import dataclass, replace
import math
from typing import NamedTuple, Optional, Tuple

from ..numerics.statistics import interpolated_quantile
from .codesign.evaluation import optimize_design_current
from .codesign.models import (
    APPLICATION_SPECIFICATIONS,
    ApplicationSpecification,
    DesignOperatingPoint,
    ModuleAssemblyAssumptions,
)
from .codesign.sampling import generate_space_filling_designs
from .literature_materials import (
    AG2SE_2026_OPTIMIZED,
    PUBLISHED_AG2SE_UNICOUPLE,
)
from .material_pair import (
    MaterialPairDesign,
    MaterialPairOperatingPoint,
    optimize_material_pair_current,
)


@dataclass(frozen=True)
class Ag2SeSubstitutionConfig:
    """Frozen baseline pool and two declared electrical-interface cases."""

    initial_design_count: int = 24
    candidate_design_count: int = 180
    seed: int = 20260821
    current_grid_size: int = 28
    specific_contact_resistivities: Tuple[float, ...] = (
        2.0e-10,
        PUBLISHED_AG2SE_UNICOUPLE.inferred_specific_contact_resistivity,
    )
    applications: Tuple[ApplicationSpecification, ...] = APPLICATION_SPECIFICATIONS
    base_assembly: ModuleAssemblyAssumptions = ModuleAssemblyAssumptions()

    def __post_init__(self) -> None:
        for name, value in (
            ("initial design count", self.initial_design_count),
            ("candidate design count", self.candidate_design_count),
            ("current grid size", self.current_grid_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.current_grid_size < 2:
            raise ValueError("current grid size must be at least two")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
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
        if not self.applications:
            raise ValueError("at least one application is required")


class MatchedSubstitutionComparison(NamedTuple):
    design_id: str
    p_sample_id: int
    original_n_sample_id: int
    application_name: str
    specific_contact_resistivity: float
    original: DesignOperatingPoint
    ag2se: MaterialPairOperatingPoint
    cooling_change: float
    cop_change: Optional[float]
    current_change: float
    utility_change: float
    feasibility_change: str


class SubstitutionSummary(NamedTuple):
    application_name: str
    application_label: str
    specific_contact_resistivity: float
    comparison_count: int
    utility_improved_fraction: float
    cooling_improved_fraction: float
    cop_improved_fraction: float
    feasibility_gained_count: int
    feasibility_lost_count: int
    median_cooling_change: float
    median_cop_change: Optional[float]
    median_current_change: float
    best_original_design_id: Optional[str]
    best_original_utility: Optional[float]
    best_ag2se_design_id: Optional[str]
    best_ag2se_utility: Optional[float]
    ag2se_produces_new_best: bool


class Ag2SeSubstitutionResult(NamedTuple):
    config: Ag2SeSubstitutionConfig
    comparisons: Tuple[MatchedSubstitutionComparison, ...]
    summaries: Tuple[SubstitutionSummary, ...]


def _feasibility_change(
    original: DesignOperatingPoint,
    ag2se: MaterialPairOperatingPoint,
) -> str:
    if not original.feasible and ag2se.feasible:
        return "gained"
    if original.feasible and not ag2se.feasible:
        return "lost"
    return "unchanged"


def _summarize(
    application: ApplicationSpecification,
    specific_contact_resistivity: float,
    comparisons: Tuple[MatchedSubstitutionComparison, ...],
) -> SubstitutionSummary:
    cop_changes = tuple(
        item.cop_change for item in comparisons if item.cop_change is not None
    )
    feasible_original = tuple(item.original for item in comparisons if item.original.feasible)
    feasible_ag2se = tuple(item.ag2se for item in comparisons if item.ag2se.feasible)
    best_original = (
        max(feasible_original, key=lambda point: point.utility)
        if feasible_original
        else None
    )
    best_ag2se = (
        max(feasible_ag2se, key=lambda point: point.utility)
        if feasible_ag2se
        else None
    )
    return SubstitutionSummary(
        application.name,
        application.label,
        specific_contact_resistivity,
        len(comparisons),
        sum(item.utility_change > 1e-12 for item in comparisons) / len(comparisons),
        sum(item.cooling_change > 1e-12 for item in comparisons) / len(comparisons),
        (
            sum(value > 1e-12 for value in cop_changes) / len(cop_changes)
            if cop_changes
            else 0.0
        ),
        sum(item.feasibility_change == "gained" for item in comparisons),
        sum(item.feasibility_change == "lost" for item in comparisons),
        interpolated_quantile(tuple(item.cooling_change for item in comparisons), 0.5),
        interpolated_quantile(cop_changes, 0.5) if cop_changes else None,
        interpolated_quantile(tuple(item.current_change for item in comparisons), 0.5),
        best_original.design.design_id if best_original is not None else None,
        best_original.utility if best_original is not None else None,
        best_ag2se.design.design_id if best_ag2se is not None else None,
        best_ag2se.utility if best_ag2se is not None else None,
        (
            best_ag2se is not None
            and (
                best_original is None
                or best_ag2se.utility > best_original.utility + 1e-12
            )
        ),
    )


def run_ag2se_substitution_study(
    config: Ag2SeSubstitutionConfig = Ag2SeSubstitutionConfig(),
) -> Ag2SeSubstitutionResult:
    """Replace only the n material in all 204 frozen-pool geometries."""

    initial = generate_space_filling_designs(
        config.initial_design_count,
        seed=config.seed,
        prefix="initial",
    )
    candidates = generate_space_filling_designs(
        config.candidate_design_count,
        seed=config.seed + 1,
        prefix="candidate",
    )
    designs = initial + candidates
    comparisons = []
    summaries = []
    for specific_contact_resistivity in config.specific_contact_resistivities:
        assembly = replace(
            config.base_assembly,
            specific_electrical_contact_resistivity=specific_contact_resistivity,
        )
        for application in config.applications:
            group = []
            for design in designs:
                original = optimize_design_current(
                    design,
                    application,
                    grid_size=config.current_grid_size,
                    assembly=assembly,
                )
                replacement_design = MaterialPairDesign.from_prototype(
                    design,
                    n_material=AG2SE_2026_OPTIMIZED.material,
                    suffix="ag2se",
                )
                ag2se = optimize_material_pair_current(
                    replacement_design,
                    application,
                    grid_size=config.current_grid_size,
                    assembly=assembly,
                )
                cop_change = (
                    ag2se.wall_cooling_cop - original.wall_cooling_cop
                    if ag2se.wall_cooling_cop is not None
                    and original.wall_cooling_cop is not None
                    else None
                )
                comparison = MatchedSubstitutionComparison(
                    design.design_id,
                    design.p_material.sample_id,
                    design.n_material.sample_id,
                    application.name,
                    specific_contact_resistivity,
                    original,
                    ag2se,
                    ag2se.delivered_cooling_rate - original.delivered_cooling_rate,
                    cop_change,
                    ag2se.mean_current - original.mean_current,
                    ag2se.utility - original.utility,
                    _feasibility_change(original, ag2se),
                )
                group.append(comparison)
                comparisons.append(comparison)
            summaries.append(
                _summarize(
                    application,
                    specific_contact_resistivity,
                    tuple(group),
                )
            )
    return Ag2SeSubstitutionResult(config, tuple(comparisons), tuple(summaries))


def format_ag2se_substitution_report(result: Ag2SeSubstitutionResult) -> str:
    lines = [
        "ThermoTwin matched Ag2Se substitution study",
        (
            f"matched designs: {result.config.initial_design_count + result.config.candidate_design_count}; "
            "continuous variables and p material held fixed"
        ),
        f"Ag2Se source: {AG2SE_2026_OPTIMIZED.source_doi}",
    ]
    for summary in result.summaries:
        lines.extend(
            (
                "",
                (
                    f"{summary.application_label}, rho_c="
                    f"{summary.specific_contact_resistivity:.6e} ohm m^2:"
                ),
                (
                    "  improved utility/cooling/COP: "
                    f"{100 * summary.utility_improved_fraction:.1f}% / "
                    f"{100 * summary.cooling_improved_fraction:.1f}% / "
                    f"{100 * summary.cop_improved_fraction:.1f}%"
                ),
                (
                    "  feasibility gained/lost: "
                    f"{summary.feasibility_gained_count}/"
                    f"{summary.feasibility_lost_count}"
                ),
                (
                    "  median delta Qc/COP/I: "
                    f"{summary.median_cooling_change:.4f} W / "
                    f"{(summary.median_cop_change or 0.0):.4f} / "
                    f"{summary.median_current_change:.4f} A"
                ),
                (
                    "  best original/Ag2Se utility: "
                    f"{summary.best_original_utility} / {summary.best_ag2se_utility}; "
                    f"new best={'yes' if summary.ag2se_produces_new_best else 'no'}"
                ),
            )
        )
    return "\n".join(lines)


__all__ = [
    "Ag2SeSubstitutionConfig",
    "Ag2SeSubstitutionResult",
    "MatchedSubstitutionComparison",
    "SubstitutionSummary",
    "format_ag2se_substitution_report",
    "run_ag2se_substitution_study",
]
