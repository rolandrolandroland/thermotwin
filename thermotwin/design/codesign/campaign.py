"""End-to-end orchestration and text reporting for the co-design campaign."""

from .evaluation import optimize_design_current
from .models import (
    APPLICATION_SPECIFICATIONS,
    CodesignCampaignConfig,
    CodesignCampaignResult,
    InitialDesignSummary,
)
from .optimization import run_bayesian_optimization
from .robustness import run_robustness_study
from .sampling import generate_space_filling_designs


def run_codesign_campaign(
    config: CodesignCampaignConfig = CodesignCampaignConfig(),
) -> CodesignCampaignResult:
    """Run all three reproducible co-design experiments."""

    initial_designs = generate_space_filling_designs(
        config.initial_design_count,
        seed=config.seed,
        prefix="initial",
    )
    candidate_designs = generate_space_filling_designs(
        config.candidate_design_count,
        seed=config.seed + 1,
        prefix="candidate",
    )
    initial_summaries = []
    bayesian_results = []
    robustness_results = []
    for application_index, application in enumerate(APPLICATION_SPECIFICATIONS):
        initial_evaluations = tuple(
            optimize_design_current(
                design,
                application,
                grid_size=config.current_grid_size,
                assembly=config.assembly,
            )
            for design in initial_designs
        )
        initial_summaries.append(
            InitialDesignSummary(
                application,
                initial_evaluations,
                sum(point.feasible for point in initial_evaluations),
                max(initial_evaluations, key=lambda point: point.utility),
            )
        )
        bayesian = run_bayesian_optimization(
            application,
            initial_designs,
            candidate_designs,
            iterations=config.bayesian_iterations,
            random_repetitions=config.random_search_repetitions,
            seed=config.seed + 100 * application_index,
            current_grid_size=config.current_grid_size,
            assembly=config.assembly,
        )
        bayesian_results.append(bayesian)
        robustness_results.append(
            run_robustness_study(
                bayesian.selected,
                trials=config.robustness_trials,
                seed=config.seed + 10000 * (application_index + 1),
                assembly=config.assembly,
            )
        )
    return CodesignCampaignResult(
        config,
        initial_designs,
        candidate_designs,
        tuple(initial_summaries),
        tuple(bayesian_results),
        tuple(robustness_results),
    )


def format_codesign_campaign_report(result: CodesignCampaignResult) -> str:
    """Return a compact, reproducible plain-text result summary."""

    assembly = result.config.assembly
    lines = [
        "ThermoTwin material/geometry Bayesian co-design campaign",
        (
            "specific electrical contact resistivity: "
            f"{assembly.specific_electrical_contact_resistivity:.2e} ohm m^2 "
            "per metal/TE interface"
        ),
        (
            f"budget: {result.config.initial_design_count} initial + "
            f"{result.config.bayesian_iterations} selected prototypes per application"
        ),
        (
            f"candidate pool: {result.config.candidate_design_count}; "
            f"random baselines: {result.config.random_search_repetitions}; "
            f"robustness trials: {result.config.robustness_trials}"
        ),
    ]
    for summary, bayesian, robustness in zip(
        result.initial_summaries,
        result.bayesian_results,
        result.robustness_results,
    ):
        selected = bayesian.selected
        contact_share = (
            100.0
            * selected.electrical_contact_resistance
            / selected.thermoelectric_parameters.electrical_resistance
        )
        lines.extend(
            (
                "",
                f"{summary.application.label}:",
                (
                    f"  initial feasible: {summary.feasible_count}/"
                    f"{len(summary.evaluations)}"
                ),
                (
                    f"  selected: {selected.design.design_id}, "
                    f"p={selected.design.p_material.sample_id}, "
                    f"n={selected.design.n_material.sample_id}, "
                    f"N={selected.design.geometry.couple_count}, "
                    f"L={selected.design.geometry.leg_length * 1e3:.3f} mm, "
                    f"A={selected.design.geometry.leg_area * 1e6:.3f} mm^2"
                ),
                (
                    f"  operating point: I={selected.mean_current:.3f} A, "
                    f"Qc={selected.delivered_cooling_rate:.3f} W, "
                    f"wall COP={selected.wall_cooling_cop:.3f}, "
                    f"cost index={selected.prototype_cost_index:.3f}"
                ),
                (
                    "  electrical resistance: "
                    f"bulk={selected.bulk_leg_electrical_resistance:.4f} ohm, "
                    f"contacts={selected.electrical_contact_resistance:.4f} ohm "
                    f"({contact_share:.1f}% total)"
                ),
                (
                    "  peak current density: "
                    f"{selected.peak_current_density / 1.0e6:.4f} A/mm^2, "
                    f"{100.0 * selected.current_density_utilization:.2f}% of limit, "
                    "binding="
                    f"{'yes' if selected.current_density_constraint_binding else 'no'}"
                ),
                (
                    f"  utility: initial={bayesian.best_utility_history[0]:.4f}, "
                    f"Bayesian={bayesian.best_utility_history[-1]:.4f}, "
                    f"random median={bayesian.random_median_history[-1]:.4f}, "
                    f"pool oracle={bayesian.oracle_best.utility:.4f}"
                ),
                (
                    "  fixed-current robustness: "
                    f"{100 * robustness.feasible_fraction:.1f}% feasible; "
                    "Qc 5/50/95%="
                    f"{robustness.cooling_rate_quantiles[0]:.3f}/"
                    f"{robustness.cooling_rate_quantiles[1]:.3f}/"
                    f"{robustness.cooling_rate_quantiles[2]:.3f} W; "
                    "COP 5/50/95%="
                    f"{robustness.wall_cop_quantiles[0]:.3f}/"
                    f"{robustness.wall_cop_quantiles[1]:.3f}/"
                    f"{robustness.wall_cop_quantiles[2]:.3f}"
                ),
            )
        )
    return "\n".join(lines)


def main() -> None:
    """Run the default CPU-first campaign and print its text report."""

    print(format_codesign_campaign_report(run_codesign_campaign()))


if __name__ == "__main__":
    main()
