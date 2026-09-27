"""Superseded legacy summary of planning the next experiment.

The two campaigns are reported separately: a local D-optimal planner ranks 25
candidate pulses before any data exists, and a paired nonlinear campaign refits
the winner against two comparators. Neither report shows the arc end to end.

This figure draws four panels from the two published sidecars: the constrained
choice, the recovered parameters, whether the cheap forecast predicted the
expensive result, and how much of the margin survives a paired test. It reads
stored results only. Its percentage conversion is known to be wrong for large
asymmetric errors. New builds are forced into a ``superseded`` directory; the
active figure is produced by ``next_experiment_story_corrected``.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, NamedTuple, Sequence
import warnings

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .paths import default_figure_path, save_figure_data, save_figure_explanation


WALKTHROUGH = "NONLINEAR_EXPERIMENT_SELECTION.md"

DEFAULT_NEXT_EXPERIMENT_STORY_PATH = (
    default_figure_path("next_experiment_story.png", WALKTHROUGH).parent
    / "superseded"
    / "next_experiment_story.png"
)

DEFAULT_PLANNER_DATA_PATH = default_figure_path(
    "experiment_selection.json", "NEXT_EXPERIMENT_WALKTHROUGH.md"
)

DEFAULT_NONLINEAR_DATA_PATH = default_figure_path(
    "nonlinear_experiment_selection.json", WALKTHROUGH
)

FIGURE_EXPLANATION = (
    "SUPERSEDED: this legacy figure reports 76.0% instead of approximately "
    "34.2% for naive sensor-lag error. Use next_experiment_story_corrected.png. "
    "This figure summarizes both next-experiment campaigns in one place. It "
    "shows the constrained candidate frontier that produced the recommendation, "
    "the per-parameter error each pulse achieved under complete nonlinear "
    "refits, whether the fast pre-experiment forecast predicted those refit "
    "errors, and the paired per-trial margin over a matched-energy control. "
    "Every value is read from the two published figure sidecars; nothing is "
    "recomputed."
)

ROLE_STYLES = {
    "selected": ("Selected", "#087F8C", "o"),
    "naive": ("Naive", "#D46A34", "^"),
    "resource_control": ("Control", "#5266A3", "s"),
}

ROLE_CANDIDATES = {
    "selected": "0.8A_20s",
    "naive": "0.4A_5s",
    "resource_control": "0.6A_30s",
}

PARAMETER_KEYS = (
    "cold_contact_resistance",
    "cold_face_thermal_capacitance",
    "sensor_time_constant",
)

PARAMETER_LABELS = (
    "Thermal contact\nresistance",
    "Heat\nstorage",
    "Sensor response\ntime",
)

PARAMETER_SHORT = ("contact R", "heat C", "lag")

PLANNER_ERROR_KEYS = (
    "resistance_log_standard_error",
    "capacitance_log_standard_error",
    "lag_log_standard_error",
)

ENERGY_BUDGET_JOULES = 30.0

INK, MUTED, RULE, GRID = "#192C40", "#64748B", "#CBD5E1", "#E2E8F0"


class ArmSummary(NamedTuple):
    """Planned and achieved performance for one candidate pulse."""

    role: str
    candidate_name: str
    current_amplitude: float
    pulse_duration: float
    electrical_energy: float
    information_gain_nats: float
    forecast_log_standard_errors: tuple[float, float, float]
    realized_log_rmse: tuple[float, float, float]
    joint_log_rmse: tuple[float, ...]


class NextExperimentStory(NamedTuple):
    candidates: tuple[Mapping[str, Any], ...]
    arms: tuple[ArmSummary, ...]
    trial_count: int
    selected_wins_vs_control: int
    selected_wins_vs_naive: int
    control_margins: tuple[float, ...]
    mean_control_margin: float
    sign_test_p_value: float
    paired_t_statistic: float
    paired_t_p_value: float


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Return I_x(a, b) by the modified Lentz continued fraction."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    tiny = 1.0e-30
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (tiny if abs(d) < tiny else d)
    fraction = d
    for step in range(1, 200):
        double = 2 * step
        for numerator in (
            step * (b - step) * x / ((a + double - 1.0) * (a + double)),
            -((a + step) * (a + b + step) * x) / ((a + double) * (a + double + 1.0)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (tiny if abs(d) < tiny else d)
            c = 1.0 + numerator / c
            c = tiny if abs(c) < tiny else c
            delta = c * d
            fraction *= delta
        if abs(delta - 1.0) < 1.0e-13:
            break
    return front * fraction / a


def student_t_two_sided_p_value(statistic: float, degrees_of_freedom: int) -> float:
    """Return the two-sided Student-t tail probability."""

    if degrees_of_freedom <= 0:
        raise ValueError("degrees of freedom must be positive")
    x = degrees_of_freedom / (degrees_of_freedom + statistic * statistic)
    return _regularized_incomplete_beta(x, 0.5 * degrees_of_freedom, 0.5)


def sign_test_two_sided_p_value(successes: int, trials: int) -> float:
    """Return the exact two-sided binomial p-value against a fair coin."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes must lie within a positive trial count")
    extreme = max(successes, trials - successes)
    tail = sum(math.comb(trials, index) for index in range(extreme, trials + 1))
    return min(1.0, 2.0 * tail / 2.0**trials)


def _load(source: Path | str, required: str) -> Mapping[str, Any]:
    path = Path(source).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    data = payload.get("data")
    if not isinstance(data, Mapping) or required not in data:
        raise ValueError(f"{path} does not contain a {required!r} block")
    return data


def build_next_experiment_story(
    planner: Mapping[str, Any],
    nonlinear: Mapping[str, Any],
) -> NextExperimentStory:
    """Combine the planner frontier and the paired nonlinear campaign."""

    by_name = {item["name"]: item for item in planner["candidates"]}
    errors: dict[str, tuple[float, ...]] = {}
    arms = []
    for role, candidate_name in ROLE_CANDIDATES.items():
        candidate = by_name[candidate_name]
        trials = sorted(
            (item for item in nonlinear["trials"] if item["role"] == role),
            key=lambda item: item["trial_index"],
        )
        if not trials:
            raise ValueError(f"the campaign sidecar has no trials for {role!r}")
        realized = []
        for index, key in enumerate(PARAMETER_KEYS):
            logs = tuple(
                math.log(item["fit"]["physical_values"][index] / item["truth"][key])
                for item in trials
            )
            realized.append(math.sqrt(fmean(value * value for value in logs)))
        errors[role] = tuple(item["physical_log_rmse"] for item in trials)
        arms.append(
            ArmSummary(
                role=role,
                candidate_name=candidate_name,
                current_amplitude=candidate["current_amplitude"],
                pulse_duration=candidate["pulse_duration"],
                electrical_energy=candidate["electrical_energy"],
                information_gain_nats=candidate["information_gain_nats"],
                forecast_log_standard_errors=tuple(
                    candidate[key] for key in PLANNER_ERROR_KEYS
                ),
                realized_log_rmse=tuple(realized),
                joint_log_rmse=errors[role],
            )
        )
    selected = errors["selected"]
    margins = tuple(
        control - pick for pick, control in zip(selected, errors["resource_control"])
    )
    wins = sum(1 for value in margins if value > 0.0)
    count = len(margins)
    mean_margin = fmean(margins)
    spread = stdev(margins) if count > 1 else 0.0
    statistic = mean_margin / (spread / math.sqrt(count)) if spread > 0.0 else math.inf
    return NextExperimentStory(
        candidates=tuple(planner["candidates"]),
        arms=tuple(arms),
        trial_count=count,
        selected_wins_vs_control=wins,
        selected_wins_vs_naive=sum(
            1
            for pick, other in zip(selected, errors["naive"])
            if other > pick
        ),
        control_margins=margins,
        mean_control_margin=mean_margin,
        sign_test_p_value=sign_test_two_sided_p_value(wins, count),
        paired_t_statistic=statistic,
        paired_t_p_value=(
            student_t_two_sided_p_value(statistic, count - 1)
            if math.isfinite(statistic)
            else 0.0
        ),
    )


def format_next_experiment_story_report(story: NextExperimentStory) -> str:
    """Return the plain-text version of the combined story."""

    feasible = sum(1 for item in story.candidates if item["feasible"])
    lines = [
        "Next-experiment selection: plan, check, and margin",
        "==================================================",
        "",
        "Step 1 - choose under a budget:",
        f"  {len(story.candidates)} candidate pulses, {feasible} feasible under "
        f"a {ENERGY_BUDGET_JOULES:.0f} J limit",
    ]
    for arm in story.arms:
        label = ROLE_STYLES[arm.role][0]
        lines.append(
            f"  {label.lower()}: {arm.current_amplitude:g} A for "
            f"{arm.pulse_duration:g} s; energy={arm.electrical_energy:.2f} J; "
            f"forecast information={arm.information_gain_nats:.3f} nats"
        )
    lines.extend(("", "Step 2 - what the nonlinear refits recovered:"))
    for arm in story.arms:
        label = ROLE_STYLES[arm.role][0]
        rendered = "; ".join(
            f"{PARAMETER_SHORT[index]}={value:.4f}"
            for index, value in enumerate(arm.realized_log_rmse)
        )
        lines.append(f"  {label.lower()}: {rendered}")
    lines.extend(("", "Step 3 - forecast against realized error:"))
    for arm in story.arms:
        label = ROLE_STYLES[arm.role][0]
        rendered = "; ".join(
            f"{PARAMETER_SHORT[index]}={realized / forecast:.2f}"
            for index, (forecast, realized) in enumerate(
                zip(arm.forecast_log_standard_errors, arm.realized_log_rmse)
            )
        )
        lines.append(f"  {label.lower()} realized/forecast: {rendered}")
    lines.extend(
        (
            "",
            f"Step 4 - paired margin across {story.trial_count} trials:",
            f"  selected beat naive in "
            f"{story.selected_wins_vs_naive}/{story.trial_count}",
            f"  selected beat the matched-energy control in "
            f"{story.selected_wins_vs_control}/{story.trial_count}; "
            f"mean margin={story.mean_control_margin:+.5f}; "
            f"sign-test p={story.sign_test_p_value:.3f}; "
            f"paired t({story.trial_count - 1})={story.paired_t_statistic:.2f}, "
            f"p={story.paired_t_p_value:.3f}",
            "",
            "Interpretation boundary:",
            "  Both campaigns are synthetic and share the same lumped equations.",
            "  The forecast standard errors include the planner's prior; the",
            "  realized errors include truth displaced from nominal, so the",
            "  two are comparable in scale but are not the same quantity.",
            "  Values are read from stored sidecars; nothing is recomputed.",
        )
    )
    return "\n".join(lines)


def _style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color(RULE)
    axis.tick_params(colors=MUTED, labelsize=11, length=0, pad=8)
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for label in (axis.xaxis.label, axis.yaxis.label):
        label.set_color(MUTED)
        label.set_size(11)


def _heading(axis, title: str, detail: str) -> None:
    axis.text(0, 1.19, title, transform=axis.transAxes,
              fontsize=15, fontweight="bold", color=INK)
    axis.text(0, 1.07, detail, transform=axis.transAxes,
              fontsize=10.5, color=MUTED)


def _plot_frontier(axis, story: NextExperimentStory) -> None:
    chosen = {arm.candidate_name for arm in story.arms}
    others = [item for item in story.candidates if item["name"] not in chosen]
    axis.scatter(
        [item["electrical_energy"] for item in others if item["feasible"]],
        [item["information_gain_nats"] for item in others if item["feasible"]],
        s=34, color="#94A3B8", alpha=0.85, label="Other feasible pulses",
        edgecolor="white", linewidth=0.5, zorder=2,
    )
    axis.scatter(
        [item["electrical_energy"] for item in others if not item["feasible"]],
        [item["information_gain_nats"] for item in others if not item["feasible"]],
        s=42, marker="x", color="#CBD5E1",
        label="Rejected: over budget", zorder=2,
    )
    axis.axvline(ENERGY_BUDGET_JOULES, color=MUTED, linestyle="--", linewidth=1.2)
    axis.text(
        ENERGY_BUDGET_JOULES - 1.6, 3.05, f"{ENERGY_BUDGET_JOULES:.0f} J budget",
        color=MUTED, fontsize=10, rotation=90, va="bottom", ha="right",
    )
    for arm in story.arms:
        _, color, marker = ROLE_STYLES[arm.role]
        axis.scatter(
            (arm.electrical_energy,), (arm.information_gain_nats,),
            marker=marker, s=150, color=color, zorder=4,
            edgecolor="white", linewidth=1.2,
        )
    axis.set_xlabel("Modeled electrical energy (J)")
    axis.set_ylabel("Forecast information gain (nats)")
    axis.legend(loc="lower right", frameon=False, fontsize=10, handletextpad=0.3)


def _plot_recovery(axis, story: NextExperimentStory) -> None:
    offsets = {"selected": -0.23, "naive": 0.0, "resource_control": 0.23}
    for arm in story.arms:
        _, color, marker = ROLE_STYLES[arm.role]
        for index, value in enumerate(arm.realized_log_rmse):
            x = index + offsets[arm.role]
            axis.plot(x, value, marker=marker, color=color, markersize=9,
                      markeredgecolor="white", markeredgewidth=0.7)
            axis.annotate(
                f"{100 * (math.exp(value) - 1):.1f}%", (x, value),
                xytext=(0, 10), textcoords="offset points", ha="center",
                color=color, fontsize=9.5,
            )
    axis.set_yscale("log")
    axis.set_xticks(range(3), PARAMETER_LABELS)
    axis.set_xlim(-0.55, 2.55)
    axis.grid(axis="x", visible=False)
    axis.set_ylabel("Error after full refits (log units)")


def _plot_forecast_vs_realized(axis, story: NextExperimentStory) -> None:
    offsets = {"selected": -0.23, "naive": 0.0, "resource_control": 0.23}
    axis.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
    for arm in story.arms:
        _, color, marker = ROLE_STYLES[arm.role]
        for index, (forecast, realized) in enumerate(
            zip(arm.forecast_log_standard_errors, arm.realized_log_rmse)
        ):
            ratio = realized / forecast
            x = index + offsets[arm.role]
            axis.plot(x, ratio, marker=marker, color=color, markersize=9,
                      markeredgecolor="white", markeredgewidth=0.7)
            axis.annotate(
                f"{ratio:.2f}", (x, ratio), xytext=(0, 10),
                textcoords="offset points", ha="center",
                color=color, fontsize=9.5,
            )
    axis.set_xticks(range(3), PARAMETER_LABELS)
    axis.set_xlim(-0.55, 2.55)
    axis.set_ylim(0.55, 3.05)
    axis.grid(axis="x", visible=False)
    axis.set_ylabel("Realized error / forecast error")
    axis.text(
        2.5, 1.04, "forecast was right", color=MUTED, fontsize=9.5,
        va="bottom", ha="right",
    )
    axis.text(
        -0.5, 2.78, "above this line the forecast was too optimistic",
        color=MUTED, fontsize=9.5,
    )


def _plot_margin(axis, story: NextExperimentStory) -> None:
    ordered = sorted(story.control_margins)
    positions = range(1, len(ordered) + 1)
    colors = [
        ROLE_STYLES["selected"][1] if value > 0 else ROLE_STYLES["resource_control"][1]
        for value in ordered
    ]
    axis.vlines(positions, 0.0, ordered, color=colors, linewidth=1.6, alpha=0.55)
    axis.scatter(positions, ordered, c=colors, s=52, edgecolor="white",
                 linewidth=0.7, zorder=3)
    axis.axhline(0.0, color=MUTED, linewidth=1.1)
    axis.axhline(story.mean_control_margin, color=INK, linestyle="--", linewidth=1.2)
    axis.set_xlim(0.2, len(ordered) + 0.8)
    axis.set_xticks(())
    axis.set_xlabel(f"{story.trial_count} paired trials, sorted by margin")
    axis.set_ylabel(
        "Control error minus selected error\n(above zero, the selected pulse won)"
    )
    axis.text(
        0.02, 0.965,
        f"Versus the cheap naive pulse: won "
        f"{story.selected_wins_vs_naive}/{story.trial_count}\n"
        f"Versus matched energy: won "
        f"{story.selected_wins_vs_control}/{story.trial_count},  "
        f"sign-test p = {story.sign_test_p_value:.2f}\n"
        f"mean margin {story.mean_control_margin:+.4f} (dashed),  "
        f"paired t p = {story.paired_t_p_value:.2f}",
        transform=axis.transAxes, fontsize=10.5, color=INK,
        va="top", ha="left", linespacing=1.5,
    )


def save_next_experiment_story_figure(
    story: NextExperimentStory,
    output: Path | str,
) -> Path:
    """Save the four-panel plan, recover, forecast, and margin summary."""

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(14.8, 11.5), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    figure.subplots_adjust(
        left=0.085, right=0.975, top=0.785, bottom=0.215,
        wspace=0.30, hspace=0.78,
    )
    figure.text(
        0.055, 0.955, "Pick the next experiment, then check the pick",
        fontsize=25, fontweight="bold", color=INK,
    )
    figure.text(
        0.975,
        0.955,
        "SUPERSEDED - USE CORRECTED FIGURE",
        fontsize=12,
        fontweight="bold",
        color="#C0152F",
        ha="right",
    )
    figure.text(
        0.055, 0.919,
        "25 candidate current pulses  |  three unknown thermal properties  |  "
        f"{story.trial_count} paired synthetic trials",
        fontsize=13, color=MUTED,
    )
    handles = [
        Line2D(
            [], [], color=ROLE_STYLES[arm.role][1],
            marker=ROLE_STYLES[arm.role][2], linestyle="none", markersize=9,
            label=(
                f"{ROLE_STYLES[arm.role][0]}: {arm.current_amplitude:g} A for "
                f"{arm.pulse_duration:g} s  ({arm.electrical_energy:.2f} J)"
            ),
        )
        for arm in story.arms
    ]
    figure.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.051, 0.89),
        ncol=3, frameon=False, fontsize=11.5, handletextpad=0.4,
        columnspacing=2.0,
    )
    for axis in axes.flat:
        _style_axis(axis)

    _heading(
        axes[0, 0], "1.  Which pulse is worth running?",
        "Forecast information per pulse  |  the budget rejects the richest ones",
    )
    _heading(
        axes[0, 1], "2.  Did it recover the properties?",
        "Error after complete nonlinear refits  |  labels give percentage error",
    )
    _heading(
        axes[1, 0], "3.  Was the forecast trustworthy?",
        "Refit error divided by the pre-experiment forecast  |  1.0 means it was right",
    )
    _heading(
        axes[1, 1], "4.  How solid is the win?",
        "Same truth and noise for every pulse in a trial",
    )

    _plot_frontier(axes[0, 0], story)
    _plot_recovery(axes[0, 1], story)
    _plot_forecast_vs_realized(axes[1, 0], story)
    _plot_margin(axes[1, 1], story)

    figure.text(
        0.055, 0.128,
        "A two-second calculation picked a pulse that survives four minutes of "
        "honest refitting, and it beats a cheap experiment decisively.",
        fontsize=13, color=INK,
    )
    figure.text(
        0.055, 0.101,
        "Its edge over an equally expensive alternative is real but small, and "
        f"{story.trial_count} trials cannot establish it.",
        fontsize=13, color=INK,
    )
    for offset, note in enumerate(
        (
            "Forecast standard errors include the planner's prior; realized errors "
            "include truth displaced from nominal. Comparable in scale, not identical.",
            "Panels 2 and 4 come from the paired nonlinear campaign; panel 1 comes "
            "from the planner. Both are read from stored sidecars, not recomputed.",
            "Synthetic throughout: truth and inference share the same lumped model, "
            "so this validates the selection method and not hardware identifiability.",
        )
    ):
        figure.text(0.055, 0.063 - 0.024 * offset, note, fontsize=10.5, color=MUTED)
    figure.savefig(destination, dpi=170)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    warnings.warn(
        "next_experiment_story is superseded; use "
        "thermotwin.reports.next_experiment_story_corrected",
        FutureWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(
        description="Render the combined next-experiment selection story figure."
    )
    parser.add_argument("--planner", type=Path, default=DEFAULT_PLANNER_DATA_PATH)
    parser.add_argument("--nonlinear", type=Path, default=DEFAULT_NONLINEAR_DATA_PATH)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_NEXT_EXPERIMENT_STORY_PATH
    )
    arguments = parser.parse_args(argv)
    story = build_next_experiment_story(
        _load(arguments.planner, "candidates"),
        _load(arguments.nonlinear, "trials"),
    )
    print(format_next_experiment_story_report(story))
    figure_path = save_next_experiment_story_figure(story, arguments.output)
    save_figure_data(
        {
            "status": "superseded",
            "superseded_by": "next_experiment_story_corrected.png",
            "known_error": (
                "naive sensor-lag error is shown as 76.0 percent instead of "
                "approximately 34.2 percent"
            ),
            "trial_count": story.trial_count,
            "arms": [item._asdict() for item in story.arms],
            "selected_wins_vs_naive": story.selected_wins_vs_naive,
            "selected_wins_vs_control": story.selected_wins_vs_control,
            "mean_control_margin": story.mean_control_margin,
            "sign_test_p_value": story.sign_test_p_value,
            "paired_t_statistic": story.paired_t_statistic,
            "paired_t_p_value": story.paired_t_p_value,
        },
        figure_path,
    )
    save_figure_explanation(
        figure_path, explanation=FIGURE_EXPLANATION, walkthrough=WALKTHROUGH
    )
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
