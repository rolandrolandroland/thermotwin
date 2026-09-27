"""Corrected one-figure summary of planning the next experiment and checking it.

This revises ``next_experiment_story`` without replacing its artifact:

* Panel 2 reports true percentage RMSE across trials. The earlier labels
  converted log error with ``exp(x) - 1``, which misstates large asymmetric
  misses (naive sensor lag read 76.0% instead of 34.16%).
* Panel 3 divides realized error by a forecast computed without the planner's
  prior, because the nonlinear refits use no prior, and shades the range a
  perfectly calibrated forecast would still produce across the trial count.
* The summary no longer calls the margin over the control real or the control
  equally expensive.

The prior-free forecast is recomputed from the planner's deterministic
sensitivity model; everything else is read from the two published sidecars.
"""

import argparse
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, NamedTuple, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

from ..inference.experiment_selection import (
    ExperimentSelectionConfig,
    score_experiment_candidate,
)
from ..numerics.matrices import gram_matrix, inverse_and_determinant
from .next_experiment_story import (
    DEFAULT_NONLINEAR_DATA_PATH,
    DEFAULT_PLANNER_DATA_PATH,
    ENERGY_BUDGET_JOULES,
    GRID,
    INK,
    MUTED,
    PARAMETER_KEYS,
    PARAMETER_LABELS,
    PARAMETER_SHORT,
    PLANNER_ERROR_KEYS,
    RULE,
    WALKTHROUGH,
    _load,
    sign_test_two_sided_p_value,
    student_t_two_sided_p_value,
)
from .paths import default_figure_path, save_figure_data, save_figure_explanation


DEFAULT_NEXT_EXPERIMENT_STORY_CORRECTED_PATH = default_figure_path(
    "next_experiment_story_corrected.png", WALKTHROUGH
)

FIGURE_EXPLANATION = (
    "This corrected summary of both next-experiment campaigns shows the "
    "constrained candidate frontier, the percentage error each compared pulse "
    "achieved under complete nonlinear refits, whether the planner's prior-free "
    "precision forecast matched those refits within the range the trial count "
    "can resolve, and the paired per-trial margin over the closest-energy "
    "control. It replaces an earlier version whose percentage labels misstated "
    "large errors and whose forecast comparison mixed a prior-based forecast "
    "with prior-free fits."
)

ROLE_STYLES = {
    "selected": ("Selected", "#C0152F", "o"),
    "naive": ("Naive", "#E08A2E", "^"),
    "resource_control": ("Control", "#5266A3", "s"),
}

ROLE_CANDIDATES = {
    "selected": (0.8, 20.0),
    "naive": (0.4, 5.0),
    "resource_control": (0.6, 30.0),
}

CANDIDATE_NAMES = {
    "selected": "0.8A_20s",
    "naive": "0.4A_5s",
    "resource_control": "0.6A_30s",
}

COVERAGE = 0.95


class ArmSummary(NamedTuple):
    role: str
    current_amplitude: float
    pulse_duration: float
    electrical_energy: float
    information_gain_nats: float
    prior_forecast_log_standard_errors: tuple[float, float, float]
    data_forecast_log_standard_errors: tuple[float, float, float]
    realized_log_rmse: tuple[float, float, float]
    realized_percent_rmse: tuple[float, float, float]
    joint_log_rmse: tuple[float, ...]


class CorrectedStory(NamedTuple):
    candidates: tuple[Mapping[str, Any], ...]
    arms: tuple[ArmSummary, ...]
    trial_count: int
    calibration_band: tuple[float, float]
    selected_wins_vs_control: int
    selected_wins_vs_naive: int
    control_margins: tuple[float, ...]
    mean_control_margin: float
    sign_test_p_value: float
    paired_t_statistic: float
    paired_t_p_value: float


def _regularized_lower_gamma(shape: float, x: float) -> float:
    """Return P(shape, x) from its power series."""

    if x <= 0.0:
        return 0.0
    term = 1.0 / shape
    total = term
    for index in range(1, 2000):
        term *= x / (shape + index)
        total += term
        if term < total * 1.0e-15:
            break
    return math.exp(shape * math.log(x) - x - math.lgamma(shape)) * total


def chi_square_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Return the chi-square quantile by bisection on its distribution function."""

    if not 0.0 < probability < 1.0 or degrees_of_freedom <= 0:
        raise ValueError("probability must lie in (0, 1) with positive freedom")
    low = 0.0
    high = degrees_of_freedom + 20.0 * math.sqrt(2.0 * degrees_of_freedom) + 50.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _regularized_lower_gamma(0.5 * degrees_of_freedom, 0.5 * middle) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def calibration_band(trial_count: int, coverage: float = COVERAGE) -> tuple[float, float]:
    """Return the range of RMS/sigma a calibrated forecast yields over the trials."""

    tail = 0.5 * (1.0 - coverage)
    return (
        math.sqrt(chi_square_quantile(tail, trial_count) / trial_count),
        math.sqrt(chi_square_quantile(1.0 - tail, trial_count) / trial_count),
    )


def _forecasts(
    amplitude: float,
    duration: float,
    stored: Mapping[str, Any],
    config: ExperimentSelectionConfig,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return prior and prior-free forecast log standard errors for one pulse."""

    score, jacobian = score_experiment_candidate(amplitude, duration, config)
    prior = (
        score.resistance_log_standard_error,
        score.capacitance_log_standard_error,
        score.lag_log_standard_error,
    )
    for recomputed, key in zip(prior, PLANNER_ERROR_KEYS):
        if not math.isclose(recomputed, stored[key], rel_tol=1.0e-6):
            raise ValueError(
                f"recomputed {key} {recomputed} disagrees with the stored planner "
                f"value {stored[key]}; the planner configuration has changed"
            )
    information = tuple(
        tuple(value / config.noise_standard_deviation**2 for value in row)
        for row in gram_matrix(jacobian)
    )
    covariance, _ = inverse_and_determinant(information)
    data_only = tuple(math.sqrt(covariance[index][index]) for index in range(3))
    return prior, data_only


def build_corrected_story(
    planner: Mapping[str, Any],
    nonlinear: Mapping[str, Any],
    config: ExperimentSelectionConfig = ExperimentSelectionConfig(),
) -> CorrectedStory:
    """Combine the planner, prior-free forecasts, and the paired refit campaign."""

    by_name = {item["name"]: item for item in planner["candidates"]}
    errors: dict[str, tuple[float, ...]] = {}
    arms = []
    for role, (amplitude, duration) in ROLE_CANDIDATES.items():
        stored = by_name[CANDIDATE_NAMES[role]]
        trials = sorted(
            (item for item in nonlinear["trials"] if item["role"] == role),
            key=lambda item: item["trial_index"],
        )
        if not trials:
            raise ValueError(f"the campaign sidecar has no trials for {role!r}")
        log_rmse, percent_rmse = [], []
        for index, key in enumerate(PARAMETER_KEYS):
            ratios = [
                item["fit"]["physical_values"][index] / item["truth"][key]
                for item in trials
            ]
            log_rmse.append(math.sqrt(fmean(math.log(value) ** 2 for value in ratios)))
            percent_rmse.append(100.0 * math.sqrt(fmean((value - 1.0) ** 2 for value in ratios)))
        prior, data_only = _forecasts(amplitude, duration, stored, config)
        errors[role] = tuple(item["physical_log_rmse"] for item in trials)
        arms.append(
            ArmSummary(
                role=role,
                current_amplitude=amplitude,
                pulse_duration=duration,
                electrical_energy=stored["electrical_energy"],
                information_gain_nats=stored["information_gain_nats"],
                prior_forecast_log_standard_errors=prior,
                data_forecast_log_standard_errors=data_only,
                realized_log_rmse=tuple(log_rmse),
                realized_percent_rmse=tuple(percent_rmse),
                joint_log_rmse=errors[role],
            )
        )
    selected = errors["selected"]
    margins = tuple(
        control - pick for pick, control in zip(selected, errors["resource_control"])
    )
    count = len(margins)
    wins = sum(1 for value in margins if value > 0.0)
    mean_margin = fmean(margins)
    spread = math.sqrt(sum((value - mean_margin) ** 2 for value in margins) / (count - 1))
    statistic = mean_margin / (spread / math.sqrt(count))
    return CorrectedStory(
        candidates=tuple(planner["candidates"]),
        arms=tuple(arms),
        trial_count=count,
        calibration_band=calibration_band(count),
        selected_wins_vs_control=wins,
        selected_wins_vs_naive=sum(
            1 for pick, other in zip(selected, errors["naive"]) if other > pick
        ),
        control_margins=margins,
        mean_control_margin=mean_margin,
        sign_test_p_value=sign_test_two_sided_p_value(wins, count),
        paired_t_statistic=statistic,
        paired_t_p_value=student_t_two_sided_p_value(statistic, count - 1),
    )


def format_corrected_story_report(story: CorrectedStory) -> str:
    """Return the plain-text version of the corrected story."""

    low, high = story.calibration_band
    lines = [
        "Next-experiment selection: plan, check, and margin (corrected)",
        "==============================================================",
        "",
        "Step 1 - choose under a budget:",
    ]
    for arm in story.arms:
        lines.append(
            f"  {ROLE_STYLES[arm.role][0].lower()}: {arm.current_amplitude:g} A for "
            f"{arm.pulse_duration:g} s; energy={arm.electrical_energy:.2f} J; "
            f"expected information={arm.information_gain_nats:.3f} nats"
        )
    lines.extend(("", "Step 2 - percentage RMSE after nonlinear refits:"))
    for arm in story.arms:
        lines.append(
            f"  {ROLE_STYLES[arm.role][0].lower()}: "
            + "; ".join(
                f"{PARAMETER_SHORT[index]}={value:.2f}%"
                for index, value in enumerate(arm.realized_percent_rmse)
            )
        )
    lines.extend(
        (
            "",
            "Step 3 - realized log RMSE / prior-free forecast log SE "
            f"({100 * COVERAGE:.0f}% calibrated range {low:.2f} to {high:.2f}):",
        )
    )
    for arm in story.arms:
        lines.append(
            f"  {ROLE_STYLES[arm.role][0].lower()}: "
            + "; ".join(
                f"{PARAMETER_SHORT[index]}={realized / forecast:.2f}"
                for index, (forecast, realized) in enumerate(
                    zip(arm.data_forecast_log_standard_errors, arm.realized_log_rmse)
                )
            )
        )
    lines.extend(
        (
            "",
            f"Step 4 - paired margin across {story.trial_count} trials:",
            f"  selected beat naive in {story.selected_wins_vs_naive}/{story.trial_count}",
            f"  selected beat the control in "
            f"{story.selected_wins_vs_control}/{story.trial_count}; "
            f"mean margin={story.mean_control_margin:+.5f}; "
            f"sign-test p={story.sign_test_p_value:.3f}; "
            f"paired t({story.trial_count - 1})={story.paired_t_statistic:.2f}, "
            f"p={story.paired_t_p_value:.3f}",
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


def _plot_frontier(axis, story: CorrectedStory) -> None:
    chosen = set(CANDIDATE_NAMES.values())
    others = [item for item in story.candidates if item["name"] not in chosen]
    axis.scatter(
        [item["electrical_energy"] for item in others if item["feasible"]],
        [item["information_gain_nats"] for item in others if item["feasible"]],
        s=34, color="#94A3B8", label="Other feasible pulses",
        edgecolor="white", linewidth=0.5, zorder=2,
    )
    axis.scatter(
        [item["electrical_energy"] for item in others if not item["feasible"]],
        [item["information_gain_nats"] for item in others if not item["feasible"]],
        s=42, marker="x", color="#CBD5E1", label="Rejected: over budget", zorder=2,
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
    axis.set_ylabel("Expected information gain (nats)")
    axis.legend(loc="lower right", frameon=False, fontsize=10, handletextpad=0.3)


def _plot_recovery(axis, story: CorrectedStory) -> None:
    offsets = {"selected": -0.23, "naive": 0.0, "resource_control": 0.23}
    for arm in story.arms:
        _, color, marker = ROLE_STYLES[arm.role]
        for index, value in enumerate(arm.realized_percent_rmse):
            x = index + offsets[arm.role]
            axis.plot(x, value, marker=marker, color=color, markersize=9,
                      markeredgecolor="white", markeredgewidth=0.7)
            axis.annotate(
                f"{value:.1f}%", (x, value), xytext=(0, 10),
                textcoords="offset points", ha="center", color=color, fontsize=9.5,
            )
    axis.set_yscale("log")
    axis.set_ylim(1.2, 70)
    axis.set_yticks((2, 5, 10, 20, 50))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks(range(3), PARAMETER_LABELS)
    axis.set_xlim(-0.55, 2.55)
    axis.grid(axis="x", visible=False)
    axis.set_ylabel("Percentage RMSE across trials")


def _plot_forecast_vs_realized(axis, story: CorrectedStory) -> None:
    low, high = story.calibration_band
    offsets = {"selected": -0.23, "naive": 0.0, "resource_control": 0.23}
    axis.axhspan(low, high, color="#E8EDF4", zorder=0)
    axis.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
    for arm in story.arms:
        _, color, marker = ROLE_STYLES[arm.role]
        for index, (forecast, realized) in enumerate(
            zip(arm.data_forecast_log_standard_errors, arm.realized_log_rmse)
        ):
            ratio = realized / forecast
            x = index + offsets[arm.role]
            axis.plot(x, ratio, marker=marker, color=color, markersize=9,
                      markeredgecolor="white", markeredgewidth=0.7, zorder=3)
            axis.annotate(
                f"{ratio:.2f}", (x, ratio), xytext=(0, 10),
                textcoords="offset points", ha="center", color=color, fontsize=9.5,
                bbox={"boxstyle": "square,pad=0.1", "facecolor": "#E8EDF4",
                      "edgecolor": "none"},
                zorder=4,
            )
    axis.set_xticks(range(3), PARAMETER_LABELS)
    axis.set_xlim(-0.55, 2.55)
    axis.set_ylim(0.42, 1.72)
    axis.grid(axis="x", visible=False)
    axis.set_ylabel("Realized error / forecast error")
    axis.text(
        2.52, 1.62, "↑  fits missed by more than forecast",
        color=MUTED, fontsize=9.5, ha="right", va="center",
    )
    axis.text(
        2.52, 0.50, "↓  fits missed by less than forecast",
        color=MUTED, fontsize=9.5, ha="right", va="center",
    )
    axis.legend(
        handles=[
            Line2D([], [], color=MUTED, linestyle="--", linewidth=1.2,
                   label="Missed exactly as forecast"),
            Patch(facecolor="#E8EDF4", edgecolor="none",
                  label=f"Range {story.trial_count} trials produce "
                        "if the forecast is exact"),
        ],
        loc="upper left", frameon=False, fontsize=9.5, handlelength=1.8,
    )


def _plot_margin(axis, story: CorrectedStory) -> None:
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
        f"Versus the control: won "
        f"{story.selected_wins_vs_control}/{story.trial_count},  "
        f"sign-test p = {story.sign_test_p_value:.2f}\n"
        f"mean margin {story.mean_control_margin:+.4f} (dashed),  "
        f"paired t p = {story.paired_t_p_value:.2f}",
        transform=axis.transAxes, fontsize=10.5, color=INK,
        va="top", ha="left", linespacing=1.5,
    )


def save_corrected_story_figure(story: CorrectedStory, output: Path | str) -> Path:
    """Save the corrected four-panel plan, recover, forecast, and margin summary."""

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
        0.055, 0.919,
        "25 candidate current pulses  |  three unknown thermal properties  |  "
        f"{story.trial_count} paired synthetic trials",
        fontsize=13, color=MUTED,
    )
    figure.legend(
        handles=[
            Line2D(
                [], [], color=ROLE_STYLES[arm.role][1],
                marker=ROLE_STYLES[arm.role][2], linestyle="none", markersize=9,
                label=(
                    f"{ROLE_STYLES[arm.role][0]}: {arm.current_amplitude:g} A for "
                    f"{arm.pulse_duration:g} s  ({arm.electrical_energy:.2f} J)"
                ),
            )
            for arm in story.arms
        ],
        loc="upper left", bbox_to_anchor=(0.051, 0.89), ncol=3, frameon=False,
        fontsize=11.5, handletextpad=0.4, columnspacing=2.0,
    )
    for axis in axes.flat:
        _style_axis(axis)

    _heading(
        axes[0, 0], "1.  Which pulse is worth running?",
        "Expected information per pulse  |  the budget rejects the richest ones",
    )
    _heading(
        axes[0, 1], "2.  Did it recover the properties?",
        "Percentage error after complete nonlinear refits  |  lower is better",
    )
    _heading(
        axes[1, 0], "3.  Was the forecast trustworthy?",
        "Refit error divided by the forecast made before any data",
    )
    _heading(
        axes[1, 1], "4.  How solid is the win?",
        "Same truth and noise for every pulse in a trial",
    )

    _plot_frontier(axes[0, 0], story)
    _plot_recovery(axes[0, 1], story)
    _plot_forecast_vs_realized(axes[1, 0], story)
    _plot_margin(axes[1, 1], story)

    control = next(arm for arm in story.arms if arm.role == "resource_control")
    selected = next(arm for arm in story.arms if arm.role == "selected")
    figure.text(
        0.055, 0.128,
        "A two-second calculation picked a pulse that holds up under full nonlinear "
        "refits, beats a cheap test in every trial, and forecast its precision.",
        fontsize=13, color=INK,
    )
    figure.text(
        0.055, 0.101,
        f"Its edge over a similar-energy alternative ({control.electrical_energy:.2f} J "
        f"vs {selected.electrical_energy:.2f} J) is not established: it won "
        f"{story.selected_wins_vs_control} of {story.trial_count} trials.",
        fontsize=13, color=INK,
    )
    low, high = story.calibration_band
    for offset, note in enumerate(
        (
            "Panel 1's information scores include the planner's prior. Panel 3's "
            "forecast is recomputed without it, because the nonlinear refits use no prior.",
            f"Panel 2 is percentage RMSE. Panels 3 and 4 use log-parameter errors; "
            f"panel 3's band is the {100 * COVERAGE:.0f}% chi-square range "
            f"({low:.2f}–{high:.2f}) for {story.trial_count} trials.",
            "Synthetic throughout: truth and inference share the same lumped model, "
            "so this validates the selection method and not hardware identifiability.",
        )
    ):
        figure.text(0.055, 0.063 - 0.024 * offset, note, fontsize=10.5, color=MUTED)
    figure.savefig(destination, dpi=170)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the corrected next-experiment selection story figure."
    )
    parser.add_argument("--planner", type=Path, default=DEFAULT_PLANNER_DATA_PATH)
    parser.add_argument("--nonlinear", type=Path, default=DEFAULT_NONLINEAR_DATA_PATH)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_NEXT_EXPERIMENT_STORY_CORRECTED_PATH
    )
    arguments = parser.parse_args(argv)
    story = build_corrected_story(
        _load(arguments.planner, "candidates"),
        _load(arguments.nonlinear, "trials"),
    )
    print(format_corrected_story_report(story))
    figure_path = save_corrected_story_figure(story, arguments.output)
    save_figure_data(
        {
            "trial_count": story.trial_count,
            "calibration_band": story.calibration_band,
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
