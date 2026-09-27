"""Paired-trial diagnostics for the nonlinear next-experiment validation.

The companion report ``nonlinear_experiment_selection`` summarizes the campaign
by averaging over trials. That view cannot show two things the paired design was
built to support: how often the selected pulse actually wins a head-to-head
trial, and whether the reported error bars match the errors that occurred.

This report answers both from the published sidecar of the same campaign. It
reads the stored result rather than repeating the multistart fits, so no fit,
covariance, interval, or profile is recomputed and no existing artifact is
replaced.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any, Mapping, NamedTuple, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .paths import default_figure_path, save_figure_data, save_figure_explanation


WALKTHROUGH = "NONLINEAR_EXPERIMENT_SELECTION.md"

DEFAULT_NONLINEAR_EXPERIMENT_SELECTION_PAIRED_PATH = default_figure_path(
    "nonlinear_experiment_selection_paired.png", WALKTHROUGH
)

DEFAULT_SOURCE_DATA_PATH = default_figure_path(
    "nonlinear_experiment_selection.json", WALKTHROUGH
)

FIGURE_EXPLANATION = (
    "This figure re-reads the stored nonlinear next-experiment campaign and "
    "shows its paired structure. It reports how many of the 20 head-to-head "
    "trials the selected pulse actually won, the per-trial margin over the "
    "closest-energy control with a sign test and a paired t statistic, whether "
    "each parameter's reported standard error matched its realized error, and "
    "the re-optimized profile curves. It recomputes nothing: every value is "
    "derived from the companion figure's JSON sidecar."
)

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

PROFILE_LABELS = (
    "Contact resistance",
    "Heat storage",
    "Sensor response time",
)

ROLE_STYLES = {
    "selected": ("Selected", "#087F8C", "o"),
    "naive": ("Naive", "#D46A34", "^"),
    "resource_control": ("Control", "#5266A3", "s"),
}

INK, MUTED, RULE, GRID = "#192C40", "#64748B", "#CBD5E1", "#E2E8F0"


class PairedComparison(NamedTuple):
    """Head-to-head record of one comparator against the selected pulse."""

    role: str
    trial_count: int
    selected_wins: int
    differences: tuple[float, ...]
    mean_difference: float
    median_difference: float
    sign_test_p_value: float
    paired_t_statistic: float
    paired_t_p_value: float


class ParameterCalibration(NamedTuple):
    """Realized error against reported standard error for one parameter."""

    role: str
    parameter_index: int
    realized_log_rmse: float
    mean_reported_log_standard_error: float
    calibration_ratio: float


class PairedDiagnostics(NamedTuple):
    trial_count: int
    definitions: tuple[Mapping[str, Any], ...]
    trial_errors: Mapping[str, tuple[float, ...]]
    comparisons: tuple[PairedComparison, ...]
    calibrations: tuple[ParameterCalibration, ...]


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
    if not math.isfinite(statistic):
        raise ValueError("t statistic must be finite")
    x = degrees_of_freedom / (degrees_of_freedom + statistic * statistic)
    return _regularized_incomplete_beta(x, 0.5 * degrees_of_freedom, 0.5)


def sign_test_two_sided_p_value(successes: int, trials: int) -> float:
    """Return the exact two-sided binomial p-value against a fair coin."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes must lie within a positive trial count")
    extreme = max(successes, trials - successes)
    tail = sum(math.comb(trials, index) for index in range(extreme, trials + 1))
    return min(1.0, 2.0 * tail / 2.0**trials)


def load_campaign_data(source: Path | str) -> Mapping[str, Any]:
    """Return the ``data`` block of a stored nonlinear-campaign sidecar."""

    path = Path(source).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    data = payload.get("data")
    if not isinstance(data, Mapping) or "trials" not in data:
        raise ValueError(f"{path} is not a nonlinear-campaign figure sidecar")
    return data


def _trials_by_role(data: Mapping[str, Any], role: str) -> tuple[Mapping[str, Any], ...]:
    trials = [item for item in data["trials"] if item["role"] == role]
    trials.sort(key=lambda item: item["trial_index"])
    if not trials:
        raise ValueError(f"the sidecar contains no trials for role {role!r}")
    return tuple(trials)


def _compare(
    selected: Sequence[Mapping[str, Any]],
    comparator: Sequence[Mapping[str, Any]],
    role: str,
) -> PairedComparison:
    if len(selected) != len(comparator):
        raise ValueError("paired comparison needs equal trial counts")
    differences = tuple(
        right["physical_log_rmse"] - left["physical_log_rmse"]
        for left, right in zip(selected, comparator)
    )
    wins = sum(1 for value in differences if value > 0.0)
    count = len(differences)
    mean_difference = fmean(differences)
    spread = stdev(differences) if count > 1 else 0.0
    statistic = (
        mean_difference / (spread / math.sqrt(count)) if spread > 0.0 else math.inf
    )
    p_value = (
        student_t_two_sided_p_value(statistic, count - 1)
        if math.isfinite(statistic)
        else 0.0
    )
    return PairedComparison(
        role=role,
        trial_count=count,
        selected_wins=wins,
        differences=differences,
        mean_difference=mean_difference,
        median_difference=median(differences),
        sign_test_p_value=sign_test_two_sided_p_value(wins, count),
        paired_t_statistic=statistic,
        paired_t_p_value=p_value,
    )


def _calibrations(
    trials: Sequence[Mapping[str, Any]],
    role: str,
) -> tuple[ParameterCalibration, ...]:
    results = []
    for index, key in enumerate(PARAMETER_KEYS):
        log_errors = tuple(
            math.log(item["fit"]["physical_values"][index] / item["truth"][key])
            for item in trials
        )
        realized = math.sqrt(fmean(value * value for value in log_errors))
        reported = fmean(
            item["fit"]["intervals"][index]["standard_error_log_or_K"]
            for item in trials
        )
        results.append(
            ParameterCalibration(
                role=role,
                parameter_index=index,
                realized_log_rmse=realized,
                mean_reported_log_standard_error=reported,
                calibration_ratio=realized / reported if reported > 0.0 else math.inf,
            )
        )
    return tuple(results)


def build_paired_diagnostics(data: Mapping[str, Any]) -> PairedDiagnostics:
    """Return paired win records and interval calibration from stored trials."""

    selected = _trials_by_role(data, "selected")
    comparisons = []
    calibrations = list(_calibrations(selected, "selected"))
    trial_errors = {
        "selected": tuple(item["physical_log_rmse"] for item in selected)
    }
    for role in ("naive", "resource_control"):
        comparator = _trials_by_role(data, role)
        comparisons.append(_compare(selected, comparator, role))
        calibrations.extend(_calibrations(comparator, role))
        trial_errors[role] = tuple(item["physical_log_rmse"] for item in comparator)
    return PairedDiagnostics(
        trial_count=len(selected),
        definitions=tuple(data["definitions"]),
        trial_errors=trial_errors,
        comparisons=tuple(comparisons),
        calibrations=tuple(calibrations),
    )


def format_paired_diagnostics_report(diagnostics: PairedDiagnostics) -> str:
    """Return the plain-text paired-trial and calibration summary."""

    lines = [
        "Paired-trial diagnostics for nonlinear next-experiment selection",
        "================================================================",
        "",
        "Question:",
        "  How often does the selected pulse win a head-to-head trial, and do",
        "  the reported standard errors match the errors that occurred?",
        "",
        f"Paired trials: {diagnostics.trial_count}",
        "",
        "Head-to-head record (joint log-parameter RMSE, selected versus each):",
    ]
    for comparison in diagnostics.comparisons:
        label = ROLE_STYLES[comparison.role][0]
        lines.append(
            f"  versus {label.lower()}: selected wins "
            f"{comparison.selected_wins}/{comparison.trial_count}; "
            f"mean margin={comparison.mean_difference:+.5f}; "
            f"median margin={comparison.median_difference:+.5f}; "
            f"sign-test p={comparison.sign_test_p_value:.3f}; "
            f"paired t({comparison.trial_count - 1})="
            f"{comparison.paired_t_statistic:.2f}, p={comparison.paired_t_p_value:.3f}"
        )
    lines.extend(("", "Interval calibration (realized log RMSE / reported log SE):"))
    for role, (label, _, _) in ROLE_STYLES.items():
        entries = [item for item in diagnostics.calibrations if item.role == role]
        rendered = "; ".join(
            f"{PROFILE_LABELS[item.parameter_index]}="
            f"{item.calibration_ratio:.2f}"
            for item in entries
        )
        lines.append(f"  {label.lower()}: {rendered}")
    lines.extend(
        (
            "",
            "Interpretation boundary:",
            "  Every value is read from the stored campaign sidecar; no fit is",
            "  recomputed and no existing artifact is replaced.",
            "  A ratio above one means the reported standard error was smaller",
            "  than the error that actually occurred for that parameter.",
            "  The sign test and paired t statistic use the same 20 paired",
            "  trials; neither is powered to resolve a small margin.",
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
    axis.yaxis.label.set_color(MUTED)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_size(11)
    axis.xaxis.label.set_size(11)


def _heading(axis, title: str, detail: str) -> None:
    axis.text(
        0, 1.19, title, transform=axis.transAxes,
        fontsize=15, fontweight="bold", color=INK,
    )
    axis.text(
        0, 1.07, detail, transform=axis.transAxes,
        fontsize=10.5, color=MUTED,
    )


def _pulse_label(diagnostics: PairedDiagnostics, role: str) -> str:
    """Return the amplitude and duration of one role's pulse."""

    candidate = next(
        item["candidate"] for item in diagnostics.definitions if item["role"] == role
    )
    return (
        f"{candidate['current_amplitude']:g} A, "
        f"{candidate['pulse_duration']:g} s"
    )


def _plot_per_pulse_errors(axis, diagnostics: PairedDiagnostics) -> None:
    """Draw each trial's error for all three pulses, joined trial by trial."""

    order = ("selected", "resource_control", "naive")
    selected = diagnostics.trial_errors["selected"]
    control = diagnostics.trial_errors["resource_control"]
    for index in range(len(selected)):
        beaten = control[index] > selected[index]
        axis.plot(
            range(3),
            [diagnostics.trial_errors[role][index] for role in order],
            color=ROLE_STYLES["selected" if beaten else "resource_control"][1],
            alpha=0.3, linewidth=1.0, zorder=1,
        )
    for position, role in enumerate(order):
        _, color, marker = ROLE_STYLES[role]
        values = diagnostics.trial_errors[role]
        axis.plot(
            [position] * len(values), values, linestyle="none", marker=marker,
            markersize=7, color=color, markeredgecolor="white",
            markeredgewidth=0.6, zorder=3,
        )
        mean = fmean(values)
        axis.plot(
            (position - 0.22, position + 0.22), (mean, mean),
            color=INK, linewidth=2.2, zorder=4,
        )
    axis.set_yscale("log")
    axis.set_xticks(
        range(3),
        tuple(
            f"{ROLE_STYLES[role][0]}\n{_pulse_label(diagnostics, role)}\n"
            f"mean {fmean(diagnostics.trial_errors[role]):.3f}"
            for role in order
        ),
    )
    axis.set_xlim(-0.5, 2.5)
    axis.grid(axis="x", visible=False)
    axis.set_ylabel("Joint log-parameter RMSE")


def _plot_margin(axis, diagnostics: PairedDiagnostics) -> None:
    comparison = next(
        item for item in diagnostics.comparisons if item.role == "resource_control"
    )
    ordered = sorted(comparison.differences)
    positions = range(1, len(ordered) + 1)
    colors = [
        ROLE_STYLES["selected"][1] if value > 0 else ROLE_STYLES["resource_control"][1]
        for value in ordered
    ]
    axis.vlines(positions, 0.0, ordered, color=colors, linewidth=1.6, alpha=0.55)
    axis.scatter(
        positions, ordered, c=colors, s=52,
        edgecolor="white", linewidth=0.7, zorder=3,
    )
    axis.axhline(0.0, color=MUTED, linewidth=1.1)
    axis.axhline(
        comparison.mean_difference, color=INK,
        linestyle="--", linewidth=1.2,
    )
    axis.set_xlim(0.2, len(ordered) + 0.8)
    axis.set_xticks(())
    axis.set_xlabel("20 paired trials, sorted by margin")
    axis.set_ylabel(
        "Control error minus selected error\n(above zero, the selected pulse won)"
    )
    axis.text(
        0.02, 0.965,
        f"Selected won {comparison.selected_wins}/{comparison.trial_count}   "
        f"sign-test p = {comparison.sign_test_p_value:.2f}\n"
        f"paired t({comparison.trial_count - 1}) = "
        f"{comparison.paired_t_statistic:.2f},  p = "
        f"{comparison.paired_t_p_value:.2f}\n"
        f"mean margin {comparison.mean_difference:+.4f}  (dashed)",
        transform=axis.transAxes, fontsize=10.5, color=INK,
        va="top", ha="left", linespacing=1.5,
    )


def _plot_calibration(axis, diagnostics: PairedDiagnostics) -> None:
    offsets = {"selected": -0.23, "naive": 0.0, "resource_control": 0.23}
    axis.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
    for item in diagnostics.calibrations:
        label, color, marker = ROLE_STYLES[item.role]
        x = item.parameter_index + offsets[item.role]
        axis.plot(
            x, item.calibration_ratio, marker=marker, color=color,
            markersize=9, markeredgecolor="white", markeredgewidth=0.7,
        )
        axis.annotate(
            f"{item.calibration_ratio:.2f}", (x, item.calibration_ratio),
            xytext=(0, 10), textcoords="offset points", ha="center",
            color=color, fontsize=10, fontweight="medium",
        )
    axis.set_xticks(range(3), PARAMETER_LABELS)
    axis.set_xlim(-0.55, 2.55)
    axis.grid(axis="x", visible=False)
    axis.set_ylabel("Realized error / reported standard error")


def _plot_profiles(axis, data: Mapping[str, Any]) -> None:
    styles = ("-", "--", ":")
    for role in ("selected", "naive"):
        label, color, _ = ROLE_STYLES[role]
        for parameter_index in range(3):
            points = [
                item
                for item in data["profiles"]
                if item["role"] == role and item["parameter_index"] == parameter_index
            ]
            if not points:
                continue
            points.sort(key=lambda item: item["fixed_log_multiplier"])
            floor = min(item["normalized_mean_squared_error"] for item in points)
            axis.plot(
                [item["fixed_log_multiplier"] for item in points],
                [
                    item["normalized_mean_squared_error"] - floor
                    for item in points
                ],
                linestyle=styles[parameter_index], marker="o", markersize=4,
                color=color, linewidth=1.7,
            )
    axis.set_yscale("symlog", linthresh=0.05)
    axis.set_xlabel("Value held fixed, as a log multiple of nominal")
    axis.set_ylabel("Increase in normalized fit error")
    axis.set_ylim(top=axis.get_ylim()[1] * 8.0)
    axis.legend(
        handles=[
            Line2D([], [], color=MUTED, linestyle=styles[index], linewidth=1.7,
                   label=PROFILE_LABELS[index])
            for index in range(3)
        ],
        fontsize=9, ncol=3, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 1.02), handlelength=2.6, columnspacing=1.4,
    )


def save_nonlinear_experiment_selection_paired_figure(
    data: Mapping[str, Any],
    diagnostics: PairedDiagnostics,
    output: Path | str,
) -> Path:
    """Save head-to-head, margin, calibration, and profile diagnostic panels."""

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(14.8, 11.5), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    figure.subplots_adjust(
        left=0.082, right=0.975, top=0.785, bottom=0.225,
        wspace=0.30, hspace=0.78,
    )
    figure.text(
        0.055, 0.955, "Did the selected experiment really win?",
        fontsize=25, fontweight="bold", color=INK,
    )
    figure.text(
        0.055, 0.919,
        f"{diagnostics.trial_count} paired trials, read back from the stored "
        "campaign  |  Same truth and same noise for every pulse",
        fontsize=13, color=MUTED,
    )
    handles = [
        Line2D(
            [], [], color=ROLE_STYLES[item["role"]][1],
            marker=ROLE_STYLES[item["role"]][2], linestyle="none", markersize=9,
            label=(
                f"{ROLE_STYLES[item['role']][0]}: "
                f"{item['candidate']['current_amplitude']:g} A for "
                f"{item['candidate']['pulse_duration']:g} s  "
                f"({item['candidate']['electrical_energy']:.2f} J)"
            ),
        )
        for item in diagnostics.definitions
    ]
    figure.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.051, 0.89),
        ncol=3, frameon=False, fontsize=11.5, handletextpad=0.4,
        columnspacing=2.0,
    )
    for axis in axes.flat:
        _style_axis(axis)

    control = next(
        item for item in diagnostics.comparisons if item.role == "resource_control"
    )
    _heading(
        axes[0, 0], "How large was the error for each pulse?",
        "One line per trial, tinted by the selected-versus-control winner"
        "  |  Lower is better",
    )
    _heading(
        axes[0, 1], "How big is the margin over matched energy?",
        "Selected minus control, per trial  |  The control spends 23.77 J",
    )
    _heading(
        axes[1, 0], "Are the reported error bars honest?",
        "Above 1.0, the reported standard error was too small",
    )
    _heading(
        axes[1, 1], "How sharply is each value pinned down?",
        "Refit with one value held wrong  |  Steeper means better determined",
    )

    _plot_per_pulse_errors(axes[0, 0], diagnostics)
    _plot_margin(axes[0, 1], diagnostics)
    _plot_calibration(axes[1, 0], diagnostics)
    _plot_profiles(axes[1, 1], data)

    figure.text(
        0.055, 0.125,
        "The selected pulse beats the cheap naive pulse in every trial, but its "
        f"edge over a matched-energy control is not resolved by {control.trial_count} trials.",
        fontsize=13, color=INK,
    )
    for offset, note in enumerate(
        (
            "Margins use the joint log-parameter RMSE stored for each trial; the "
            "companion figure reports per-parameter percentage error instead.",
            "The sign test and paired t statistic describe this frozen campaign "
            "only; neither is powered to resolve a margin of a few percent.",
            "Read back from the stored sidecar. No fit, covariance, interval, or "
            "profile is recomputed, and no existing artifact is replaced.",
        )
    ):
        figure.text(
            0.055, 0.086 - 0.024 * offset, note, fontsize=10.5, color=MUTED,
        )
    figure.savefig(destination, dpi=170)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render paired-trial diagnostics from a stored nonlinear "
            "next-experiment selection campaign."
        )
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE_DATA_PATH,
        help="campaign JSON sidecar to read back",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DEFAULT_NONLINEAR_EXPERIMENT_SELECTION_PAIRED_PATH,
        help="destination figure path",
    )
    arguments = parser.parse_args(argv)
    data = load_campaign_data(arguments.source)
    diagnostics = build_paired_diagnostics(data)
    print(format_paired_diagnostics_report(diagnostics))
    figure_path = save_nonlinear_experiment_selection_paired_figure(
        data, diagnostics, arguments.output
    )
    save_figure_data(
        {
            "trial_count": diagnostics.trial_count,
            "source_sidecar": Path(arguments.source).name,
            "comparisons": [item._asdict() for item in diagnostics.comparisons],
            "calibrations": [item._asdict() for item in diagnostics.calibrations],
            "trial_errors": dict(diagnostics.trial_errors),
        },
        figure_path,
    )
    save_figure_explanation(
        figure_path, explanation=FIGURE_EXPLANATION, walkthrough=WALKTHROUGH
    )
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
