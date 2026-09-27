"""Build a four-panel experiment-selection story from saved evidence only."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from .paths import FIGURES_DIRECTORY, save_figure_data, save_figure_explanation


DEFAULT_PLANNER = FIGURES_DIRECTORY / "NEXT_EXPERIMENT_WALKTHROUGH/experiment_selection.json"
DEFAULT_NONLINEAR = FIGURES_DIRECTORY / "NONLINEAR_EXPERIMENT_SELECTION/nonlinear_experiment_selection.json"
DEFAULT_OUTPUT = FIGURES_DIRECTORY / "EXPERIMENT_SELECTION_SUMMARY/experiment_selection_summary.png"
STYLES = {
    "selected": ("Selected", "#087F8C", "o", -0.23),
    "naive": ("Naive", "#D46A34", "^", 0.0),
    "resource_control": ("Control", "#5266A3", "s", 0.23),
}
PARAMETERS = ("cold_contact_resistance", "cold_face_thermal_capacitance", "sensor_time_constant")
LABELS = ("Thermal contact\nresistance", "Heat\nstorage", "Sensor response\ntime")
INK, MUTED, GRID = "#192C40", "#64748B", "#E2E8F0"


def load_evidence(path):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    return json.loads(raw)["data"], {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
    }


def build_summary(planner, nonlinear):
    """Retain the forecast and calculate descriptive results for paired trials."""
    candidates = {item["name"]: item for item in planner["candidates"]}
    trials = {}
    arms = []
    for role in STYLES:
        block = sorted((item for item in nonlinear["trials"] if item["role"] == role),
                       key=lambda item: item["trial_index"])
        if not block or len({item["trial_index"] for item in block}) != len(block):
            raise ValueError(f"Missing or duplicate trials for {role}")
        trials[role] = block
        definition = next(item for item in nonlinear["definitions"] if item["role"] == role)
        candidate = definition["candidate"]
        if candidate != candidates[candidate["name"]]:
            raise ValueError("Planner and nonlinear campaign use different candidate definitions")
        rmse = [
            100 * math.sqrt(fmean(
                (trial["fit"]["physical_values"][index] / trial["truth"][key] - 1) ** 2
                for trial in block
            ))
            for index, key in enumerate(PARAMETERS)
        ]
        arms.append({
            "role": role, "candidate": candidate, "percentage_rmse": rmse,
            "mean_combined_log_error": fmean(item["physical_log_rmse"] for item in block),
        })
    selected = trials["selected"]
    for role in ("naive", "resource_control"):
        if len(selected) != len(trials[role]):
            raise ValueError("Paired campaigns have different trial counts")
        for left, right in zip(selected, trials[role]):
            for key in ("trial_index", "truth_seed", "noise_seed", "truth"):
                if left[key] != right[key]:
                    raise ValueError(f"Trials differ in {key}; cannot claim a paired comparison")
    margins = [
        {"trial_index": left["trial_index"],
         "control_minus_selected_log_error": right["physical_log_rmse"] - left["physical_log_rmse"]}
        for left, right in zip(selected, trials["resource_control"])
    ]
    means = {item["role"]: item["mean_combined_log_error"] for item in arms}
    return {
        "candidates": list(candidates.values()), "arms": arms,
        "trial_count": len(selected), "control_margins": margins,
        "control_wins": sum(item["control_minus_selected_log_error"] < 0 for item in margins),
        "selected_wins_vs_control": sum(item["control_minus_selected_log_error"] > 0 for item in margins),
        "selected_wins_vs_naive": sum(
            left["physical_log_rmse"] < right["physical_log_rmse"]
            for left, right in zip(selected, trials["naive"])
        ),
        "mean_control_margin": fmean(item["control_minus_selected_log_error"] for item in margins),
        "selected_mean_error_reduction_vs_control_percent": 100 * (1 - means["selected"] / means["resource_control"]),
        "selected_mean_error_reduction_vs_naive_percent": 100 * (1 - means["selected"] / means["naive"]),
        "energy_budget_J": nonlinear["config"]["selection"]["maximum_electrical_energy"],
        "temperature_limits_K": [
            nonlinear["config"]["selection"]["minimum_cold_face_temperature"],
            nonlinear["config"]["selection"]["maximum_hot_face_temperature"],
        ],
        "pulse_start_s": 5.0, "observation_horizon_s": 80.0,
        "percentage_rmse_definition": "100 * sqrt(mean((estimate / truth - 1)^2)), separately for each parameter",
        "paired_margin_definition": "control minus selected per-trial combined log-parameter RMSE; positive favors selected",
        "boundary": "Forecast ranking and complete nonlinear validation share synthetic lumped physics. No hardware, speedup, or general superiority claim.",
    }


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#CBD5E1")
    axis.tick_params(colors=MUTED, labelsize=11, length=0, pad=8)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for label in (axis.xaxis.label, axis.yaxis.label):
        label.set_color(MUTED)
        label.set_size(11)


def heading(axis, title, detail):
    axis.text(0, 1.19, title, transform=axis.transAxes, fontsize=15,
              fontweight="bold", color=INK)
    axis.text(0, 1.07, detail, transform=axis.transAxes, fontsize=10.5, color=MUTED)


def render_summary(story, output):
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(14.8, 11.8), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    figure.subplots_adjust(left=0.082, right=0.975, top=0.78, bottom=0.195,
                           wspace=0.29, hspace=0.73)
    figure.text(0.055, 0.955, "Choose a useful test, then check what it teaches",
                fontsize=24, fontweight="bold", color=INK)
    figure.text(0.055, 0.919,
                "ThermoTwin  |  Conventional physics + experiment planning  |  Synthetic validation",
                fontsize=13, color=MUTED)
    figure.legend(handles=[
        Line2D([], [], marker=STYLES[item["role"]][2], color=STYLES[item["role"]][1],
               linestyle="none", markersize=9,
               label=(f"{STYLES[item['role']][0]}: {item['candidate']['current_amplitude']:g} A for "
                      f"{item['candidate']['pulse_duration']:g} s ({item['candidate']['electrical_energy']:.2f} J)"))
        for item in story["arms"]
    ], loc="upper left", bbox_to_anchor=(0.051, 0.89), ncol=3, frameon=False,
        fontsize=11.5, handletextpad=0.4, columnspacing=2)
    for axis in axes.flat:
        style_axis(axis)

    axis = axes[0, 0]
    heading(axis, "1. Apply a pulse; read two sensors",
            "Change the current, then watch the heat exchangers warm and cool")
    for arm in story["arms"]:
        name, color, marker, _ = STYLES[arm["role"]]
        candidate = arm["candidate"]
        start = story["pulse_start_s"]
        stop = start + candidate["pulse_duration"]
        current = candidate["current_amplitude"]
        axis.plot([0, start, start, stop, stop, 40], [0, 0, current, current, 0, 0],
                  color=color, linewidth=2.5, alpha=0.95)
        axis.plot((start + stop) / 2, current, marker=marker, color=color, markersize=8)
    axis.set(xlim=(0, 40), ylim=(-0.035, 1.02), xlabel="Time since test starts (s)",
             ylabel="Electrical current (A)")
    axis.text(0.04, 0.93, "Infer: thermal contact resistance, heat storage, sensor response time",
              transform=axis.transAxes, color=INK, fontsize=10, va="top")
    axis.annotate("Temperature readings\ncontinue to 80 s", xy=(39, 0.015),
                  xytext=(26, 0.30), color=MUTED, fontsize=10.5,
                  arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.1})

    axis = axes[0, 1]
    feasible = [item for item in story["candidates"] if item["feasible"]]
    rejected = [item for item in story["candidates"] if not item["feasible"]]
    heading(axis, "2. Choose the most informative feasible pulse",
            f"{len(story['candidates'])} choices; {len(feasible)} meet the energy and temperature limits")
    budget = story["energy_budget_J"]
    upper = max(item["electrical_energy"] for item in story["candidates"]) * 1.07
    axis.axvspan(budget, upper, color="#F5F7FA", zorder=0)
    axis.scatter([item["electrical_energy"] for item in feasible],
                 [item["information_gain_nats"] for item in feasible],
                 s=30, color="#94A3B8", label="Other feasible pulses")
    axis.scatter([item["electrical_energy"] for item in rejected],
                 [item["information_gain_nats"] for item in rejected],
                 s=40, marker="x", color="#AEBBCC", label="Rejected pulses")
    axis.axvline(budget, linestyle="--", color=MUTED, linewidth=1.2)
    axis.text(budget+2, 3.3, f"{budget:g} J limit", color=MUTED, fontsize=10, rotation=90)
    for arm in story["arms"]:
        _, color, marker, _ = STYLES[arm["role"]]
        c = arm["candidate"]
        axis.scatter(c["electrical_energy"], c["information_gain_nats"],
                     s=120, color=color, marker=marker, edgecolor="white", linewidth=1, zorder=5)
    axis.set(xlim=(0, upper), xlabel="Predicted electrical energy (J)",
             ylabel="Expected information gain (nats)")
    axis.legend(loc="lower right", frameon=False, fontsize=10)

    axis = axes[1, 0]
    heading(axis, "3. Check what the test actually recovers",
            f"Full nonlinear fits in {story['trial_count']} paired trials  |  Lower error is better")
    for arm in story["arms"]:
        _, color, marker, offset = STYLES[arm["role"]]
        for index, value in enumerate(arm["percentage_rmse"]):
            x = index + offset
            axis.plot(x, value, marker=marker, color=color, markersize=9)
            axis.annotate(f"{value:.2f}%", (x, value), xytext=(0, 10),
                          textcoords="offset points", ha="center", color=color, fontsize=10)
    axis.set_xticks(range(3), LABELS)
    axis.set(xlim=(-0.55, 2.55), ylim=(0, 42), ylabel="Root-mean-square percentage error")
    axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))

    axis = axes[1, 1]
    heading(axis, "4. Test the advantage against a closer alternative",
            "Each point compares the same hidden truth and noise under both pulses")
    margins = sorted(item["control_minus_selected_log_error"] for item in story["control_margins"])
    colors = [STYLES["selected" if value > 0 else "resource_control"][1] for value in margins]
    positions = range(1, len(margins)+1)
    axis.vlines(positions, 0, margins, color=colors, alpha=0.6, linewidth=1.5)
    axis.scatter(positions, margins, c=colors, s=46, edgecolor="white", linewidth=0.6, zorder=4)
    axis.axhline(0, color=MUTED, linewidth=1)
    axis.axhline(story["mean_control_margin"], color=INK, linestyle="--", linewidth=1.1)
    axis.set(xlim=(0.3, len(margins)+0.7), ylim=(min(-0.055, min(margins)*1.15), max(0.095, max(margins)*1.8)),
             xlabel=f"{story['trial_count']} paired trials, sorted by difference",
             ylabel="Control error minus selected error\n(combined log-error score)")
    axis.set_xticks([])
    axis.text(0.02, 0.97,
              f"Selected wins {story['selected_wins_vs_control']}/{story['trial_count']} against control; "
              f"{story['selected_wins_vs_naive']}/{story['trial_count']} against naive.\n"
              f"Mean combined error: {story['selected_mean_error_reduction_vs_control_percent']:.1f}% lower than control.\n"
              "Above zero: selected wins. Dashed line: mean difference.",
              transform=axis.transAxes, va="top", fontsize=10.5, color=INK, linespacing=1.5)

    figure.text(0.055, 0.113,
                "The selected pulse learns much more than the smallest pulse; its advantage over the control varies across trials.",
                fontsize=12, color=INK)
    notes = [
        "Panel 2 predicts information before measurements. Panels 3–4 check the recommendation with complete nonlinear fits.",
        "The control uses 23.77 J versus 27.54 J for selected. Its comparison does not isolate pulse timing from energy exactly.",
        "Synthetic truth and inference share equations. No measured reduction in build count or development time is claimed.",
    ]
    for index, note in enumerate(notes):
        figure.text(0.055, 0.078-index*0.024, note, fontsize=10, color=MUTED)
    figure.savefig(output, dpi=200)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--nonlinear", type=Path, default=DEFAULT_NONLINEAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    planner, source1 = load_evidence(args.planner)
    nonlinear, source2 = load_evidence(args.nonlinear)
    story = build_summary(planner, nonlinear)
    story["sources"] = [source1, source2]
    output = render_summary(story, args.output)
    save_figure_data(story, output)
    save_figure_explanation(output, walkthrough="EXPERIMENT_SELECTION_SUMMARY.md", explanation=(
        "Read panels 1 through 4: (1) apply one of three pulses and measure two exchanger temperatures; "
        "(2) rank 25 candidate pulses before measurements, keeping 17 within a 30 J energy cap and temperature limits; "
        "(3) compare per-parameter percentage RMSE in 20 paired nonlinear trials; "
        "(4) inspect every paired control-minus-selected combined log-error difference. "
        "Selected wins 20/20 against naive and 12/20 against the similar-energy control. Its mean combined error "
        "is 81.46% lower than naive and 11.77% lower than control, but these are different metrics from the "
        "per-parameter percentage errors in panel 3. All input fits and forecasts are preserved; plotted summaries "
        "are calculated from those stored results. The 250-trial linear validation is not mixed into the 20-trial nonlinear result."
    ))
    print(output)


if __name__ == "__main__":
    main()
