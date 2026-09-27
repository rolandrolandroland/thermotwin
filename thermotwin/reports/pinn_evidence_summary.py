"""Combine two complementary PINN experiments without mixing their evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from .paths import FIGURES_DIRECTORY, save_figure_data, save_figure_explanation


DEFAULT_RECONSTRUCTION = FIGURES_DIRECTORY / "FORWARD_RECONSTRUCTION_COMPARISON/forward_reconstruction_comparison.json"
DEFAULT_INVERSE = FIGURES_DIRECTORY / "DISTRIBUTED_PROFILE_COVERAGE/distributed_profile_coverage.json"
DEFAULT_OUTPUT = FIGURES_DIRECTORY / "PINN_EVIDENCE_SUMMARY/pinn_evidence_summary.png"
INK, MUTED, GRID = "#192C40", "#64748B", "#E2E8F0"
PHYSICS, DATA, CONVENTIONAL = "#087F8C", "#D46A34", "#5266A3"
METHODS = ("conventional_regularized", "pinn_regularized")


def load_evidence(path):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    return json.loads(raw)["data"], {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def build_summary(reconstruction, inverse):
    """Use all matched reconstruction trials and only paired inverse trials."""
    reconstruction_trials = reconstruction["trials"]
    if any(trial["initialization_maximum_absolute_difference"] != 0 for trial in reconstruction_trials):
        raise ValueError("Reconstruction networks do not have matched initial weights")
    means = {}
    for method in ("physics_informed", "data_only"):
        means[method] = {
            metric: fmean(trial[method][metric] for trial in reconstruction_trials)
            for metric in ("missing_exchanger_rmse", "hidden_face_rmse", "energy_rate_closure_rms")
        }
    inverse_trials = []
    for trial in sorted(inverse["trials"], key=lambda item: item["trial_index"]):
        estimators = {item["name"]: item for item in trial["estimators"]}
        if "pinn_regularized" not in estimators:
            continue
        if not all(method in estimators for method in METHODS):
            raise ValueError("Missing paired conventional estimate")
        inverse_trials.append({
            "trial_index": trial["trial_index"], "seeds": trial["seeds"],
            "estimators": {
                method: {
                    "property_relative_rmse_percent": 100 * estimators[method]["property_relative_rmse"],
                    "holdout_voltage_rmse_microvolts": 1e6 * estimators[method]["holdout_voltage_rmse"],
                    "multipliers": estimators[method]["multipliers"],
                } for method in METHODS
            },
        })
    if len(inverse_trials) != inverse["config"]["pinn_trial_count"]:
        raise ValueError("Incomplete paired inverse campaign")
    if len({trial["trial_index"] for trial in inverse_trials}) != len(inverse_trials):
        raise ValueError("Duplicate inverse trial indices")
    inverse_means = {
        method: {
            metric: fmean(trial["estimators"][method][metric] for trial in inverse_trials)
            for metric in ("property_relative_rmse_percent", "holdout_voltage_rmse_microvolts")
        } for method in METHODS
    }
    wins = {
        metric: sum(trial["estimators"]["pinn_regularized"][metric] <
                    trial["estimators"]["conventional_regularized"][metric]
                    for trial in inverse_trials)
        for metric in ("property_relative_rmse_percent", "holdout_voltage_rmse_microvolts")
    }
    return {
        "reconstruction": {
            "trial_count": len(reconstruction_trials),
            "retained_readings": reconstruction["retained_observation_count"],
            "missing_start_s": reconstruction["config"]["missing_start_time"],
            "missing_end_s": reconstruction["config"]["missing_end_time"],
            "means": means, "trace": reconstruction["representative_trace"],
            "gap_error_reduction_percent": 100 * (1 - means["physics_informed"]["missing_exchanger_rmse"] /
                                                   means["data_only"]["missing_exchanger_rmse"]),
            "training_time_ratio": next(s["mean_training_seconds"] for s in reconstruction["summaries"] if s["method"] == "physics_informed") /
                                   next(s["mean_training_seconds"] for s in reconstruction["summaries"] if s["method"] == "data_only"),
        },
        "inverse": {
            "trial_count": len(inverse_trials), "trials": inverse_trials,
            "means": inverse_means, "pinn_wins": wins,
            "explicit_priors": {key: inverse["config"][key] for key in ("smoothness_weight", "shrinkage_weight")},
            "epochs": inverse["config"]["inverse_pinn_epochs"],
            "fit_currents_A": [0, 0.8, -0.8],
            "held_out_current_A": 0.4, "held_out_lift_K": 20,
            "truth": "25-node nodal model, SSPRK3 integration, cubic resistivity; shared continuum physics",
            "support_K": [285, 315],
            "fitting_temperature_range_K": [294.83, 305.21],
            "extrapolative_interval_fraction": 0.654,
        },
        "interpretation": [
            "Top row: same-equation synthetic reconstruction, five matched trials; only cold traces shown, metrics pool both sides.",
            "Bottom row: independent numerical truth, ten paired trials from later profile-coverage study; stronger multistart conventional baseline with matched explicit shrinkage and curvature priors.",
            "The conventional 20-trial means must not be substituted for the ten paired-trial means shown here.",
            "These short inverse runs do not establish fully converged PDE fields or unique temperature-dependent curve-shape recovery.",
            "Physics-informed reconstruction trains more slowly. No generic forward-solver speed or accuracy claim and no hardware validation.",
        ],
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


def plot_temperature(axis, reconstruction, index, show_readings):
    trace = reconstruction["trace"]
    initial = trace["reference"][index][0]
    axis.axvspan(reconstruction["missing_start_s"], reconstruction["missing_end_s"],
                 color="#DFE6ED", alpha=0.75)
    for key, color, style in (("reference", INK, "-"), ("physics_informed", PHYSICS, "--"), ("data_only", DATA, ":")):
        axis.plot(trace["time"], [value-initial for value in trace[key][index]],
                  color=color, linestyle=style, linewidth=2.3)
    if show_readings:
        points = [(time, value-initial) for time, location, value in
                  zip(trace["observation_time"], trace["observation_location"], trace["observed_temperature"])
                  if location == "cold_exchanger"]
        axis.scatter([p[0] for p in points], [p[1] for p in points], s=18, color="#94A3B8", zorder=5)
    axis.set(xlim=(0, 60), xlabel="Time (s)", ylabel="Temperature change from start (°C)")


def plot_paired(axis, inverse, metric, percent=False):
    trials = inverse["trials"]
    colors = (CONVENTIONAL, PHYSICS)
    markers = ("s", "o")
    for index, trial in enumerate(trials):
        offset = (index - (len(trials)-1)/2) * 0.018
        values = [trial["estimators"][method][metric] for method in METHODS]
        axis.plot([offset, 1+offset], values, color="#B5C0CC", alpha=0.6, linewidth=1, zorder=1)
        for x, value, color, marker in zip((offset, 1+offset), values, colors, markers):
            axis.plot(x, value, marker=marker, markersize=6.5, color=color,
                      markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    for x, method in enumerate(METHODS):
        mean = inverse["means"][method][metric]
        axis.plot([x-0.21, x+0.21], [mean, mean], color=INK, linewidth=2.6, zorder=4)
    axis.set_xticks([0, 1], ["Conventional fit", "Inverse PINN"])
    axis.set_xlim(-0.42, 1.42)
    if percent:
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    left = inverse["means"][METHODS[0]][metric]
    right = inverse["means"][METHODS[1]][metric]
    unit = "%" if percent else " µV"
    axis.text(0.03, 0.96,
              f"Mean: {left:.2f}{unit} conventional; {right:.2f}{unit} PINN\n"
              f"PINN has lower error in {inverse['pinn_wins'][metric]}/{len(trials)} paired trials.",
              transform=axis.transAxes, va="top", color=INK, fontsize=11, linespacing=1.7)


def render_summary(story, output):
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reconstruction, inverse = story["reconstruction"], story["inverse"]
    figure = Figure(figsize=(15, 12.3), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    figure.subplots_adjust(left=0.082, right=0.975, top=0.78, bottom=0.195,
                           wspace=0.29, hspace=0.97)
    figure.text(0.055, 0.955, "What physics-informed learning adds",
                fontsize=25, fontweight="bold", color=INK)
    figure.text(0.055, 0.92,
                "Study 1: recover temperatures from incomplete measurements  |  "
                f"{reconstruction['retained_readings']} noisy readings  |  {reconstruction['trial_count']} paired trials",
                fontsize=12.5, color=MUTED)
    figure.legend(handles=[
        Line2D([], [], color=INK, linewidth=2, label="Synthetic truth"),
        Line2D([], [], color=PHYSICS, linestyle="--", linewidth=2.4, label="Physics-informed network"),
        Line2D([], [], color=DATA, linestyle=":", linewidth=2.4, label="Data-only network"),
        Line2D([], [], color="#94A3B8", marker="o", linestyle="none", markersize=5, label="Sensor readings"),
    ], loc="upper left", bbox_to_anchor=(0.052, 0.891), ncol=4, frameon=False,
        fontsize=11.5, columnspacing=1.8, handlelength=2.5)
    for axis in axes.flat:
        style_axis(axis)

    axis = axes[0, 0]
    heading(axis, "1. Reconstruct a gap in the measurements",
            "Cold exchanger shown; both sensors are absent in the shaded interval")
    plot_temperature(axis, reconstruction, 2, True)
    axis.set_ylim(-1.45, 0.72)
    physics = reconstruction["means"]["physics_informed"]
    data = reconstruction["means"]["data_only"]
    axis.text(0.035, 0.965,
              f"Across both sensors: gap error {physics['missing_exchanger_rmse']:.4f}°C vs {data['missing_exchanger_rmse']:.4f}°C\n"
              f"{reconstruction['gap_error_reduction_percent']:.0f}% lower mean error with physics",
              transform=axis.transAxes, va="top", fontsize=10.5, color=INK, linespacing=1.7)

    axis = axes[0, 1]
    heading(axis, "2. Recover temperatures that were not measured",
            "Cold module face shown; neither face supplies temperature labels")
    plot_temperature(axis, reconstruction, 0, False)
    axis.set_ylim(-3.7, 1.55)
    axis.text(0.035, 0.965,
              f"Across both hidden faces: mean error\n"
              f"{physics['hidden_face_rmse']:.4f}°C with physics vs {data['hidden_face_rmse']:.2f}°C from data alone",
              transform=axis.transAxes, va="top", fontsize=10.5, color=INK, linespacing=1.7)

    figure.text(0.082, 0.496,
                "Study 2: infer a material property, then test a new operating condition  |  "
                f"{inverse['trial_count']} paired trials with independent numerical truth",
                fontsize=12.5, color=MUTED)
    axis = axes[1, 0]
    heading(axis, "3. Estimate electrical resistance as temperature changes",
            "Paired 10-trial comparison; metric spans 285–315 K, fits reached about 294.83–305.21 K")
    plot_paired(axis, inverse, "property_relative_rmse_percent", percent=True)
    axis.set(ylim=(0, 9.4), ylabel="Resistance-curve error (relative RMSE)")

    axis = axes[1, 1]
    heading(axis, "4. Predict a condition excluded from fitting",
            "Freeze the inferred curve; predict voltage at 0.4 A and a 20 K lift")
    plot_paired(axis, inverse, "holdout_voltage_rmse_microvolts")
    axis.set(ylim=(0, 56), ylabel="Voltage prediction error (RMSE, µV)")

    figure.text(0.055, 0.116,
                "Physics improves missing and hidden temperatures; the inverse PINN gives lower average errors in the paired property study.",
                fontsize=11.5, color=INK)
    notes = [
        "Bottom row: one connected pair per trial; black bars mark means. Only the same 10 trials are compared for both methods.",
        "Property RMSE spans 285–315 K; fitting reached about 294.83–305.21 K, so roughly two-thirds of that interval is extrapolative.",
        "These are two different synthetic experiments. Both know the initial/boundary conditions; the top-row data-only model has no hidden-state prior.",
        "Lower property error does not establish full curve-shape recovery. PINN training is not shown to be faster; no hardware validation is claimed.",
    ]
    for index, note in enumerate(notes):
        figure.text(0.055, 0.088-index*0.021, note, fontsize=9.25, color=MUTED)
    figure.savefig(output, dpi=200)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, default=DEFAULT_RECONSTRUCTION)
    parser.add_argument("--inverse", type=Path, default=DEFAULT_INVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    reconstruction, source1 = load_evidence(args.reconstruction)
    inverse, source2 = load_evidence(args.inverse)
    story = build_summary(reconstruction, inverse)
    story["sources"] = [source1, source2]
    output = render_summary(story, args.output)
    save_figure_data(story, output)
    save_figure_explanation(output, walkthrough="PINN_EVIDENCE_SUMMARY.md", explanation=(
        "Panels 1–2 use five paired matched-network reconstruction trials. Cold traces illustrate what sensors see "
        "and what they never measure; error annotations aggregate both exchanger channels or both hidden faces. "
        "Panels 3–4 use the ten paired trials with both regularized estimators from the later distributed profile-coverage "
        "study. The conventional comparator has multiple initial guesses and shrinkage-plus-curvature priors, with "
        "the same explicit prior weights given to the inverse PINN. Each connected pair is one shared noisy dataset; "
        "black bars show means. Property RMSE covers 285–315 K while fitting temperatures reached only about "
        "294.83–305.21 K, leaving roughly two-thirds of the evaluation interval extrapolative. Truth uses a different "
        "grid, integrator, and cubic property basis, while still sharing "
        "the continuum equations. The inferred curve is frozen for prediction at a withheld 0.4 A, 20 K-lift condition. "
        "The result shows lower average point-estimation error, not generic forward-solver superiority, calibrated "
        "PINN uncertainty, full curve-shape convergence, or a simulation speedup. All data remain synthetic."
    ))
    print(output)
    print(json.dumps(story["inverse"]["means"], indent=2))


if __name__ == "__main__":
    main()
