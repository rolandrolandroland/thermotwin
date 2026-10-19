"""Prospective 1-D inverse-PINN validation using independent synthetic truth.

This additive study does not change a trainer or the operating-decision source.
Its final checkpoint is fixed by epoch count before truth is evaluated.  The
independent nodal/SSPRK3/cubic truth differs from the finite-volume/RK4/knot
truth used by the earlier training-budget audit.  Synthetic validation does not
establish recovery of a hardware material curve.
"""

from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import fmean
import sys
import time
import traceback
from typing import Sequence

from ..core.controls import current_at
from ..observations.distributed import (
    DistributedObservationChannels,
    add_distributed_gaussian_noise,
)
from ..simulation.distributed import run_distributed_leg_experiment
from ..simulation.distributed_independent import (
    independent_distributed_voltage,
    interpolate_independent_position,
    interpolate_independent_temperature,
    observe_independent_distributed_result,
    run_independent_distributed_experiment,
)
from .distributed_independent_validation import (
    DistributedIndependentValidationConfig,
    distributed_independent_validation_experiments,
    evaluate_independent_prediction,
)
from .distributed_pinn_training_audit import (
    distributed_curve_shape_metrics,
    reference_temperature_rate_rms,
)


@dataclass(frozen=True)
class Fresh1DValidationConfig:
    """Fixed validation design; changing it defines a different protocol."""

    namespace: str = "pinn_spatial_v1_2026_10_fresh_1d"
    trial_count: int = 3
    epochs: int = 2_400
    physics_weight: float = 10.0
    smoothness_weight: float = 1.0e-4
    truth_node_count: int = 25
    truth_time_step: float = 2.5e-4
    observation_interval: float = 0.08
    temperature_standard_deviation: float = 0.01
    voltage_standard_deviation: float = 1.0e-5
    maximum_normalized_observation_loss: float = 2.0
    maximum_physics_residual_ratio: float = 0.25
    minimum_shape_ratio: float = 0.75
    maximum_shape_ratio: float = 1.50

    def __post_init__(self):
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("a nonempty seed namespace is required")
        for name in ("trial_count", "epochs", "truth_node_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.truth_node_count < 5:
            raise ValueError("independent truth needs at least five nodes")
        for name in (
            "physics_weight", "truth_time_step", "observation_interval",
            "temperature_standard_deviation", "voltage_standard_deviation",
            "maximum_normalized_observation_loss", "maximum_physics_residual_ratio",
            "minimum_shape_ratio", "maximum_shape_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.smoothness_weight) or self.smoothness_weight < 0:
            raise ValueError("smoothness weight must be finite and nonnegative")
        if self.minimum_shape_ratio >= self.maximum_shape_ratio:
            raise ValueError("shape limits must be ordered")


def fresh_1d_seeds(namespace: str, trial_index: int, experiment_count: int = 3):
    """Derive stable, role-separated seeds outside historical small seed blocks."""

    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("a nonempty seed namespace is required")
    for name, value, minimum in (
        ("trial_index", trial_index, 0), ("experiment_count", experiment_count, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"invalid {name}")

    def derive(role):
        digest = hashlib.sha256(
            f"{namespace}\0trial={trial_index}\0{role}".encode("utf-8")
        ).digest()
        return 1_000_000 + int.from_bytes(digest[:8], "big") % (2**31 - 1_000_001)

    result = {
        "neural": derive("neural"),
        "observations": tuple(derive(f"observation={i}") for i in range(experiment_count)),
    }
    if len(set((result["neural"], *result["observations"]))) != experiment_count + 1:
        raise ValueError("seed collision; declare a new namespace before evaluation")
    return result


def fresh_1d_operational_decision(observation_loss, physics_ratio, config):
    """Truth-blind gate; no property coefficient or truth value is accepted."""

    reasons = []
    for name, value, limit in (
        ("observation_loss", observation_loss, config.maximum_normalized_observation_loss),
        ("physics_residual", physics_ratio, config.maximum_physics_residual_ratio),
    ):
        if not math.isfinite(value):
            reasons.append(f"nonfinite_{name}")
        elif value > limit:
            reasons.append(name)
    return {"acceptable": not reasons, "failure_reasons": tuple(reasons)}


def _rmse(values):
    return math.sqrt(fmean(value * value for value in values))


def fresh_1d_curve_diagnostics(inferred, baseline, truth, config, training_temperature_range=None):
    """Distinguish knot recovery, the actual cubic, and representation error.

    The least-squares projection is a function-space benchmark.  It is not an
    effective inverse truth inferred from these data and is never a fit target.
    """

    import numpy as np

    temperatures = tuple(285.0 + 0.1 * i for i in range(301))
    knot_truth = tuple(truth.value(t) / v for t, v in zip(baseline.temperatures, baseline.values))
    multipliers = tuple(v / b for v, b in zip(inferred.values, baseline.values))
    shape = distributed_curve_shape_metrics(multipliers, knot_truth)
    reasons = []
    if not shape.improves_over_best_constant:
        reasons.append("does_not_beat_constant_knot_multiplier")
    if not config.minimum_shape_ratio <= shape.amplitude_ratio <= config.maximum_shape_ratio:
        reasons.append("knot_amplitude_ratio")
    if not config.minimum_shape_ratio <= shape.center_contrast_ratio <= config.maximum_shape_ratio:
        reasons.append("knot_center_contrast_ratio")
    target = np.asarray([truth.value(t) for t in temperatures])
    predicted = np.asarray([inferred.value(t) for t in temperatures])
    basis = np.asarray([
        [baseline.with_values(tuple(float(i == j) for j in range(3))).value(t) for i in range(3)]
        for t in temperatures
    ])
    # Minimize relative curve error with a uniform temperature measure.
    projection, *_ = np.linalg.lstsq(basis / target[:, None], np.ones_like(target), rcond=None)
    projected = basis @ projection
    baseline_values = np.asarray([baseline.value(t) for t in temperatures])
    target_multiplier = target / baseline_values
    predicted_multiplier = predicted / baseline_values
    centered_target = target_multiplier - target_multiplier.mean()
    centered_prediction = predicted_multiplier - predicted_multiplier.mean()
    best_constant = float(np.sum(target / baseline_values) / len(target))
    result = {
        "knot_truth_multipliers": knot_truth,
        "inferred_multipliers": multipliers,
        "knot_shape": shape._asdict(),
        "knot_shape_recovered": not reasons,
        "knot_shape_failure_reasons": reasons,
        "actual_cubic_relative_rmse": float(np.sqrt(np.mean(((predicted - target) / target)**2))),
        "actual_cubic_maximum_relative_error": float(np.max(np.abs((predicted - target) / target))),
        "actual_cubic_centered_multiplier_shape_rmse": float(np.sqrt(np.mean((centered_prediction - centered_target)**2))),
        "actual_cubic_centered_shape_relative_rmse": float(np.sqrt(np.mean((centered_prediction - centered_target)**2) / np.mean(centered_target**2))),
        "constant_multiplier_baseline_rmse": float(np.sqrt(np.mean((best_constant - target_multiplier)**2))),
        "representation_projection_values": tuple(float(v) for v in projection),
        "representation_projection_relative_rmse": float(np.sqrt(np.mean(((projected - target) / target)**2))),
        "projection_is_data_effective_truth": False,
        "temperatures": temperatures,
        "actual_cubic_values": tuple(float(v) for v in target),
        "inferred_values": tuple(float(v) for v in predicted),
        "representation_projection_curve": tuple(float(v) for v in projected),
    }
    if training_temperature_range is not None:
        low, high = training_temperature_range
        observed_temperatures = np.linspace(low, high, 301)
        relative = np.asarray([
            (inferred.value(float(t)) - truth.value(float(t))) / truth.value(float(t))
            for t in observed_temperatures
        ])
        result["training_temperature_range_K"] = (low, high)
        result["actual_cubic_relative_rmse_in_training_temperature_range"] = float(np.sqrt(np.mean(relative**2)))
        result["actual_cubic_maximum_relative_error_in_training_temperature_range"] = float(np.max(np.abs(relative)))
    return result


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = (
        "studies/pinn_spatial_fresh_1d.py", "studies/distributed_pinn_training_audit.py",
        "studies/distributed_independent_validation.py", "pinn/distributed_inverse.py",
        "pinn/distributed_forward.py", "simulation/distributed.py",
        "simulation/distributed_independent.py", "physics/distributed.py",
        "observations/distributed.py", "core/controls.py",
    )
    return {f"thermotwin/{name}": hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _frozen_model_metrics(training, experiments, observations, config, nominal_rate_rms):
    import torch
    from ..pinn.distributed_forward import (
        DistributedForwardPINNConfig, _collocation_tensors, distributed_physics_residuals,
    )
    from ..pinn.distributed_inverse import _temperature_model_observation_predictions

    training.model.eval()
    for parameter in training.model.parameters():
        parameter.requires_grad_(False)
    observation_terms = []
    per_regime = []
    for model, experiment, dataset in zip(training.model.temperature_models, experiments, observations):
        # Voltage prediction differentiates the field in space, even when all
        # model parameters are frozen.  Input autograd must remain enabled.
        predicted = _temperature_model_observation_predictions(
            model, training.model.baseline_material, training.model.property_overrides(),
            experiment, dataset, voltage_space_points=16,
        )
        values = torch.tensor(dataset.values(), dtype=torch.float32)
        scales = torch.tensor([
            config.temperature_standard_deviation if item.channel.endswith("temperature")
            else config.voltage_standard_deviation for item in dataset.observations
        ], dtype=torch.float32)
        observation_terms.append(float(((predicted - values) / scales).detach().square().mean()))
        regime = {}
        for label, space_points, time_points in (("training_grid", 7, 18), ("denser_grid", 13, 35)):
            tensors = _collocation_tensors(experiment, DistributedForwardPINNConfig(
                interior_space_points=space_points, time_points=time_points,
            ), torch.device("cpu"))
            residuals = distributed_physics_residuals(
                model, *tensors, experiment, property_values=training.model.property_overrides(),
            )
            family_mse = tuple(float(x.detach().square().mean()) for x in residuals)
            regime[label] = {"family_mse_K2_per_s2": family_mse, "rate_rms_K_per_s": math.sqrt(fmean(family_mse))}
        per_regime.append(regime)
    physics_rms = math.sqrt(fmean(item["training_grid"]["rate_rms_K_per_s"]**2 for item in per_regime))
    dense_rms = math.sqrt(fmean(item["denser_grid"]["rate_rms_K_per_s"]**2 for item in per_regime))
    observation_loss = fmean(observation_terms)
    ratio = physics_rms / nominal_rate_rms
    return {
        "observation_loss": observation_loss,
        "observation_loss_by_regime": observation_terms,
        "physics_rate_rms_K_per_s": physics_rms,
        "physics_residual_ratio": ratio,
        "denser_grid_physics_residual_ratio": dense_rms / nominal_rate_rms,
        "nominal_reference_rate_rms_K_per_s": nominal_rate_rms,
        "operational": fresh_1d_operational_decision(observation_loss, ratio, config),
        "denser_grid_operational": fresh_1d_operational_decision(observation_loss, dense_rms / nominal_rate_rms, config),
        "per_regime_physics": per_regime,
        "state": "frozen final model after the last optimizer update",
    }


def _truth_refinement(experiments, coarse_results, config):
    """Compare truth to twice the spatial resolution and one-quarter time step."""

    checks = []
    for experiment, coarse in zip(experiments, coarse_results):
        fine = run_independent_distributed_experiment(
            experiment, node_count=2 * config.truth_node_count - 1,
            time_step=config.truth_time_step / 4,
        )
        errors, voltages = [], []
        for index in range(21):
            instant = experiment.duration * index / 20
            c = interpolate_independent_temperature(coarse.trajectory, instant)
            f = interpolate_independent_temperature(fine.trajectory, instant)
            for fraction in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
                x = fraction * experiment.geometry.length
                errors.append(interpolate_independent_position(coarse.trajectory.positions, c, x) - interpolate_independent_position(fine.trajectory.positions, f, x))
            current = current_at(experiment.current, instant)
            voltages.append(independent_distributed_voltage(experiment, c, current=current) - independent_distributed_voltage(experiment, f, current=current))
        checks.append({
            "temperature_rmse_K": _rmse(errors),
            "maximum_temperature_difference_K": max(map(abs, errors)),
            "voltage_rmse_V": _rmse(voltages),
            "maximum_voltage_difference_V": max(map(abs, voltages)),
        })
    accepted = all(
        item["maximum_temperature_difference_K"] <= config.temperature_standard_deviation / 10
        and item["maximum_voltage_difference_V"] <= config.voltage_standard_deviation / 10
        for item in checks
    )
    return {
        "coarse_nodes": config.truth_node_count, "fine_nodes": 2 * config.truth_node_count - 1,
        "coarse_time_step": config.truth_time_step, "fine_time_step": config.truth_time_step / 4,
        "criteria": "maximum differences <= one tenth of each observation noise standard deviation",
        "accepted": accepted, "regimes": checks,
        "meaning": "combined space/time refinement check, not a proof of independent physical truth",
    }


def _transfer_metrics(holdouts, inferred):
    results = []
    for holdout in holdouts:
        experiment = replace(holdout.baseline_experiment, material=replace(
            holdout.baseline_experiment.material, electrical_resistivity=inferred,
        ))
        metrics = evaluate_independent_prediction(
            experiment, run_distributed_leg_experiment(experiment),
            holdout.truth_experiment, holdout.truth_result,
        )
        checks = (
            ("cold_face_rmse", metrics.cold_face_rmse, 0.03),
            ("hot_face_rmse", metrics.hot_face_rmse, 0.03),
            ("internal_temperature_rmse", metrics.internal_temperature_rmse, 0.03),
            ("voltage_rmse", metrics.voltage_rmse, 5e-5),
            ("maximum_temperature_error", metrics.maximum_absolute_temperature_error, 0.08),
            ("energy_residual", metrics.maximum_prediction_energy_balance_residual, 1e-10),
        )
        reasons = [name for name, value, limit in checks if not math.isfinite(value) or value > limit]
        results.append({
            "name": holdout.name, "counts_toward_gate": holdout.counts_toward_gate,
            "metrics": metrics, "accepted": not reasons, "failure_reasons": reasons,
            "interpretation": "FV forward transfer of the frozen inferred property; no held-out PINN field is trained",
            "support_definition": "constitutive inference-knot domain, not the narrower training-trajectory temperature range",
        })
    return results


def run_fresh_1d_validation(output_directory, config=Fresh1DValidationConfig()):
    """Run the fixed campaign once, saving every trial including failures.

    The caller must use a new directory.  There are no resumes, checkpoint
    choices, neural retries, truth-based adjustments, or dropped failed trials.
    """

    import torch
    from ..pinn.distributed_inverse import (
        InverseDistributedPropertyConfig, train_multi_experiment_inverse_distributed_property_pinn,
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    seeds = [fresh_1d_seeds(config.namespace, i) for i in range(config.trial_count)]
    flat_seeds = [seed for block in seeds for seed in (block["neural"], *block["observations"])]
    if len(set(flat_seeds)) != len(flat_seeds):
        raise ValueError("cross-trial seed collision; freeze a new namespace")
    manifest = {
        "status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "seeds": seeds, "source_sha256": _source_hashes(),
        "runtime": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "platform": platform.platform(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()},
        "protocol": {
            "checkpoint": "final epoch only; no checkpoint selection",
            "old_audit_truth": "cell-centered FV/RK4 and 3-knot piecewise-linear resistivity",
            "fresh_truth": "independent nodal/SSPRK3 and existing smooth cubic resistivity",
            "changed_truth_protocol": True,
            "cubic_formula_changed_from_prior_independent_study": False,
            "freshness": "new namespaced neural initializations and observation-noise realizations; predefined regimes and cubic retained",
            "primary_operational": "frozen final normalized observation MSE<=2 and physics RMS/nominal model rate RMS<=0.25",
            "denser_grid": "13 interior positions x 35 times, diagnostic plus separately reported operational gate",
            "shape": "truth-known diagnostics only; amplitude and signed center contrast ratios in [0.75,1.5], beating best constant knot multiplier",
            "actual_property": "dense 285..315 K cubic error reported independently of knot shape",
            "transfer": "two predefined within-knot-domain gates, extending beyond training temperatures; 280..320 K holdout diagnostic only",
            "in_support_transfer_limits": {"face_rmse_K": 0.03, "internal_rmse_K": 0.03, "voltage_rmse_V": 5e-5, "maximum_temperature_error_K": 0.08, "energy_balance_residual_W": 1e-10},
        },
    }
    _write(output / "manifest.json", manifest)
    independent_config = DistributedIndependentValidationConfig(
        trial_count=config.trial_count, inverse_pinn_epochs=config.epochs,
        truth_node_count=config.truth_node_count, truth_time_step=config.truth_time_step,
    )
    names, baselines, truths, holdouts, baseline, truth_property = distributed_independent_validation_experiments(independent_config)
    truth_results = tuple(run_independent_distributed_experiment(
        experiment, node_count=config.truth_node_count, time_step=config.truth_time_step,
    ) for experiment in truths)
    regime_temperature_ranges = tuple((
        min(min(state) for state in result.trajectory.temperature),
        max(max(state) for state in result.trajectory.temperature),
    ) for result in truth_results)
    training_temperature_range = (
        min(low for low, _ in regime_temperature_ranges),
        max(high for _, high in regime_temperature_ranges),
    )
    clean = tuple(observe_independent_distributed_result(
        experiment, result, observation_interval=config.observation_interval,
        channels=DistributedObservationChannels(),
    ) for experiment, result in zip(truths, truth_results))
    _write(output / "truth-fixtures.json", {
        "training_names": names, "training_baselines": baselines,
        "training_experiments": truths, "training_results": truth_results,
        "holdouts": holdouts,
    })
    _write(output / "clean-observations.json", {
        "regime_names": names, "datasets": clean, "truth_property": truth_property,
        "training_temperature_ranges_K": regime_temperature_ranges,
        "inference_knot_domain_K": (285.0, 315.0),
    })
    refinement = _truth_refinement(
        (*truths, *(item.truth_experiment for item in holdouts)),
        (*truth_results, *(item.truth_result for item in holdouts)), config,
    )
    _write(output / "truth-refinement.json", refinement)
    if not refinement["accepted"]:
        # A failed numerical reference cannot support an inverse claim.  This
        # is a preflight stop, not a new truth-grid choice or a training retry.
        result = {
            "status": "blocked", "numerical_reference_passed": False,
            "blocked_reason": "independent_truth_refinement_failed",
            "config": config, "truth_refinement": refinement,
            "training_temperature_range_K": training_temperature_range,
            "inference_knot_domain_K": (285.0, 315.0),
            "denominator": config.trial_count, "completed_count": 0,
            "operational_accepted_count": 0, "denser_grid_accepted_count": 0,
            "knot_shape_recovered_count": 0, "in_support_transfer_accepted_count": 0,
            "combined_gate_accepted_count": 0, "not_run_count": config.trial_count,
            "trials": [{"trial_index": index, "seeds": block, "completed": False,
                        "not_run_reason": "independent_truth_refinement_failed"}
                       for index, block in enumerate(seeds)],
            "effective_property_truth_identified": False,
        }
        _write(output / "result.json", result)
        manifest["status"] = "blocked"
        manifest["blocked_reason"] = result["blocked_reason"]
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["source_unchanged_during_run"] = manifest["source_sha256"] == _source_hashes()
        _write(output / "manifest.json", manifest)
        print("fresh1d blocked: independent truth refinement failed; no inverse training run", flush=True)
        return result
    nominal_rate = reference_temperature_rate_rms(baselines)
    trials = []
    scales = {"cold_face_temperature": config.temperature_standard_deviation,
              "hot_face_temperature": config.temperature_standard_deviation,
              "voltage": config.voltage_standard_deviation}
    for index, block in enumerate(seeds):
        directory = output / f"trial-{index:02d}"
        directory.mkdir()
        observations = tuple(add_distributed_gaussian_noise(
            dataset, standard_deviations=scales, seed=seed,
        ) for dataset, seed in zip(clean, block["observations"]))
        _write(directory / "observations.json", {"seeds": block, "datasets": observations})
        start = time.monotonic()
        print(f"fresh1d trial {index + 1}/{config.trial_count} started", flush=True)
        trial = {"trial_index": index, "seeds": block, "completed": False}
        try:
            training_config = InverseDistributedPropertyConfig(
                property_name="electrical_resistivity", hidden_width=20, hidden_layers=3,
                interior_space_points=7, time_points=18, voltage_space_points=16,
                epochs=config.epochs, network_learning_rate=2e-3, property_learning_rate=2e-3,
                initial_log_multipliers=(math.log(0.9),) * 3,
                physics_weight=config.physics_weight, smoothness_weight=config.smoothness_weight,
                temperature_standard_deviation=config.temperature_standard_deviation,
                voltage_standard_deviation=config.voltage_standard_deviation,
                seed=block["neural"], device="cpu",
            )
            _write(directory / "training-config.json", training_config)
            training = train_multi_experiment_inverse_distributed_property_pinn(
                baselines, observations, training_config, baseline_material=baselines[0].material,
            )
            _write(directory / "training-history.json", {
                "history": training.history,
                "loss_timing": "losses before optimizer step; property values after optimizer step; never used to select a checkpoint",
            })
            operational = _frozen_model_metrics(training, baselines, observations, config, nominal_rate)
            torch.save({"epoch": config.epochs, "model_state_dict": training.model.state_dict(), "training_config": asdict(training_config)}, directory / "frozen-final.pt")
            values = tuple(float(v) for v in training.model.property_values.detach())
            inferred = baseline.with_values(values)
            curve = fresh_1d_curve_diagnostics(
                inferred, baseline, truth_property, config, training_temperature_range,
            )
            transfer = _transfer_metrics(holdouts, inferred)
            in_support_transfer = all(item["accepted"] for item in transfer if item["counts_toward_gate"])
            trial.update({
                "completed": True, "operational": operational, "curve": curve, "transfer": transfer,
                "in_support_transfer_accepted": in_support_transfer,
                "combined_gate_accepted": bool(refinement["accepted"] and operational["operational"]["acceptable"] and operational["denser_grid_operational"]["acceptable"] and curve["knot_shape_recovered"] and in_support_transfer),
                "frozen_checkpoint_sha256": hashlib.sha256((directory / "frozen-final.pt").read_bytes()).hexdigest(),
            })
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as error:
            trial.update({"failure_type": type(error).__name__, "failure": str(error), "traceback": traceback.format_exc(), "combined_gate_accepted": False})
        trial["elapsed_seconds"] = time.monotonic() - start
        _write(directory / "result.json", trial)
        trials.append(trial)
        print(f"fresh1d trial {index + 1}: completed={trial['completed']}; combined_gate={trial['combined_gate_accepted']}; seconds={trial['elapsed_seconds']:.1f}", flush=True)
    result = {
        "status": "completed", "numerical_reference_passed": True, "blocked_reason": None,
        "config": config, "truth_refinement": refinement,
        "training_temperature_range_K": training_temperature_range,
        "inference_knot_domain_K": (285.0, 315.0),
        "trials": trials, "denominator": config.trial_count,
        "completed_count": sum(item["completed"] for item in trials),
        "operational_accepted_count": sum(item.get("operational", {}).get("operational", {}).get("acceptable", False) for item in trials),
        "denser_grid_accepted_count": sum(item.get("operational", {}).get("denser_grid_operational", {}).get("acceptable", False) for item in trials),
        "knot_shape_recovered_count": sum(item.get("curve", {}).get("knot_shape_recovered", False) for item in trials),
        "in_support_transfer_accepted_count": sum(item.get("in_support_transfer_accepted", False) for item in trials),
        "combined_gate_accepted_count": sum(item["combined_gate_accepted"] for item in trials),
        "effective_property_truth_identified": False,
        "interpretation": "A small prospective synthetic validation, retaining failures; no model selection or hardware validation claim.",
    }
    _write(output / "result.json", result)
    manifest["status"] = "completed"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["source_unchanged_during_run"] = manifest["source_sha256"] == _source_hashes()
    _write(output / "manifest.json", manifest)
    return result


def main(argv: Sequence[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="A new directory for the frozen fresh evaluation")
    args = parser.parse_args(argv)
    result = run_fresh_1d_validation(args.output)
    print(json.dumps({key: value for key, value in result.items() if key.endswith("count") or key == "denominator"}, indent=2))


if __name__ == "__main__":
    main()
