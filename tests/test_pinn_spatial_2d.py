"""Disposable helper tests; no named neural fits or benchmark partitions."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
    from thermotwin.studies import pinn_spatial_2d as study
except ModuleNotFoundError:
    study = None


@unittest.skipIf(study is None, "optional NumPy/SciPy/PyTorch dependencies are not installed")
class PlanarBenchmarkHelpersTests(unittest.TestCase):
    def _report(self, observation=1.0, residual=.01):
        return {"normalized_observation_mse": observation,
                "independent_residuals": {
                    "normalized_rms": {name: residual for name in study._RESIDUAL_FAMILIES},
                    "bottom_region_point_counts": {"bottom_contact": 30, "bottom_insulated": 40}}}

    def _candidate(self, name, epochs, weight, observation=1.0, residual=.01):
        return {"candidate_id": name, "adam_epochs": epochs, "physics_weight": weight,
                "status": "completed", "report": self._report(observation, residual)}

    def test_protocol_declares_bounded_truth_independent_selection_and_no_fit_baseline(self):
        spec = study.BENCHMARK_SPEC
        self.assertEqual(spec["campaign"], "pinn_spatial_v1_2026_10")
        self.assertEqual(spec["evaluation_trials"], 3)
        self.assertEqual(spec["h_bounds_w_m2k"], [400., 2000.])
        self.assertFalse(spec["frozen_h0_baseline"]["fit"])
        self.assertEqual(spec["neural"]["development_physics_weights"], [10., 100.])
        self.assertEqual(spec["neural"]["development_adam_epochs"], [1800, 5400])
        self.assertEqual(spec["neural"]["lbfgs_iterations"], 200)
        json.dumps(spec, allow_nan=False)

    def test_sensor_missingness_retains_exactly_36_declared_points(self):
        full, mask = study._observation_coordinates()
        self.assertEqual(full.shape, (39, 3))
        self.assertEqual(int(mask.sum()), 36)
        np.testing.assert_array_equal(full[~mask], [[.020, .003, 40], [.020, .003, 50], [.020, .003, 60]])
        self.assertEqual(int(np.sum(full[mask, 2] == 0)), 3)

    def test_hidden_grid_saves_exact_early_times_and_physical_cell_centres(self):
        coordinates, shape = study._field_coordinates()
        self.assertEqual(shape, (16, 20, 40))
        self.assertEqual(coordinates.shape, (12800, 3))
        np.testing.assert_array_equal(np.unique(coordinates[:, 2]), [0, 1, 2, 5] + list(range(10, 121, 10)))
        self.assertAlmostEqual(coordinates[:, 0].min(), .0005)
        self.assertAlmostEqual(coordinates[:, 0].max(), .0395)
        self.assertAlmostEqual(coordinates[:, 1].min(), .0001)
        self.assertAlmostEqual(coordinates[:, 1].max(), .0039)

    def test_semantic_random_streams_are_reproducible_and_separated(self):
        first = study._paired_seeds("evaluation", 0)
        self.assertEqual(first, study._paired_seeds("evaluation", 0))
        self.assertEqual(len(set(first.values())), 4)
        self.assertNotEqual(first, study._paired_seeds("evaluation", 1))
        self.assertNotEqual(first, study._paired_seeds("development", 0))
        audit = study._seed_audit()
        self.assertEqual(audit["distinct_semantic_streams"], 16)
        self.assertTrue(audit["all_unique"])
        with patch.object(study, "_semantic_seed", return_value=123):
            with self.assertRaisesRegex(ValueError, "collision"):
                study._seed_audit()

    def test_noisy_observations_have_reproducible_pointwise_noise_and_no_field_payload(self):
        class Analytic:
            def sample(self, x, z, time):
                return 300. + .3 * x - .01 * time

        one, payload = study._observations(Analytic(), 123)
        two, _ = study._observations(Analytic(), 123)
        other, _ = study._observations(Analytic(), 124)
        np.testing.assert_array_equal(one.temperature, two.temperature)
        self.assertFalse(np.array_equal(one.temperature, other.temperature))
        np.testing.assert_allclose(payload["noise"] + payload["clean_temperature"], one.temperature)
        self.assertEqual(one.coordinates.shape, (36, 3))
        self.assertFalse(one.coordinates.flags.writeable)
        self.assertFalse(hasattr(one, "hidden_field"))

    def test_mixed_bottom_residual_cannot_be_hidden_by_pooled_average(self):
        report = self._report()
        self.assertTrue(study._independent_gate(report)["passes"])
        report["independent_residuals"]["normalized_rms"]["bottom_contact"] = .06
        self.assertFalse(study._independent_gate(report)["passes"])
        report = self._report()
        report["independent_residuals"]["bottom_region_point_counts"]["bottom_insulated"] = 0
        self.assertFalse(study._independent_gate(report)["passes"])
        report = self._report()
        report["independent_residuals"]["normalized_rms"]["right"] = None
        self.assertFalse(study._independent_gate(report)["passes"])

    def test_development_selection_prefers_passing_smallest_cap_then_declared_tiebreaks(self):
        records = [self._candidate("long", 5400, 10., .1),
                   self._candidate("short_bad", 1800, 10., 2.1),
                   self._candidate("short100", 1800, 100., 1.2),
                   self._candidate("short10", 1800, 10., 1.2)]
        selection = study._select_development(records)
        self.assertEqual(selection["candidate_id"], "short10")
        self.assertTrue(selection["development_gate_passed"])
        self.assertFalse(selection["selection_uses_hidden_fields_or_true_h"])

    def test_failed_development_selection_uses_least_normalized_violation_and_keeps_failure(self):
        records = [self._candidate("obs", 1800, 10., 3.),
                   self._candidate("physics", 5400, 100., 1., .06),
                   {"candidate_id": "crash", "adam_epochs": 1800, "physics_weight": 10., "status": "failed"}]
        selection = study._select_development(records)
        self.assertEqual(selection["candidate_id"], "physics")
        self.assertFalse(selection["development_gate_passed"])
        self.assertAlmostEqual(selection["normalized_gate_violation"], .2)
        self.assertEqual(len(records), 3)

    def test_h900_passes_engineering_tolerance_but_fails_meaningful_recovery(self):
        score = study._score(np.zeros(4), np.zeros(4), 1., 900., 1000.,
                             report=self._report(), method="inverse_pinn")
        self.assertTrue(score["gates"]["contact"])
        self.assertFalse(score["gates"]["meaningful_parameter_recovery"])
        self.assertFalse(score["gates"]["joint"])
        self.assertLess(score["fraction_absolute_error_reduction"], .50)
        recovered = study._score(np.zeros(4), np.zeros(4), 1., 950., 1000.,
                                 report=self._report(), method="inverse_pinn")
        self.assertTrue(recovered["gates"]["meaningful_parameter_recovery"])

    def test_error_reduction_is_signed_and_data_only_never_emits_contact_success(self):
        worse = study._score(np.zeros(2), np.zeros(2), 1., 1300., 1000., method="conventional")
        self.assertLess(worse["fraction_absolute_error_reduction"], 0.)
        frozen = study._score(np.zeros(2), np.zeros(2), 1., study.BENCHMARK_SPEC["initial_h_w_m2k"],
                              1000., method="frozen_h0")
        self.assertEqual(frozen["fraction_absolute_error_reduction"], 0.)
        data = study._score(np.zeros(2), np.zeros(2), 1., None, 1000.,
                            report=self._report(), method="data_only")
        self.assertIsNone(data["gates"]["joint"])
        self.assertFalse(data["contact_inference_eligible"])

    def test_field_max_gate_detects_local_failure_hidden_by_low_rms(self):
        prediction = np.zeros(10000)
        prediction[0] = 1.
        score = study._score(prediction, np.zeros(10000), 1., 1000., 1000., method="conventional")
        self.assertLess(score["hidden_field"]["rmse_kelvin"], .05)
        self.assertFalse(score["gates"]["hidden_field"])
        with self.assertRaises(ValueError):
            study._field_metrics([float("nan")], [0.])

    def test_holdout_transfer_failure_blocks_an_otherwise_successful_joint_gate(self):
        score = study._score(np.zeros(2), np.zeros(2), 1., 950., 1000.,
                             report=self._report(), method="inverse_pinn")
        self.assertTrue(score["gates"]["joint"])
        study._apply_holdout_gate(score, {"rmse_kelvin": .06, "max_abs_kelvin": .10})
        self.assertFalse(score["gates"]["holdout_transfer"])
        self.assertFalse(score["gates"]["joint"])

    def test_progress_callback_nests_optimizer_stage_without_keyword_collision(self):
        captured = []

        def emit(stage, status, **details):
            captured.append({"stage": stage, "status": status, **details})

        event = {"stage": "adam", "step": 100, "mode": "inverse"}
        study._optimizer_progress(emit, "development_optimizer", candidate="disposable")(event)
        study._optimizer_progress(emit, "evaluation_optimizer", trial=0, method="data_only")(event)
        self.assertEqual(captured[0]["optimizer_event"], event)
        self.assertEqual(captured[1]["stage"], "evaluation_optimizer")
        self.assertEqual(captured[1]["method"], "data_only")

    def test_prefit_sensitivity_checks_local_information_at_all_four_contacts(self):
        class Analytic:
            def __init__(self, h, flat=False):
                self.h, self.flat = h, flat

            def sample(self, x, z, time):
                return np.full(np.broadcast(x, z, time).shape, 300. if self.flat else 300. - np.log(self.h))

        with patch.object(study, "_reference_trajectory", side_effect=lambda h, *a, **k: Analytic(h)):
            result, arrays = study._sensitivity()
        self.assertTrue(result["passes"])
        self.assertEqual([r["h_w_m2k"] for r in result["points"]], [400., 800., 1000., 2000.])
        np.testing.assert_allclose(arrays["d_temperature_d_log_h"], -1., atol=1e-9)
        with patch.object(study, "_reference_trajectory", side_effect=lambda h, *a, **k: Analytic(h, flat=True)):
            flat, _ = study._sensitivity()
        self.assertFalse(flat["passes"])
        self.assertIsNone(flat["points"][0]["optimistic_95_log_h_halfwidth"])

    def test_conventional_optimizer_recovers_constructed_scalar_without_true_h_argument(self):
        class Analytic:
            def __init__(self, h):
                self.h = h

            def sample(self, x, z, time):
                return 300. - np.log(self.h) * (1. + .01 * time)

        coordinates, mask = study._observation_coordinates()
        coordinates = coordinates[mask]
        observations = study.PlanarObservations(coordinates, Analytic(1200.).sample(*coordinates.T), .02)
        with patch.object(study, "_reference_trajectory", side_effect=lambda h, *a, **k: Analytic(h)):
            result = study._fit_conventional(observations)
        self.assertTrue(result["optimizer_success"])
        self.assertAlmostEqual(result["contact_estimate"], 1200., places=5)
        self.assertFalse(result["truth_information_used_in_fit"])

    def test_training_adapter_saves_actual_returned_state_and_failure_without_retry(self):
        coordinates, mask = study._observation_coordinates()
        observations = study.PlanarObservations(coordinates[mask], np.full(36, 300.), .02)
        seeds = study._paired_seeds("disposable_unit", 0)
        model = torch.nn.Linear(3, 1, dtype=torch.float64)
        with torch.no_grad():
            model.weight.fill_(7.25)
        config = study._config(1800)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "returned"
            with patch.object(study, "train_planar_contact_pinn", return_value=(model, {"history": [{"step": 1}]})) as train:
                returned, record = study._fit_neural(directory, observations, config=config,
                    mode="inverse", weight=10., seeds=seeds)
            self.assertIs(returned, model)
            self.assertTrue(record["checkpoint_available"])
            self.assertIsNone(train.call_args.kwargs["known_h"])
            state = torch.load(directory / "final_checkpoint.pt", weights_only=True)
            torch.testing.assert_close(state["state_dict"]["weight"], model.weight)
            compact = study._compact_fit(record)
            self.assertNotIn("history", compact["report"])
            self.assertIn("history", json.loads((directory / "fit_record.json").read_text())["report"])
            failed_directory = Path(temp) / "failed"
            with patch.object(study, "train_planar_contact_pinn", side_effect=FloatingPointError("constructed failure")) as train:
                returned, failed = study._fit_neural(failed_directory, observations, config=config,
                    mode="inverse", weight=10., seeds=seeds)
            self.assertIsNone(returned)
            self.assertEqual(train.call_count, 1)
            self.assertEqual(failed["status"], "failed")
            self.assertTrue((failed_directory / "failure.json").exists())

    def test_manifest_hashes_artifacts_and_existing_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "record.json").write_text("disposable evidence\n")
            manifest = study._manifest(root)
            self.assertEqual(manifest["record.json"]["sha256"],
                             hashlib.sha256((root / "record.json").read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                study.run_planar_benchmark(root)
            self.assertEqual((root / "record.json").read_text(), "disposable evidence\n")

    def test_reference_gate_blocks_every_fit_and_preserves_failure_record(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "disposable_blocked"
            with patch.dict(study.BENCHMARK_SPEC, campaign="disposable_unit_blocked"), \
                 patch.object(study, "verify_planar_contact_reference", return_value={"passes": False}), \
                 patch.object(study, "_sensitivity") as sensitivity, \
                 patch.object(study, "train_planar_contact_pinn") as train:
                result = study.run_planar_benchmark(output)
            self.assertEqual(result["status"], "blocked_reference")
            sensitivity.assert_not_called()
            train.assert_not_called()
            self.assertFalse((output / "development").exists())
            self.assertTrue((output / "result.json").exists())
            self.assertTrue((output / "artifact_manifest.json").exists())

    def test_disposable_mock_pipeline_retains_failed_trials_and_all_five_methods(self):
        # All expensive scientific functions are replaced with constructed
        # functions. No optimizer, named campaign or physical fit is launched.
        class FlatReference:
            energy_balance_residual = np.zeros(1)

            def sample(self, x, z, time):
                return np.full(np.broadcast(x, z, time).shape, 300.)

        calls, events = [], []

        def constructed_training(observations, **kwargs):
            calls.append(kwargs)
            kwargs["progress"]({"stage": "adam", "mode": kwargs["mode"], "step": 1})
            if len(calls) == 5:
                raise FloatingPointError("disposable evaluation failure")
            model = torch.nn.Linear(3, 1, dtype=torch.float64)
            report = self._report()
            report.update(contact_estimate=None if kwargs["mode"] == "data_only" else 1000.,
                          initial_field_digest="same_constructed_initial_state", history=[{"step": 1}])
            return model, report

        conventional = {"status": "completed", "contact_estimate": 1000.,
                        "normalized_observation_mse": 1., "wall_seconds": .001}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "disposable_pipeline"
            with patch.dict(study.BENCHMARK_SPEC, campaign="disposable_unit_pipeline"), \
                 patch.object(study, "verify_planar_contact_reference", return_value={"passes": True}), \
                 patch.object(study, "_reference_trajectory", return_value=FlatReference()), \
                 patch.object(study, "train_planar_contact_pinn", side_effect=constructed_training), \
                 patch.object(study, "predict_planar_temperature", side_effect=lambda model, c: np.full(len(c), 300.)), \
                 patch.object(study, "_fit_conventional", return_value=conventional):
                result = study.run_planar_benchmark(output, progress=events.append)
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["sensitivity_gate_passed"])
            self.assertFalse(result["contact_recovery_claim_allowed"])
            self.assertEqual(len(calls), 13)
            self.assertEqual(len(result["trials"]), 3)
            self.assertEqual(result["summary"]["inverse_pinn"]["failed_trials"], 1)
            self.assertEqual(result["summary"]["inverse_pinn"]["joint_gate_denominator"], 3)
            self.assertEqual(result["summary"]["data_only"]["reconstruction_gate_passes"], 3)
            for trial in result["trials"]:
                self.assertEqual(set(trial["methods"]), set(study.BENCHMARK_SPEC["evaluation_methods"]))
                for method in trial["methods"].values():
                    for relative in method["artifacts"].values():
                        self.assertTrue((output / relative).exists())
                    self.assertNotIn("history", method["fit"].get("report", {}))
            self.assertTrue(any(event["stage"] == "evaluation_optimizer" for event in events))
            self.assertTrue((output / "evaluation/trial_00/inverse_pinn/failure.json").exists())
            manifest = json.loads((output / "artifact_manifest.json").read_text())
            self.assertEqual(manifest["result.json"]["sha256"], study._digest(output / "result.json"))
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
