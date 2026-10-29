"""Constructed diagnostic tests; no scientific checkpoint solves or fitting."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from thermotwin.studies import pinn_spatial_cutoff_audit as audit
from thermotwin.pinn import planar_contact_continuity as frozen
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.pinn.planar_contact_step_control import StepControlConfig, StepControlPINN


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.networks = nn.ModuleList([nn.Linear(2, 1, bias=False, dtype=torch.float64)])
        with torch.no_grad():
            self.networks[0].weight.copy_(torch.tensor([[2., .5]], dtype=torch.float64))
        self.coefficients = nn.Parameter(torch.tensor([3., -2.], dtype=torch.float64), requires_grad=False)
        self.config = SimpleNamespace(rank_rcond=1.e-7)


def toy_matrix(model):
    w = model.networks[0].weight.flatten()
    one = torch.ones((), dtype=torch.float64)
    return torch.stack((torch.stack((one, w[0])), torch.stack((w[1], one)), torch.stack((w[0], w[1]))))


def toy_objective(model, points, observations):
    value = torch.sum((toy_matrix(model)@model.coefficients-torch.tensor([1., 2., .3], dtype=torch.float64))**2)
    return value, value, value, {"constructed": value}


def toy_system(model, points, observations):
    return toy_matrix(model).detach().numpy(), np.array([1., 2., .3]), {"families": [], "coefficient_count": 2}


class CutoffAuditTests(unittest.TestCase):
    def setUp(self):
        audit._one_thread()

    def toy_context(self):
        return patch.multiple(audit, objective_continuity=toy_objective, linear_system=toy_system)

    def test_direction_uses_original_scalar_loss_and_all_hidden_components(self):
        model = ToyModel()
        head = model.coefficients.detach().clone()
        with self.toy_context():
            direction, metadata = audit.fixed_head_direction(model, {}, None)
        value = toy_objective(model, {}, None)[0]
        gradient = torch.autograd.grad(value, list(model.networks.parameters()))[0]
        self.assertEqual(metadata["terminal_fixed_head_objective"], float(value.detach()))
        self.assertEqual(direction[0].shape, torch.Size([1, 2]))
        self.assertEqual(float(torch.max(abs(direction[0]))), 1.)
        torch.testing.assert_close(direction[0], -gradient/torch.max(abs(gradient)))
        self.assertAlmostEqual(metadata["fixed_head_ad_directional_derivative"], -float(torch.sum(gradient**2)/torch.max(abs(gradient))))
        self.assertTrue(torch.equal(model.coefficients, head))
        self.assertTrue(all(parameter.grad is None for parameter in model.networks.parameters()))

    def test_nonfinite_gradient_is_rejected_even_with_finite_objective(self):
        model = ToyModel()
        hook = model.networks[0].weight.register_hook(lambda gradient: gradient*float("nan"))
        try:
            with self.toy_context(), self.assertRaisesRegex(FloatingPointError, "nonfinite"):
                audit.fixed_head_direction(model, {}, None)
        finally:
            hook.remove()

    def test_single_solve_matches_frozen_policy_on_actual_tiny_thermal_operator(self):
        model = StepControlPINN(config=StepControlConfig(width=2, layers=1, interior_points=32,
            bottom_points=8, projection_chunk=16), known_h=800.)
        points = {"interior": np.array([[.17, .13, .12], [.42, .37, .2], [.83, .72, .8]]),
            "bottom_contact": np.array([[.4, 0., .2], [.5, 0., .7]]),
            "bottom_insulated": np.array([[.1, 0., .2], [.9, 0., .7]])}
        observations = PlanarObservations(np.array([[.004, .002, 0.], [.03, .002, 20.]]), np.array([300., 299.]), .02)
        original = deepcopy(model.state_dict())
        solve = np.linalg.lstsq
        with patch.object(np.linalg, "lstsq", wraps=solve) as calls:
            result = audit.solve_readout_once(model, points, observations, audit.SolveBudget())
            self.assertEqual(calls.call_count, 1)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]))
        with patch.object(np.linalg, "lstsq", wraps=solve) as calls:
            expected = frozen.project_output(model, points, observations)
            self.assertEqual(calls.call_count, 1)
        self.assertEqual(result["monotone_policy_accepted"], expected["accepted"])
        self.assertAlmostEqual(result["pure_tsvd_objective"], expected["objective_proposed"], places=9)
        self.assertAlmostEqual(result["monotone_policy_objective"], expected["objective_after"], places=9)
        self.assertEqual(result["rank"], expected["rank"])
        self.assertLess(result["algebraic_strong_objective_abs_difference"], 1.e-8)
        json.dumps(result, allow_nan=False)

    def test_discarded_direction_can_make_raw_candidate_worse_and_policy_keeps_head(self):
        model = ToyModel()
        with torch.no_grad():
            model.coefficients.copy_(torch.tensor([-1.e9, 1.e9], dtype=torch.float64))
        matrix, rhs = np.array([[1., 1.], [0., 1.e-9]]), np.array([0., 1.])
        def objective(model, points, observations):
            value = torch.sum((torch.tensor(matrix)@model.coefficients-torch.tensor(rhs))**2)
            return value, value, value, {}
        before = model.coefficients.detach().clone()
        with patch.multiple(audit, objective_continuity=objective, linear_system=lambda *args: (matrix, rhs, {})):
            result = audit.solve_readout_once(model, {}, None, audit.SolveBudget())
        self.assertEqual(result["rank"], 1)
        self.assertFalse(result["monotone_policy_accepted"])
        self.assertEqual(result["monotone_policy_objective"], 0.)
        self.assertGreater(result["pure_tsvd_objective"], .9)
        self.assertTrue(torch.equal(model.coefficients, before))

    def test_nonfinite_raw_loss_is_not_a_successful_required_point(self):
        model = ToyModel(); before = model.coefficients.detach().clone()
        def objective(model, points, observations):
            value = toy_objective(model, points, observations)[0]
            if not torch.equal(model.coefficients.detach(), before):
                value = value*float("inf")
            return value, value, value, {}
        budget = audit.SolveBudget()
        with patch.multiple(audit, objective_continuity=objective, linear_system=toy_system):
            result = audit.solve_readout_once(model, {}, None, budget)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(budget.attempted, 1)
        self.assertEqual(budget.successful, 0)
        self.assertFalse(result["monotone_policy_accepted"])
        self.assertTrue(torch.equal(model.coefficients, before))
        json.dumps(result, allow_nan=False)

    def test_seven_points_reset_terminal_hidden_and_head_and_never_construct_optimizer(self):
        model = ToyModel(); original = deepcopy(model.state_dict())
        raw = {"final_objective": float(toy_objective(model, {}, None)[0].detach())}
        seen = []
        def progress(record):
            self.assertTrue(torch.equal(model.coefficients, original["coefficients"]))
            seen.append((record, model.networks[0].weight.detach().clone()))
        budget = audit.SolveBudget()
        with self.toy_context(), patch.object(audit, "_load_case", return_value=(model, {}, None, raw)), \
             patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer forbidden")):
            row = audit.audit_case("t02", {}, budget, progress)
            direction, _ = audit.fixed_head_direction(model, {}, None)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(budget.attempted, 7)
        self.assertEqual(budget.successful, 7)
        for record, weight in seen:
            torch.testing.assert_close(weight, original["networks.0.weight"]+record["sign"]*record["radius"]*direction[0], rtol=0., atol=0.)
        self.assertTrue(row["original_state_restored_exactly"])
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]))

    def test_rank_changes_suppress_fd_and_equal_rank_is_only_sampled_stability(self):
        samples = [dict(point, status="solved", rank=2, pure_tsvd_objective=point["sign"]*point["radius"]+3.,
                        monotone_policy_objective=2*point["sign"]*point["radius"]+3.)
                   for point in audit._protocol() if point["case"] == "t02"]
        samples[1]["rank"] = 1
        rows = audit.directional_differences(samples, -3.)
        self.assertFalse(rows[0]["sampled_rank_stable"])
        self.assertIsNone(rows[0]["pure_tsvd_centered_fd"])
        self.assertTrue(rows[1]["sampled_rank_stable"])
        self.assertAlmostEqual(rows[1]["pure_tsvd_centered_fd"], 1.)
        self.assertAlmostEqual(rows[1]["monotone_policy_centered_fd"], 2.)
        self.assertIn("no envelope-theorem equality", rows[1]["limitation"])

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def fixture(self, root):
        campaign = root/"campaign"; campaign.mkdir()
        for source in ("model.py", "thermotwin/studies/pinn_spatial_cutoff_audit.py", "tests/test_pinn_spatial_cutoff_audit.py"):
            path = root/source; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# constructed source\n")
        self.write(campaign/"plan.json", {"cases": list(audit.common.CASES), "source_sha256": {"model.py": audit.common.digest_file(root/"model.py")}})
        self.write(campaign/"state.json", {"status": "completed"})
        for case in audit.CASES:
            worker = campaign/case; worker.mkdir()
            for name in audit.INPUT_NAMES:
                (worker/name).write_text("constructed input")
            index = {name: {"sha256": audit.common.digest_file(worker/name), "bytes": (worker/name).stat().st_size} for name in audit.INPUT_NAMES}
            self.write(worker/"artifact-index.json", index)
            self.write(worker/"state.json", {"status": "completed", "artifact_index_sha256": audit.common.digest_file(worker/"artifact-index.json")})
        return campaign

    def fixture_context(self, root):
        root = root.resolve()
        def validate(directory):
            return {"plan_sha256": audit.common.digest_file(Path(directory)/"plan.json"), "role": "constructed authentication"}
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(audit, "ROOT", root))
        stack.enter_context(patch.object(audit.common, "ROOT", root))
        stack.enter_context(patch.object(audit, "_parent_commit", return_value="a"*40))
        stack.enter_context(patch.object(audit, "_validate_campaign", side_effect=validate))
        return stack

    def toy_loader(self, case, records):
        model = ToyModel()
        return model, {}, None, {"final_objective": float(toy_objective(model, {}, None)[0].detach())}

    def test_plan_freeze_source_input_protection_and_exact_fourteen_constructed_solves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); campaign = self.fixture(root); output = root/"audit"
            snapshot = audit.common._tree_records(campaign)
            with self.fixture_context(root), self.toy_context(), patch.object(audit, "_load_case", side_effect=self.toy_loader):
                plan = audit.prepare_cutoff_audit(output, campaign_directory=campaign)
                self.assertEqual(len(plan["protocol"]), 14)
                self.assertEqual(plan["parent_commit"], "a"*40)
                self.assertEqual(plan["runtime"]["torch_intraop_threads"], 1)
                self.assertIn("numpy_blas_lapack_configuration", plan["runtime"])
                with patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer forbidden")):
                    result = audit.run_cutoff_audit(output)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["attempted_readout_solves"], 14)
                self.assertEqual(result["successful_readout_solves"], 14)
                self.assertFalse(result["automatic_followup_launched"])
                self.assertTrue(result["inputs_and_sources_preserved"])
                self.assertEqual(len((output/"progress.jsonl").read_text().splitlines()), 14)
                with self.assertRaisesRegex(RuntimeError, "consumed"):
                    audit.run_cutoff_audit(output)
                with patch.object(audit, "_parent_commit", return_value="b"*40):
                    self.assertTrue(audit.verify_plan(plan, output))
                (campaign/"t02"/"observations.npz").write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "input changed"):
                    audit.verify_plan(plan, output)
            self.assertEqual(snapshot["plan.json"], audit.common._tree_records(campaign)["plan.json"])
            self.assertEqual(list(output.rglob("*.pt")), [])

    def test_backend_failure_is_retained_without_retry_or_false_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); campaign = self.fixture(root); output = root/"audit"
            backend, calls = np.linalg.lstsq, 0
            def failed_once(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise np.linalg.LinAlgError("constructed SVD failure")
                return backend(*args, **kwargs)
            with self.fixture_context(root), self.toy_context(), patch.object(audit, "_load_case", side_effect=self.toy_loader):
                audit.prepare_cutoff_audit(output, campaign_directory=campaign)
                with patch.object(np.linalg, "lstsq", side_effect=failed_once):
                    result = audit.run_cutoff_audit(output)
            self.assertEqual(calls, 14)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["attempted_readout_solves"], 14)
            self.assertEqual(result["successful_readout_solves"], 13)
            samples = [sample for row in result["cases"] for sample in row["samples"]]
            self.assertEqual(len(samples), 14)
            self.assertEqual(samples[2]["status"], "failed")
            self.assertEqual((samples[2]["attempt_before"], samples[2]["attempt_after"]), (2, 3))
            self.assertEqual(sum(sample["readout_attempted"] for sample in samples), 14)

    def test_resource_failure_preserves_prefix_current_failure_and_unrun_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); campaign = self.fixture(root); output = root/"audit"
            backend, calls = np.linalg.lstsq, 0
            def resource_once(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise audit.common.ResourceLimit("constructed resource cap")
                return backend(*args, **kwargs)
            with self.fixture_context(root), self.toy_context(), patch.object(audit, "_load_case", side_effect=self.toy_loader):
                audit.prepare_cutoff_audit(output, campaign_directory=campaign)
                with patch.object(np.linalg, "lstsq", side_effect=resource_once):
                    result = audit.run_cutoff_audit(output)
            self.assertEqual(calls, 3)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["attempted_readout_solves"], 3)
            self.assertEqual(result["successful_readout_solves"], 2)
            self.assertEqual([row["status"] for row in result["cases"]], ["failed", "not_run"])
            samples = [sample for row in result["cases"] for sample in row["samples"]]
            self.assertEqual(len(samples), 14)
            self.assertEqual([sample["status"] for sample in samples[:4]], ["solved", "solved", "failed", "not_run"])
            self.assertTrue(result["cases"][0]["original_state_restored_exactly"])
            progress = [json.loads(line) for line in (output/"progress.jsonl").read_text().splitlines()]
            self.assertEqual(progress, samples[:3])
            for row in result["cases"]:
                self.assertEqual(json.loads((output/(row["case"]+".json")).read_text()), row)
            self.assertTrue(result["inputs_and_sources_preserved"])

    def test_budget_cannot_claim_a_fifteenth_solve(self):
        budget = audit.SolveBudget()
        for _ in range(14):
            budget.consume()
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            budget.consume()
        self.assertEqual(budget.attempted, 14)

    def test_training_input_loader_checks_physical_case_points_and_weights_without_solving(self):
        from thermotwin.pinn.planar_contact_step_control import save_checkpoint
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = StepControlConfig(width=2, layers=1, interior_points=32, bottom_points=8)
            model = StepControlPINN(config=config, known_h=800.)
            points = audit.graded_training_points(model.physics, config, 71)
            raw = {"physics": asdict(model.physics), "config": asdict(config), "known_h": 800.,
                "sampling_seed": 71, "training_points": {key: value.tolist() for key, value in points.items()},
                "training_point_weights": {key: np.full(len(value), 1/len(value)).tolist() for key, value in points.items()},
                "graded_point_metadata": audit.graded_point_metadata(model.physics, config, 71)}
            save_checkpoint(model, root/"final_checkpoint.pt")
            self.write(root/"training-report.json", raw)
            np.savez(root/"observations.npz", coordinates=np.array([[.004, .002, 0.]]), temperature=np.array([300.]))
            records = {name: {"path": str(root/name)} for name in audit.INPUT_NAMES}
            with patch.object(audit, "objective_continuity", side_effect=AssertionError("loader cannot solve/evaluate")):
                loaded, loaded_points, observations, _ = audit._load_case("t02", records)
            self.assertEqual(loaded.known_h, 800.)
            self.assertEqual(observations.standard_deviation, .02)
            self.assertTrue(np.array_equal(loaded_points["interior"], points["interior"]))
            with self.assertRaisesRegex(ValueError, "metadata differ"):
                audit._load_case("t04", records)
            raw["training_point_weights"]["interior"][0] = .99
            self.write(root/"training-report.json", raw)
            with self.assertRaisesRegex(ValueError, "row weights"):
                audit._load_case("t02", records)


if __name__ == "__main__":
    unittest.main()
