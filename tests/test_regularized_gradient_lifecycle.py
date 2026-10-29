"""Constructed end-to-end probe accounting; no scientific input is opened."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests import test_pinn_spatial_regularized_probe as gate_fixture
from tests import test_planar_contact_regularized as model_fixture


probe = gate_fixture.probe


class RegularizedGradientLifecycleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def factory(self, physics, config, known_h, seed):
        values = asdict(config)
        values.pop("width"); values.pop("layers")
        model = model_fixture.IllConditionedModel(**values)
        model.physics, model.known_h = physics, known_h
        return model, {"common_weights_sha256": "constructed-common-initialization"}

    def fixture(self, root):
        output = root/"probe"; output.mkdir()
        config = asdict(model_fixture.IllConditionedModel().config)
        inputs = {}
        for case in probe.CASES:
            directory = root/"inputs"/case; directory.mkdir(parents=True)
            declared = next(value for value in probe.common.CASES if value["id"] == case)
            model, _ = self.factory(probe.common.physics_for(declared), probe.api.RegularizedConfig(**config), declared["known_h"], 0)
            with torch.no_grad():
                model.networks[0].weight.copy_(torch.tensor([[1.2, 1.4]], dtype=torch.float64))
            state = {name: value for name, value in model.state_dict().items() if name not in probe.api._READOUT_BUFFERS}
            torch.save({"known_h": declared["known_h"], "physics": asdict(model.physics), "config": config,
                        "state_dict": state}, directory/"final_checkpoint.pt")
            (directory/"training-report.json").write_text(json.dumps({"training_points": {}}))
            np.savez(directory/"observations.npz", coordinates=np.array([[.004, .002, 0.]]), temperature=np.array([300.]))
            inputs[case] = {name: {"path": str(directory/name)} for name in
                ("final_checkpoint.pt", "training-report.json", "observations.npz")}
        plan = {"inputs": inputs, "model_config": config}
        probe.common._write(output/"plan.json", plan)
        probe.common._write(output/"state.json", {"status": "prepared", "plan_sha256": probe.common.digest_file(output/"plan.json")})
        return output, root/"tracked-result.json"

    def context(self, output, tracked):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(probe, "OUT", output))
        stack.enter_context(patch.object(probe, "TRACKED_RESULT", tracked))
        stack.enter_context(patch.object(probe, "verify_plan", return_value=True))
        stack.enter_context(patch.object(probe, "graded_training_points", return_value={}))
        stack.enter_context(patch.object(probe.api, "matched_regularized_model", side_effect=self.factory))
        stack.enter_context(patch.object(probe.api, "linear_system", side_effect=model_fixture.toy_system))
        stack.enter_context(patch.object(probe.api, "objective_continuity", side_effect=model_fixture.toy_objective))
        stack.enter_context(patch.object(torch.optim, "Adam", side_effect=AssertionError("zero neural updates")))
        return stack

    def test_exact_fourteen_heads_two_calibrations_seal_and_reverify(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); output, tracked = self.fixture(root)
            before = {path: path.read_bytes() for path in (root/"inputs").rglob("*") if path.is_file()}
            solve, spectral = probe.api.project_regularized, np.linalg.svd
            with self.context(output, tracked), patch.object(probe.api, "project_regularized", wraps=solve) as calls, \
                 patch.object(np.linalg, "svd", wraps=spectral) as svds:
                result = probe.run()
                context = probe.verify_gradient_probe(output)
            self.assertEqual(calls.call_count, 14)
            self.assertEqual(svds.call_count, 2)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["method_checks_passed"])
            self.assertTrue(context["method_checks_passed"])
            self.assertEqual(result["counters"], {"calibrations": 2, "head_solves": 14})
            self.assertEqual(len((output/"progress.jsonl").read_text().splitlines()), 14)
            self.assertEqual(tracked.read_bytes(), (output/"result.json").read_bytes())
            for path, value in before.items():
                self.assertEqual(path.read_bytes(), value)

    def test_resource_failure_retains_current_head_and_every_unrun_opportunity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); output, tracked = self.fixture(root)
            actual, count = probe.api.project_regularized, 0
            def stopped(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 3:
                    raise probe.common.ResourceLimit("constructed head resource cap")
                return actual(*args, **kwargs)
            with self.context(output, tracked), patch.object(probe.api, "project_regularized", side_effect=stopped):
                result = probe.run()
                context = probe.verify_gradient_probe(output)
            self.assertEqual(count, 3)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["method_checks_passed"])
            self.assertFalse(context["method_checks_passed"])
            self.assertEqual(result["counters"], {"calibrations": 1, "head_solves": 3})
            self.assertEqual([row["status"] for row in result["cases"]], ["failed", "not_run"])
            samples = [sample for row in result["cases"] for sample in row["samples"]]
            self.assertEqual(len(samples), 14)
            self.assertEqual([sample["status"] for sample in samples[:4]], ["solved", "solved", "failed", "not_run"])
            self.assertEqual(sum(sample["status"] == "not_run" for sample in samples), 11)
            self.assertTrue(result["cases"][0]["original_state_restored_exactly"])
            self.assertEqual([json.loads(line) for line in (output/"progress.jsonl").read_text().splitlines()], samples[:3])

    def test_calibration_failure_retains_claimed_attempt_without_claiming_any_head_solve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); output, tracked = self.fixture(root)
            with self.context(output, tracked), patch.object(probe.api, "calibrate_readout", side_effect=FloatingPointError("constructed calibration failure")):
                result = probe.run()
                context = probe.verify_gradient_probe(output)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(context["method_checks_passed"])
            self.assertEqual(result["counters"], {"calibrations": 1, "head_solves": 0})
            self.assertEqual([row["status"] for row in result["cases"]], ["failed", "not_run"])
            attempts = result["calibration_attempts"]
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["case"], "t02")
            self.assertEqual(attempts[0]["status"], "failed")
            self.assertEqual(attempts[0]["attempt"], 1)
            self.assertTrue(all(sample["status"] == "not_run" for row in result["cases"] for sample in row["samples"]))


if __name__ == "__main__":
    unittest.main()
