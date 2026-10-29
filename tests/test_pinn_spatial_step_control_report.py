"""Saved-only report checks with failed attempts and bounded training stops."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thermotwin.reports import pinn_spatial_step_control as report
from thermotwin.reports.pinn_spatial_continuity import validate_campaign
from tests import test_pinn_spatial_optimized_report as fixture_module


class StepControlReportTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixture_module.OptimizedReportTests()

    def write(self, path, value):
        self.helper.write(path, value)

    def fixture(self, root):
        prior = self.helper.fixture(root)
        progress = {"stage": "transactional_profiled_adam", "step": 223, "accepted": False,
            "objective_before": .8, "objective_after": .8, "backtracks": 4,
            "gradient_norm_before_clip": .1, "optimizer_step_counts_after": [124]}
        path = prior/"t03"/"progress.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(progress)+"\n{incomplete")
        self.helper.seal(prior)
        directory = root/"controlled"; directory.mkdir()
        prior_plan = report._read(prior/"plan.json")
        cases = prior_plan["cases"]
        old = root/"graded"
        plan = {"cases": cases, "source_sha256": prior_plan["source_sha256"],
            "model_config": {"adam_epochs": 320, "learning_rate": .0064, "maximum_trial_solves": 480},
            "step_control_provenance": {
                "graded_comparison": {"directory": str(old), "context": self.helper.validate_graded(old),
                    "plan_sha256": report.digest_file(old/"plan.json")},
                "prior_optimized_campaign": {"directory": str(prior), "context": validate_campaign(prior),
                    "plan_sha256": report.digest_file(prior/"plan.json")}},
            "decision": prior_plan["optimization_provenance"]["decision"],
            "initialization_policy": "fresh original weights, no continuation",
            "selection_boundary": "training-only fixed controller rule",
            "topology": {"workers": 4, "concurrent_workers": 2, "case_wall_seconds": 3000., "threads": 1}}
        self.write(directory/"plan.json", plan)
        results = deepcopy(report._read(prior/"final-forward-selection.json"))
        results["plan_sha256"] = report.digest_file(directory/"plan.json")
        results["wall_seconds"] = 300.
        for case in cases[:2]:
            raw = self.helper.raw(True)
            state = {"learning_rate": .0064, "attempts": 0, "consecutive_failures": 0,
                "full_scale_successes": 0, "moment_restarts": 0, "trial_solves": 0,
                "stopped": False, "stop_reason": None}
            controllers = []
            for index, event in enumerate(raw["hidden_update_history"]):
                before = deepcopy(state)
                state.update(attempts=index+1, trial_solves=1+5*index)
                if index:
                    state.update(learning_rate=.0032, moment_restarts=1, consecutive_failures=1)
                after = deepcopy(state)
                event.update(controller_before=before, controller_after=after,
                    controller_transitions=["reduce after exhausted rejection"] if index else [],
                    controller_moments_restarted=bool(index), next_learning_rate=state["learning_rate"])
                controllers.append({"step": index+1, "before": before, "after": after,
                                    "transitions": event["controller_transitions"]})
            state.update(stopped=True, stop_reason="projection budget")
            raw.update(controller_history=controllers, final_controller_state=state,
                completed_updates=2, requested_updates=320, stop_reason="projection_budget",
                termination="projection budget", trial_solve_count=6, maximum_trial_solves=480,
                additional_initial_final_solves=2, total_projection_solve_calls=8,
                controller_moment_restarts=1, policy_moment_resets=0)
            self.write(directory/case["id"]/"training-report.json", raw)
        self.write(directory/"result.json", results)
        self.write(directory/"execution.json", {"status": "completed", "wall_seconds": 300.})
        self.helper.seal(directory)
        return directory

    def context(self, root):
        return patch.multiple(report, ROOT=root, _validate_step_control=self.helper.validate_graded,
            _validate_prior=validate_campaign, _validate_graded=self.helper.validate_graded)

    @staticmethod
    def plots(output, comparison):
        names = [name+extension for name in ("comparison", "trajectories", "controller-history") for extension in (".png", ".pdf")]
        for name in names:
            (output/name).write_bytes(b"constructed plot")
        return names

    def test_saved_gate_photocopy_and_all_failed_attempt_rows_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); directory = self.fixture(root)
            with self.context(root):
                summary = report.load_comparison(directory)
            self.assertEqual(len(summary["rows"]), 4)
            self.assertEqual(summary["denominator"]["declared_cases_per_attempt"], 4)
            self.assertEqual(summary["denominator"]["planned_case_attempts"], 12)
            self.assertEqual(summary["denominator"]["prior_failed_attempt_cases_retained"], 4)
            source = report._read(directory/"result.json")["cases"]
            self.assertEqual([row["methods"]["controlled"]["saved"] for row in summary["rows"]], source)
            # Deliberately inconsistent saved gate proves this report does not re-gate.
            self.assertTrue(source[0]["gate"]["all"])
            self.assertEqual(source[0]["independent_residuals"]["normalized_rms"]["interior"], 9.)
            prior = [row["methods"]["prior_attempt"]["saved"]["status"] for row in summary["rows"]]
            self.assertEqual(prior, ["completed", "completed", "failed", "not_run"])
            progress = summary["rows"][2]["methods"]["prior_attempt"]["progress_provenance"]
            self.assertEqual(progress["last_update"]["step"], 223)
            self.assertTrue(progress["incomplete_json_line_present"])
            self.assertIsNone(summary["rows"][2]["methods"]["prior_attempt"]["training"]["hidden_update_count"])
            markdown = report._markdown(summary)
            self.assertIn("| completed | pass | 9 |", markdown)
            self.assertIn("1 saved partial progress events; no completed training report", markdown)
            self.assertIn("| failed | not recorded |", markdown)
            self.assertIn("| not_run | not recorded |", markdown)
            self.assertIn("not independent replications", markdown)
            json.dumps(summary, allow_nan=False)

    def test_early_stop_controller_and_endpoint_budget_are_copied_without_convergence_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); directory = self.fixture(root)
            with self.context(root):
                summary = report.load_comparison(directory)
            training = summary["rows"][0]["methods"]["controlled"]["training"]
            self.assertEqual(training["completed_updates"], 2)
            self.assertEqual(training["requested_updates"], 320)
            self.assertEqual(training["stop_reason"], "projection_budget")
            self.assertEqual(training["trial_solve_count"], 6)
            self.assertEqual(training["additional_initial_final_solves"], 2)
            self.assertFalse(training["controller_history"][-1]["after"]["stopped"])
            self.assertTrue(training["final_controller_state"]["stopped"])
            self.assertEqual(training["trajectory"][1]["next_learning_rate"], .0032)
            self.assertEqual(training["trajectory"][1]["optimizer_step_counts_after"], [1])
            markdown = report._markdown(summary)
            self.assertIn("No endpoint is labeled convergence", markdown)
            self.assertIn("| t01 | 2 / 320 | projection_budget | 6 / 480 | 2 | 1 | 0.0032 | 1 |", markdown)

    def test_separate_sealed_output_preserves_all_bound_evidence_and_protects_prior_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); directory = self.fixture(root)
            snapshot = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with self.context(root), patch.object(report, "_plots", self.plots):
                rendered = report.render_report(directory, output_directory=root/"report")
                self.assertEqual(rendered["files"], 8)
                for path in (directory/"bad", root/"optimized"/"bad", root/"graded"/"bad", root/"probe"/"bad", root):
                    with self.assertRaisesRegex(ValueError, "separate"):
                        report.render_report(directory, output_directory=path)
                with self.assertRaisesRegex(ValueError, "separate"):
                    report.render_report(directory, output_directory=root/"report"/"nested")
            index = report._read(root/"report"/"report-index.json")
            self.assertEqual(index["digest"], report.canonical_digest(index["files"]))
            self.assertIn("prior_attempt_evidence", index)
            for row in index["files"]:
                self.assertEqual(report.digest_file(root/"report"/row["path"]), row["sha256"])
            for path, value in snapshot.items():
                self.assertEqual(path.read_bytes(), value)

    def test_unfinished_corrupt_missing_and_extra_evidence_fail_closed(self):
        for mutation in ("running", "prior_running", "missing", "extra", "source", "probe", "prior_progress"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); directory = self.fixture(root)
                if mutation == "running": self.write(directory/"state.json", {"status": "running"})
                elif mutation == "prior_running": self.write(root/"optimized"/"state.json", {"status": "running"})
                elif mutation == "missing": (directory/"t01"/"training-report.json").unlink()
                elif mutation == "extra": self.write(directory/"unbound.json", {})
                elif mutation == "source": (root/"model.py").write_text("changed")
                elif mutation == "probe": self.write(root/"probe"/"result.json", {})
                else: (root/"optimized"/"t03"/"progress.jsonl").write_text("changed")
                with self.context(root), self.assertRaises(ValueError):
                    report.load_comparison(directory)

    def test_resealed_duplicate_missing_and_substituted_rows_are_rejected(self):
        for mutation in ("duplicate", "missing", "physical", "prior_omission"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); directory = self.fixture(root)
                path = directory/"result.json"; saved = report._read(path)
                if mutation == "duplicate": saved["cases"].append(deepcopy(saved["cases"][0]))
                elif mutation == "missing": saved["cases"].pop()
                elif mutation == "physical": saved["cases"][0]["known_h"] = 999.
                else:
                    prior = root/"optimized"; p = prior/"final-forward-selection.json"; value = report._read(p)
                    value["cases"].pop(); self.write(p, value); self.helper.seal(prior)
                    plan = report._read(directory/"plan.json")
                    plan["step_control_provenance"]["prior_optimized_campaign"]["context"] = validate_campaign(prior)
                    self.write(directory/"plan.json", plan)
                    saved["plan_sha256"] = report.digest_file(directory/"plan.json")
                self.write(path, saved); self.helper.seal(directory)
                with self.context(root), self.assertRaises(ValueError):
                    report.load_comparison(directory)


if __name__ == "__main__":
    unittest.main()
