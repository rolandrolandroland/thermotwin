"""Constructed saved-only report and strict completion/QA proof checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thermotwin.reports import pinn_spatial_regularized as report
from tests import test_pinn_spatial_step_control_report as old_fixture

FINALIZER_PATH = Path(__file__).resolve().parents[1]/"tools/pinn_spatial/finalize-regularized-result.py"
SPEC = importlib.util.spec_from_file_location("regularized_result_finalizer_tests", FINALIZER_PATH)
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


class RegularizedReportTests(unittest.TestCase):
    def setUp(self):
        self.old = old_fixture.StepControlReportTests(); self.old.setUp()
        self.helper = self.old.helper

    def fixture(self, root):
        previous = self.old.fixture(root)
        directory, validation, cutoff = root/"regularized", root/"regularized-validation", root/"cutoff"
        directory.mkdir(); validation.mkdir(); cutoff.mkdir()
        self.helper.write(cutoff/"plan.json", {"protected_trees": {str(root/"probe"): {}}})
        prior_plan = report._read(previous/"plan.json")
        context = self.helper.validate_graded(previous)
        plan = {"cases": prior_plan["cases"], "model_config": {"adam_epochs": 320},
            "source_sha256": {"model.py": report.digest_file(root/"model.py")},
            "preserved_step_control": {"directory": str(previous), "evidence": context},
            "preserved_cutoff_audit": {"directory": str(cutoff)},
            "method_checks": {"method_checks_passed": True, "probe_context": {"directory": str(root/"probe")}},
            "topology": {"maximum_concurrent_workers": 2, "threads_per_worker": 1}}
        self.helper.write(directory/"plan.json", plan)
        result = deepcopy(report._read(previous/"result.json"))
        result.update(status="completed", forward_status="blocked_forward_gate", initial_method_checks_passed=True,
            plan_sha256=report.digest_file(directory/"plan.json"), calibration_accounting={"preparation_matrix_builds": 4,
                "preparation_spectral_calculations": 4, "worker_matrix_builds": 0, "worker_spectral_calculations": 0})
        for case in plan["cases"][:2]:
            raw = report._read(previous/case["id"]/"training-report.json")
            raw.update(final_objective=2., final_unpenalized_objective=1.5, final_regularization_penalty=.5,
                final_head_stationarity=1.e-12, initial_calibration_matrix_builds=0,
                initial_calibration_spectral_calculations=0, hard_singular_value_truncation=False,
                readout_calibration={"sha256": "fixed-calibration", "fixed_lambda": 2.e-7, "column_scale_sha256": "frozen-scale"})
            self.helper.write(directory/case["id"]/"training-report.json", raw)
        self.helper.write(directory/"result.json", result)
        self.helper.write(directory/"execution.json", {"status": "completed"})
        self.helper.seal(directory)
        context = self.helper.validate_graded(directory)
        self.helper.write(validation/"execution-state.json", {"status": "completed", "evidence": context,
            "plan_sha256": report.digest_file(directory/"plan.json")})
        return directory, validation

    def context(self):
        return patch.multiple(report, _validate_regularized=self.helper.validate_graded,
            _validate_step_control=self.helper.validate_graded, _validate_graded=self.helper.validate_graded)

    @staticmethod
    def plots(output, summary):
        for name in ("comparison.png", "comparison.pdf"):
            (output/name).write_bytes(b"constructed figure")
        return ["comparison.png", "comparison.pdf"]

    def qa(self, root, validation):
        repository = Path(__file__).resolve().parents[1]
        for name in finalizer.REQUIRED_QA_SOURCES:
            path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((repository/name).read_bytes())
        sources = {name: finalizer.digest(root/name) for name in sorted(finalizer.REQUIRED_QA_SOURCES)}
        if (root/"model.py").is_file():
            sources["model.py"] = finalizer.digest(root/"model.py")
        log = validation/"full-suite.log"; log.write_text("Ran 5 tests in 1.0s\n\nOK\n")
        state = {"status": "completed", "passed": True, "return_code": 0, "tests_run": 5,
            "discovered_test_count": 5, "expected_test_count": 5, "failures": 0, "errors": 0,
            "expected_failures": 0, "unexpected_successes": 0, "skipped": 0,
            "source_unchanged": True, "source_sha256": sources,
            "source_manifest_sha256": finalizer.canonical(sources), "log_sha256": finalizer.digest(log)}
        if (root/"regularized/plan.json").is_file():
            state["campaign_plan_sha256"] = finalizer.digest(root/"regularized/plan.json")
        self.helper.write(validation/"full-suite-state.json", state)
        return state

    def test_all_twelve_saved_rows_and_separate_objectives_and_gates_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); directory, validation = self.fixture(root)
            with self.context():
                summary = report.load_comparison(directory, validation_directory=validation)
            self.assertEqual(summary["denominator"]["planned_case_attempts"], 12)
            self.assertEqual(summary["denominator"]["new_datasets"], 0)
            self.assertTrue(summary["initial_method_checks_passed"])
            self.assertEqual(summary["saved_forward_status"], "blocked_forward_gate")
            expected = report._read(directory/"result.json")["cases"]
            self.assertEqual([row["methods"]["regularized"]["saved"] for row in summary["rows"]], expected)
            self.assertTrue(expected[0]["gate"]["all"])
            self.assertEqual(expected[0]["independent_residuals"]["normalized_rms"]["interior"], 9.)
            self.assertEqual([row["methods"]["regularized"]["saved"]["status"] for row in summary["rows"]], ["completed", "completed", "failed", "not_run"])
            training = summary["rows"][0]["methods"]["regularized"]["training"]
            self.assertEqual((training["final_objective"], training["final_unpenalized_objective"], training["final_regularization_penalty"]), (2., 1.5, .5))
            text = report._markdown(summary)
            self.assertIn("must not be ranked directly", text)
            self.assertIn("| completed | pass | 9 |", text)
            self.assertIn("not convergence certificates", text)
            json.dumps(summary, allow_nan=False)

    def test_unfinished_supervisor_and_missing_corrupt_or_omitted_evidence_fail_closed(self):
        for mutation in ("running", "missing", "corrupt", "omitted"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve(); directory, validation = self.fixture(root)
                if mutation == "running": self.helper.write(validation/"execution-state.json", {"status": "running"})
                elif mutation == "missing": (directory/"t01/training-report.json").unlink()
                elif mutation == "corrupt": (directory/"t01/training-report.json").write_text("changed")
                else:
                    result = report._read(directory/"result.json"); result["cases"].pop()
                    self.helper.write(directory/"result.json", result); self.helper.seal(directory)
                    self.helper.write(validation/"execution-state.json", {"status": "completed", "evidence": self.helper.validate_graded(directory),
                        "plan_sha256": report.digest_file(directory/"plan.json")})
                with self.context(), self.assertRaises(ValueError):
                    report.load_comparison(directory, validation_directory=validation)

    def test_report_is_separate_four_file_seal_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); directory, validation = self.fixture(root)
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with self.context(), patch.object(report, "_plots", self.plots):
                result = report.render_report(directory, output_directory=root/"report", validation_directory=validation)
                self.assertEqual(result["files"], 4)
                for protected in (directory/"nested", root/"graded/nested", root/"controlled/nested", root/"probe/nested", validation/"nested"):
                    with self.assertRaisesRegex(ValueError, "separate"):
                        report.render_report(directory, output_directory=protected, validation_directory=validation)
            index = finalizer.sealed_report(root/"report")
            self.assertEqual(len(index["files"]), 4)
            for path, value in before.items():
                self.assertEqual(path.read_bytes(), value)

    def test_finalizer_requires_canonical_counts_hashes_current_sources_and_terminal_log(self):
        for mutation in (None, "legacy", "not_boolean", "counts", "log", "manifest", "source", "skipped"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve(); validation = root/"validation"; validation.mkdir()
                state = self.qa(root, validation)
                if mutation == "legacy": state["status"] = "passed"
                elif mutation == "not_boolean": state["passed"] = 1
                elif mutation == "counts": state["tests_run"] = 4
                elif mutation == "log": (validation/"full-suite.log").write_text("Ran 5 tests in 1.0s\n\nOK\nchanged\n")
                elif mutation == "manifest": state["source_manifest_sha256"] = "wrong"
                elif mutation == "source": (root/next(iter(state["source_sha256"]))).write_text("changed")
                elif mutation == "skipped": state["skipped"] = 1
                self.helper.write(validation/"full-suite-state.json", state)
                if mutation is None:
                    self.assertTrue(finalizer.qa_evidence(validation, root=root)["passed"])
                else:
                    with self.assertRaises(RuntimeError):
                        finalizer.qa_evidence(validation, root=root)

    def test_constructed_finalization_copies_saved_rows_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); directory, validation = self.fixture(root)
            self.qa(root, validation)
            tracked_plan = root/"tracked-plan.json"; tracked_plan.write_bytes((directory/"plan.json").read_bytes())
            with self.context(), patch.object(report, "_plots", self.plots):
                report.render_report(directory, output_directory=root/"report", validation_directory=validation)
            with self.context(), patch.multiple(finalizer, ROOT=root, CAMPAIGN=directory, VALIDATION=validation,
                REPORT=root/"report", FINALIZATION=root/"finalization", TRACKED_PLAN=tracked_plan,
                TRACKED_RESULT=root/"tracked-result.json", __file__=str(root/"tools/pinn_spatial/finalize-regularized-result.py")):
                result = finalizer.finalize()
                self.assertEqual(result["status"], "completed")
                saved = finalizer.read(root/"tracked-result.json")
                self.assertEqual(saved["rows"][0]["methods"]["regularized"]["saved"], report._read(directory/"result.json")["cases"][0])
                self.assertEqual(saved["new_fits_or_gate_calculations"], 0)
                self.assertEqual((root/"tracked-result.json").read_bytes(), (root/"finalization/result.json").read_bytes())
                with self.assertRaisesRegex(RuntimeError, "no overwrite"):
                    finalizer.finalize()


if __name__ == "__main__":
    unittest.main()
