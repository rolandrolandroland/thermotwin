"""Constructed saved-evidence reporting checks; no model fits or cohorts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thermotwin.reports import pinn_spatial_graded as report


class GradedReportTests(unittest.TestCase):
    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False))

    def seal(self, directory, *, top=False):
        records = {}
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path not in (directory / "state.json", directory / "artifact-index.json"):
                records[str(path.relative_to(directory))] = {
                    "bytes": path.stat().st_size, "sha256": report.digest_file(path),
                }
        self.write(directory / "artifact-index.json", records)
        self.write(directory / "state.json", {
            "status": "completed", "plan_sha256": report.digest_file(directory / "plan.json"),
            "artifact_index_sha256": report.digest_file(directory / "artifact-index.json"),
        })

    def coordinator_validator(self, directory, plan):
        # A small constructed counterpart of the separately tested numerical
        # coordinator validator: cover every top artifact and its terminal bind.
        index = json.loads((directory / "artifact-index.json").read_text())
        state = json.loads((directory / "state.json").read_text())
        if state["artifact_index_sha256"] != report.digest_file(directory / "artifact-index.json"):
            raise ValueError("constructed top index changed")
        actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
                  if path.is_file() and path not in (directory / "state.json", directory / "artifact-index.json")}
        if actual != set(index):
            raise ValueError("constructed top file set changed")
        for name, record in index.items():
            path = directory / name
            if report.digest_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise ValueError("constructed top artifact changed")
        return {"artifact_index_sha256": report.digest_file(directory / "artifact-index.json"),
                "plan_sha256": report.digest_file(directory / "plan.json"), "files": len(index)}

    def fixture(self, root):
        directory = root / "comparison"
        directory.mkdir()
        source = root / "model.py"
        source.write_text("# constructed source only\n")
        source_records = {"model.py": report.digest_file(source)}
        protected = root / "old-evidence"
        protected.mkdir()
        (protected / "preserved.txt").write_text("unchanged")
        cases = [{"id": key, "geometry": geometry, "known_h": conductance}
                 for key, geometry, conductance in (
                     ("t01", "full_contact", 800.), ("t02", "partial_contact", 800.),
                     ("t03", "full_contact", 1400.), ("t04", "partial_contact", 1400.))]
        top = {"schema": "paired_graded_log_ablation_v5", "source_sha256": source_records,
               "cases": cases, "subplans": {}, "protected_campaigns": [{"directory": str(protected)}]}
        result = {"status": "completed", "arms": {}, "paired_case_rows": [], "no_winner_selected": True}
        saved = {}
        for arm in report.ARMS:
            sub = directory / arm
            sub.mkdir()
            self.write(sub / "plan.json", {"cases": cases, "source_sha256": source_records,
                "model_config": {"adam_epochs": 80, "bottom_log_feature": arm == "log_on"}})
            rows = []
            for index, case in enumerate(cases):
                if index >= 2:
                    rows.append(dict(case, status="failed" if index == 2 else "not_run",
                                     reason="constructed numerical failure" if index == 2 else "constructed resource stop"))
                    continue
                normalized = {"interior": 9., "bottom_contact": .1, "top": 0., "left": 0., "right": 0.}
                if case["geometry"] == "partial_contact":
                    normalized.update(bottom_insulated=.3, interface_0_temperature=0.,
                                      interface_0_flux=1e-15, interface_1_temperature=0., interface_1_flux=1e-15)
                row = dict(case, status="completed", gate={"all": True},
                           field_metrics={"rmse_K": .01, "maximum_error_K": .03},
                           normalized_observation_mse=.8,
                           independent_residuals={"normalized_rms": normalized,
                               "physical_rms": {name: value * (.1 if name.endswith("temperature") else
                                                               1 / 24 if name == "interior" else 500)
                                                for name, value in normalized.items()}})
                rows.append(row)
                self.write(sub / case["id"] / "training-report.json", {
                    "wall_seconds": 100., "final_objective": 2., "coefficient_count": 30,
                    "projections": [{"rank": 26, "columns": 30, "resolved_condition": 123.,
                                     "coefficient_norm_returned": 2., "accepted": True}],
                    "hidden_update_history": [{"step": 1, "objective_before": 3., "objective_after": 2.,
                                                "accepted": True, "backtracks": 0,
                                                "trials": [{"accepted": True, "objective": 2., "rank": 26,
                                                    "resolved_condition": 123., "rank_rcond": 1e-7,
                                                    "projection": {"coefficient_norm_returned": 2.,
                                                                   "objective_after": 2.}}]}],
                    "hidden_rejected_updates": 0, "hidden_backtrack_count": 0,
                    "optimizer_resets": 2, "hidden_accepted_objective_increases": 0,
                    "graded_point_metadata": {"point_digest": "same-grid", "point_counts": {"interior": 8192},
                        "strata": [{"name": "bulk", "count": 4096, "mean_weight": .5}],
                        "objective_weights": {"interior": "uniform mean"},
                        "point_identities": {"frozen": True}},
                    "initialization": {"shared_function_sha256": "same-function", "arm": arm},
                })
                self.write(sub / case["id"] / "near-interface-audit.json",
                    {"status": "not_applicable", "role": "report only"} if index == 0 else {
                        "status": "completed", "role": "report only",
                        "diagnostics": {"normalized_rms": {"interior": 7.}}})
                self.write(sub / case["id"] / "near-bottom-audit.json", {
                    "status": "completed", "role": "report only",
                    "diagnostics": {"normalized_rms": {"interior": 75.}},
                    "by_height": [{"z_over_H": .002, "point_count": 192,
                        "diagnostics": {"normalized_rms": {"interior": 75.},
                                        "physical_rms": {"interior": 3.125}}}],
                })
            self.write(sub / "final-forward-selection.json", {
                "cases": rows, "status": "blocked_forward_gate", "source_unchanged": True,
                "source_sha256": source_records, "plan_sha256": report.digest_file(sub / "plan.json"),
                "wall_seconds": 400., "peak_rss_bytes": 123456,
                "inverse_opened": False, "fresh_evaluation_opened": False,
            })
            self.seal(sub)
            top["subplans"][arm] = {"relative_directory": arm,
                                    "plan_sha256": report.digest_file(sub / "plan.json")}
            result["arms"][arm] = {"artifact_index_sha256": report.digest_file(sub / "artifact-index.json")}
            saved[arm] = rows
        for index, case in enumerate(cases):
            result["paired_case_rows"].append(dict(case, methods={arm: saved[arm][index] for arm in report.ARMS},
                pairing={"predeclared_matched_initialization": True,
                         "predeclared_identical_graded_grid": True,
                         "executed_proofs": {arm: {"status": "verified" if index < 2 else "not_recorded_after_failure"}
                                             for arm in report.ARMS}}))
        self.write(directory / "plan.json", top)
        self.write(directory / "result.json", result)
        self.seal(directory, top=True)
        return directory

    def context(self, root):
        return patch.multiple(report, ROOT=root, _coordinator_validation=self.coordinator_validator)

    @staticmethod
    def plots(output, comparison):
        names = ["comparison.png", "comparison.pdf", "trajectories.png", "trajectories.pdf"]
        for name in names:
            (output / name).write_bytes(b"constructed plot")
        return names

    def test_all_eight_saved_rows_and_gates_are_retained_without_recalculation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.fixture(root)
            with self.context(root):
                comparison = report.load_comparison(directory)
            self.assertEqual(comparison["denominator"]["method_fits"], 8)
            self.assertEqual(len(comparison["rows"]), 4)
            self.assertTrue(comparison["rows"][0]["methods"]["log_on"]["saved"]["gate"]["all"])
            markdown = report._markdown(comparison)
            self.assertIn("| log_on | completed | pass | 9 |", markdown)
            self.assertIn("| log_off | failed | not recorded | not recorded |", markdown)
            self.assertIn("| log_on | not_run | not recorded | not recorded |", markdown)
            self.assertIn("| t01 | log_on | 0.002 | 192 | 75 | 3.125 |", markdown)
            self.assertIn("not certificates of convergence", markdown)
            self.assertIn("not a controlled sampling-only comparison", markdown)
            self.assertEqual(comparison["rows"][0]["methods"]["log_on"]["training"]["trajectory"][0]["objective_after"], 2.)
            training = comparison["rows"][0]["methods"]["log_on"]["training"]
            self.assertEqual(training["training_point_digest"], "same-grid")
            self.assertEqual(training["training_strata"][0]["mean_weight"], .5)
            self.assertEqual(training["graded_point_metadata"]["objective_weights"], {"interior": "uniform mean"})
            self.assertEqual(training["trajectory"][0]["trials"][0]["rank"], 26)
            self.assertEqual(training["trajectory"][0]["trials"][0]["projection"]["coefficient_norm_returned"], 2.)
            json.dumps(comparison, allow_nan=False)

    def test_render_is_separate_complete_and_does_not_touch_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.fixture(root)
            original = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
            with self.context(root), patch.object(report, "_plots", self.plots):
                result = report.render_report(directory, output_directory=root / "report")
                self.assertEqual(result["files"], 6)
                with self.assertRaisesRegex(ValueError, "separate"):
                    report.render_report(directory, output_directory=directory / "bad")
                with self.assertRaisesRegex(ValueError, "separate"):
                    report.render_report(directory, output_directory=root / "old-evidence" / "bad")
            index = json.loads((root / "report" / "report-index.json").read_text())
            self.assertEqual(index["digest"], report.canonical_digest(index["files"]))
            for record in index["files"]:
                self.assertEqual(record["sha256"], report.digest_file(root / "report" / record["path"]))
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content)

    def test_running_changed_missing_extra_and_source_corruption_are_rejected(self):
        for mutation in ("running", "changed", "missing", "extra", "source"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory = self.fixture(root)
                target = directory / "log_on" / "t01" / "training-report.json"
                if mutation == "running":
                    self.write(directory / "state.json", {"status": "running"})
                elif mutation == "changed":
                    target.write_text("changed")
                elif mutation == "missing":
                    target.unlink()
                elif mutation == "extra":
                    (directory / "unindexed.json").write_text("{}")
                else:
                    (root / "model.py").write_text("changed")
                with self.context(root), self.assertRaises(ValueError):
                    report.load_comparison(directory)

    def test_resealed_duplicate_or_substituted_saved_row_is_rejected(self):
        for mutation in ("duplicate", "saved_row", "index_bind", "plan_bind"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory = self.fixture(root)
                result = report._read(directory / "result.json")
                if mutation == "duplicate":
                    result["paired_case_rows"].append(deepcopy(result["paired_case_rows"][0]))
                elif mutation == "saved_row":
                    result["paired_case_rows"][0]["methods"]["log_off"]["gate"]["all"] = False
                elif mutation == "index_bind":
                    result["arms"]["log_on"]["artifact_index_sha256"] = "substituted"
                else:
                    plan = report._read(directory / "plan.json")
                    plan["subplans"]["log_on"]["plan_sha256"] = "substituted"
                    self.write(directory / "plan.json", plan)
                self.write(directory / "result.json", result)
                self.seal(directory, top=True)
                with self.context(root), self.assertRaises(ValueError):
                    report.load_comparison(directory)


if __name__ == "__main__":
    unittest.main()
