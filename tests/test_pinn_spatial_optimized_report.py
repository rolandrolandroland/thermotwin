"""Constructed saved-evidence report checks; no fits or scientific cohorts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thermotwin.reports import pinn_spatial_optimized as report
from thermotwin.reports.pinn_spatial_continuity import validate_campaign


class OptimizedReportTests(unittest.TestCase):
    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False))

    def seal(self, directory):
        index = {str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": report.digest_file(path)}
            for path in sorted(directory.rglob("*")) if path.is_file() and path not in (directory/"state.json", directory/"artifact-index.json")}
        self.write(directory/"artifact-index.json", index)
        self.write(directory/"state.json", {"status": "completed", "plan_sha256": report.digest_file(directory/"plan.json"),
            "artifact_index_sha256": report.digest_file(directory/"artifact-index.json")})

    def validate_graded(self, directory):
        directory = Path(directory)
        state = report._read(directory/"state.json")
        index = report._read(directory/"artifact-index.json")
        if state["artifact_index_sha256"] != report.digest_file(directory/"artifact-index.json"):
            raise ValueError("constructed graded index changed")
        actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()
                  and path not in (directory/"state.json", directory/"artifact-index.json")}
        if actual != set(index):
            raise ValueError("constructed graded file set changed")
        for name, saved in index.items():
            if report.digest_file(directory/name) != saved["sha256"] or (directory/name).stat().st_size != saved["bytes"]:
                raise ValueError("constructed graded artifact changed")
        return {"artifact_index_sha256": report.digest_file(directory/"artifact-index.json"),
                "plan_sha256": report.digest_file(directory/"plan.json"), "files": len(index)}

    @staticmethod
    def raw(optimized=False):
        updates = [{"step": 1, "objective_before": 3., "objective_after": 2., "accepted": True,
            "backtracks": 0, "optimizer_step_counts_before": [], "optimizer_step_counts_proposed": [1],
            "optimizer_step_counts_after": [1], "optimizer_state_committed": True,
            "optimizer_state_rolled_back": False, "trials": [{"accepted": True, "objective": 2., "rank": 26,
                "resolved_condition": 123., "rank_rcond": 1e-7, "projection": {"rank": 26, "coefficient_norm_returned": 8.}}]}]
        if optimized:
            updates.append({"step": 2, "objective_before": 2., "objective_after": 2., "accepted": False,
                "backtracks": 4, "optimizer_step_counts_before": [1], "optimizer_step_counts_proposed": [2],
                "optimizer_step_counts_after": [1], "optimizer_state_committed": False,
                "optimizer_state_rolled_back": True, "trials": []})
        return {"wall_seconds": 20., "final_objective": 2., "coefficient_count": 30,
            "training_family_mse": {"interior": .01, "bottom_contact": .02, "bottom_insulated": .03},
            "projections": [{"objective_after": 3., "rank": 26, "columns": 30, "resolved_condition": 123.,
                             "coefficient_norm_returned": 8.}], "hidden_update_history": updates,
            "hidden_rejected_updates": int(optimized), "hidden_backtrack_count": 4 if optimized else 0,
            "hidden_accepted_objective_increases": 0, "optimizer_resets": 0 if optimized else 81,
            "optimizer_history": "persistent" if optimized else None, "optimizer_instances": 1 if optimized else None,
            "optimizer_state_rollbacks": 1 if optimized else None, "final_optimizer_step_counts": [1] if optimized else None,
            "graded_point_metadata": {"point_digest": "same-grid", "strata": [{"name": "bulk", "mean_weight": .5}]},
            "initialization": {"common_weights_sha256": "same-original-weights"}}

    def fixture(self, root):
        directory, old, probe = root/"optimized", root/"graded", root/"probe"
        directory.mkdir(); old.mkdir(); probe.mkdir()
        source = root/"model.py"; source.write_text("# constructed source\n")
        sources = {"model.py": report.digest_file(source)}
        cases = [{"id": case, "geometry": geometry, "known_h": h} for case, geometry, h in
            (("t01", "full_contact", 800.), ("t02", "partial_contact", 800.),
             ("t03", "full_contact", 1400.), ("t04", "partial_contact", 1400.))]
        baseline, current = [], []
        for index, case in enumerate(cases):
            residuals = {"interior": 9., "bottom_contact": .2, "top": 0., "left": 0., "right": 0.}
            if case["geometry"] == "partial_contact":
                residuals.update(bottom_insulated=.3, interface_0_temperature=0., interface_0_flux=0.,
                                 interface_1_temperature=0., interface_1_flux=0.)
            row = dict(case, status="completed", gate={"all": True, "observations": True, "field": True, "each_independent_residual": True},
                independent_residuals={"normalized_rms": residuals}, field_metrics={"rmse_K": .01, "maximum_error_K": .03}, normalized_observation_mse=.8)
            baseline.append(row)
            self.write(old/"log_off"/case["id"]/"training-report.json", self.raw())
            if index < 2:
                current.append(deepcopy(row))
                self.write(directory/case["id"]/"training-report.json", self.raw(True))
            else:
                current.append(dict(case, status="failed" if index == 2 else "not_run", reason="constructed numerical or resource stop"))
            for base in (old/"log_off", directory):
                if base == directory and index >= 2:
                    continue
                self.write(base/case["id"]/"near-interface-audit.json", {"status": "not_applicable", "role": "report only"})
                self.write(base/case["id"]/"near-bottom-audit.json", {"status": "completed", "role": "report only",
                    "diagnostics": {"normalized_rms": {"interior": 75.}}, "by_height": [{"z_over_H": .002,
                        "point_count": 192, "diagnostics": {"normalized_rms": {"interior": 75.}, "physical_rms": {"interior": 3.125}}}]})
        self.write(old/"plan.json", {"cases": cases, "source_sha256": sources, "model_configs": {"log_off": {"adam_epochs": 80}}})
        self.write(old/"log_off"/"final-forward-selection.json", {"cases": baseline, "status": "blocked_forward_gate", "wall_seconds": 80., "peak_rss_bytes": 1000})
        self.write(old/"result.json", {"paired_case_rows": [dict(case, methods={"log_off": row}) for case, row in zip(cases, baseline)]})
        self.seal(old)
        self.write(probe/"plan.json", {"training_only": True})
        self.write(probe/"state.json", {"status": "completed"})
        self.write(probe/"result.json", {"status": "completed", "wall_seconds": 97., "peak_rss_bytes": 1024})
        decision = {"status": "approved_for_forward_development", "training_only": True,
            "reference_fields_used": False, "independent_diagnostics_used_for_selection": False,
            "probe_evidence": [{"path": "probe/result.json", "sha256": report.digest_file(probe/"result.json")} ]}
        self.write(probe/"decision.json", decision)
        plan = {"cases": cases, "source_sha256": sources,
            "model_config": {"adam_epochs": 320, "learning_rate": .0064, "optimizer_history": "persistent"},
            "optimization_provenance": {"graded_comparison": {"directory": str(old), "context": self.validate_graded(old),
                "plan_sha256": report.digest_file(old/"plan.json")},
                "decision": {"path": "probe/decision.json", "sha256": report.digest_file(probe/"decision.json"), "decision": decision},
                "initialization_policy": "fresh original weights; no checkpoint continuation",
                "selection_boundary": "training-only objectives; no independent diagnostics"}}
        self.write(directory/"plan.json", plan)
        self.write(directory/"final-forward-selection.json", {"cases": current, "source_sha256": sources,
            "plan_sha256": report.digest_file(directory/"plan.json"), "status": "blocked_forward_gate", "source_unchanged": True,
            "wall_seconds": 320., "peak_rss_bytes": 2048, "inverse_opened": False, "fresh_evaluation_opened": False})
        self.seal(directory)
        return directory

    def context(self, root):
        return patch.multiple(report, ROOT=root, _validate_optimized=validate_campaign, _validate_graded=self.validate_graded)

    @staticmethod
    def plots(output, comparison):
        names = [base+extension for base in ("comparison", "trajectories", "optimizer-history") for extension in (".png", ".pdf")]
        for name in names:
            (output/name).write_bytes(b"constructed plot")
        return names

    def test_retains_all_four_rows_and_saved_gates_without_recalculation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); directory = self.fixture(root)
            with self.context(root):
                comparison = report.load_comparison(directory)
            self.assertEqual(comparison["denominator"]["declared_cases_per_method"], 4)
            self.assertEqual(comparison["denominator"]["method_fits"], 8)
            self.assertEqual(len(comparison["rows"]), 4)
            markdown = report._markdown(comparison)
            self.assertIn("| completed | pass | 9 |", markdown)
            self.assertIn("| failed | not recorded | not recorded |", markdown)
            self.assertIn("| not_run | not recorded | not recorded |", markdown)
            self.assertIn("| t01 | optimized | 0.002 | 192 | 75 | 3.125 |", markdown)
            self.assertIn("not independent replications", markdown)
            self.assertIn("Budget endpoints are not convergence certificates", markdown)
            train = comparison["rows"][0]["methods"]["optimized"]["training"]
            self.assertEqual(train["trajectory"][1]["optimizer_step_counts_after"], [1])
            self.assertTrue(train["trajectory"][1]["optimizer_state_rolled_back"])
            self.assertEqual(train["final_projection"]["resolved_condition"], 123.)
            self.assertEqual(comparison["probe_evidence"][0]["wall_seconds"], 97.)
            json.dumps(comparison, allow_nan=False)

    def test_render_is_separate_indexed_and_does_not_modify_bound_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); directory = self.fixture(root)
            evidence = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with self.context(root), patch.object(report, "_plots", self.plots):
                result = report.render_report(directory, output_directory=root/"report")
                self.assertEqual(result["files"], 8)
                for output in (directory/"bad", root/"graded"/"bad", root/"probe"/"bad"):
                    with self.assertRaisesRegex(ValueError, "separate"):
                        report.render_report(directory, output_directory=output)
            index = report._read(root/"report"/"report-index.json")
            self.assertEqual(index["digest"], report.canonical_digest(index["files"]))
            for row in index["files"]:
                self.assertEqual(report.digest_file(root/"report"/row["path"]), row["sha256"])
            for path, value in evidence.items():
                self.assertEqual(path.read_bytes(), value)

    def test_running_missing_changed_extra_and_source_corruption_fail_closed(self):
        for mutation in ("running", "old_running", "missing", "changed", "extra", "source", "probe"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); directory = self.fixture(root)
                target = directory/"t01"/"training-report.json"
                if mutation == "running": self.write(directory/"state.json", {"status": "running"})
                elif mutation == "old_running": self.write(root/"graded"/"state.json", {"status": "running"})
                elif mutation == "missing": target.unlink()
                elif mutation == "changed": target.write_text("changed")
                elif mutation == "extra": (directory/"extra.json").write_text("{}")
                elif mutation == "source": (root/"model.py").write_text("changed")
                else: self.write(root/"probe"/"result.json", {"wall_seconds": 1.})
                with self.context(root), self.assertRaises(ValueError):
                    report.load_comparison(directory)

    def test_resealed_duplicate_omitted_or_substituted_rows_are_rejected(self):
        for mutation in ("duplicate", "omitted", "physical_case", "baseline_substitution"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); directory = self.fixture(root)
                path = directory/"final-forward-selection.json"; saved = report._read(path)
                if mutation == "duplicate": saved["cases"].append(deepcopy(saved["cases"][0]))
                elif mutation == "omitted": saved["cases"].pop()
                elif mutation == "physical_case": saved["cases"][0]["known_h"] = 900.
                else:
                    old = root/"graded"/"log_off"/"final-forward-selection.json"; value = report._read(old)
                    value["cases"][0]["gate"]["all"] = False; self.write(old, value); self.seal(root/"graded")
                    plan = report._read(directory/"plan.json")
                    plan["optimization_provenance"]["graded_comparison"]["context"] = self.validate_graded(root/"graded")
                    self.write(directory/"plan.json", plan); saved["plan_sha256"] = report.digest_file(directory/"plan.json")
                self.write(path, saved); self.seal(directory)
                with self.context(root), self.assertRaises(ValueError):
                    report.load_comparison(directory)


if __name__ == "__main__":
    unittest.main()
