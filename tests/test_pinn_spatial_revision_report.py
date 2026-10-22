"""Mocked evidence rendering and CLI routing; no scientific cohorts are run."""

import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from thermotwin.reports.pinn_spatial_revision import main,render_revision_report


def _result():
    return {"status":"completed_with_failures","source_commit":"abc123","protocol_digest":"protocol-digest",
        "selection_summary":{"architecture":"hard_boundary_decomposed","criterion":"provided by coordinator"},
        "planned_denominators":{"two_d":6,"one_d":6},"numerical_reference":{"passed":True},
        "two_d_trials":[
            {"index":0,"true_h":1000.,"methods":{
                "revised":{"status":"completed","score":{"field":{"rmse_K":.08,"maximum_error_K":.2},"relative_h_error":.12,"observation_mse":.9,"physical":{"passed":False,"normalized_rms":{"interior":.1}},"holdout_parameter_transfer":{"rmse_K":.03},"passed":False}},
                "conventional":{"status":"completed","score":{"field":{"rmse_K":.005,"maximum_error_K":.01},"relative_h_error":.01,"observation_mse":1.,"passed":True}},
                "nominal":{"status":"completed","score":{"passed":False}}}},
            {"index":1,"true_h":1200.,"methods":{"revised":{"status":"failed","failure":"fixed budget exhausted"}}},
            {"index":2,"true_h":800.,"not_run_reason":"development gate failed"}],
        "one_d_result":{"status":"completed","planned_trials":6,"trials":[
            {"family":"matched_pwl","index":0,"methods":{
                "scalar":{"completed":True,"multipliers":[1.01]*3,"mse":4.,"curve":{"relative_rmse":.02},"maximum_knot_multiplier_error":.025,"matched_curve_recovery_passed":None},
                "three_knot":{"completed":True,"multipliers":[1.,1.035,1.],"mse":.9,"curve":{"relative_rmse":.001},"maximum_knot_multiplier_error":.001,"matched_curve_recovery_passed":True}}},
            {"family":"cubic_mismatch","index":0,"methods":{"three_knot":{"completed":False,"failure":"example failure"}}},
            {"family":"matched_pwl","index":1,"completed":False,"not_run_reason":"numerical reference failed"}]}}


class RevisionReportTests(unittest.TestCase):
    def test_renderer_preserves_provided_failures_denominators_and_pairing(self):
        result=_result(); original=copy.deepcopy(result)
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch("thermotwin.reports.pinn_spatial_revision._plots",return_value=([],None)):
                artifacts=render_revision_report(temporary,result)
            report=Path(artifacts["report"]).read_text()
        self.assertEqual(result,original)
        self.assertIn("**6 trials**",report)
        self.assertIn("**6 datasets**",report)
        self.assertIn("three paired independent noise blocks",report)
        self.assertIn("| revised | failed |",report)
        self.assertIn("| not_run | not_run |",report)
        self.assertIn("cubic_mismatch",report)
        self.assertIn("| three_knot | failed |",report)
        self.assertIn("no new 1D neural training",report)
        # An apparently attractive observation fit cannot change supplied pass.
        self.assertIn("0.9 | no |",report)
        self.assertIn("hard_boundary_decomposed",report)
        self.assertIn("fixed budget exhausted",report)

    def test_physical_score_and_alternate_normalized_key_are_displayed(self):
        for key in ("physical","physicalnormalized"):
            result=_result()
            score=result["two_d_trials"][0]["methods"]["revised"]["score"]
            score.pop("physical")
            score[key]={"normalized_rms":{"residual_marker":.123},"passed":False}
            with tempfile.TemporaryDirectory() as temporary:
                with mock.patch("thermotwin.reports.pinn_spatial_revision._plots",return_value=([],None)):
                    report=Path(render_revision_report(temporary,result)["report"]).read_text()
            self.assertIn("residual_marker",report)
            self.assertIn("0.123",report)

    def test_blocked_evidence_is_renderable_without_fabricating_rows(self):
        result={"status":"blocked","planned_denominators":{"two_d":6,"one_d":6},
                "two_d_trials":[],"one_d_result":{"status":"blocked","blocked_reason":"independent_truth_refinement_failed","trials":[]}}
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch("thermotwin.reports.pinn_spatial_revision._plots",return_value=([],None)):
                report=Path(render_revision_report(temporary,result)["report"]).read_text()
        self.assertIn("No 2D trial rows were supplied",report)
        self.assertIn("No 1D dataset rows were supplied",report)
        self.assertIn("independent_truth_refinement_failed",report)

    def test_cost_table_uses_recorded_scopes_and_retains_failed_unrun_rows(self):
        result=_result()
        result["two_d_trials"][0]["methods"]["revised"]["report"]={
            "wall_seconds":12.5,"cpu_seconds":11.25,"timing_scope":"optimizer stages only"}
        result["two_d_trials"][0]["methods"]["conventional"]["fit"]={
            "wall_seconds":.25,"cpu_seconds":.2,"timing_scope":"parameter fit only"}
        result["two_d_trials"][1]["methods"]["revised"]["report"]={"wall_seconds":3.5}
        first=result["one_d_result"]["trials"][0]
        first.update(completed=True,elapsed_seconds=.5)
        first["methods"]["scalar"].update(wall_seconds=None,elapsed_seconds=.04)
        original=copy.deepcopy(result)
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch("thermotwin.reports.pinn_spatial_revision._plots",return_value=([],None)):
                report=Path(render_revision_report(temporary,result)["report"]).read_text()
        self.assertEqual(result,original)
        self.assertIn("| 2D | 0 | revised | completed | 12.5 | 11.25 | optimizer stages only |",report)
        self.assertIn("| 2D | 0 | conventional | completed | 0.25 | 0.2 | parameter fit only |",report)
        self.assertIn("| 2D | 1 | revised | failed | 3.5 | not recorded |",report)
        self.assertIn("| 2D | 2 | not_run | not_run | not recorded | not recorded |",report)
        self.assertIn("| 1D matched_pwl | 0 | dataset total | completed | 0.5 |",report)
        self.assertIn("| 1D matched_pwl | 0 | scalar | completed | 0.04 |",report)
        self.assertIn("| 1D matched_pwl | 0 | three_knot | completed | not recorded |",report)
        self.assertNotIn("Resource metadata supplied",report)

    def test_resource_caps_and_peak_rss_appear_only_when_supplied(self):
        result=_result()
        result["resources"]={"wall_limit_seconds":10800,"rss_limit_bytes":2147483648,"peak_rss_bytes":123456789,"threads":1}
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch("thermotwin.reports.pinn_spatial_revision._plots",return_value=([],None)):
                report=Path(render_revision_report(temporary,result)["report"]).read_text()
        self.assertIn('"wall_limit_seconds": 10800',report)
        self.assertIn('"rss_limit_bytes": 2147483648',report)
        self.assertIn('"peak_rss_bytes": 123456789',report)
        self.assertNotIn('"cpu_limit_seconds"',report)

    def test_evaluate_routes_without_rerendering_sealed_artifacts(self):
        study=types.ModuleType("thermotwin.studies.pinn_spatial_revision")
        study.run_fresh_revision=mock.Mock(return_value=_result())
        study.validate_revision_evidence=mock.Mock()
        with mock.patch.dict(sys.modules,{study.__name__:study}):
            with mock.patch("thermotwin.reports.pinn_spatial_revision.render_revision_report") as renderer:
                with contextlib.redirect_stdout(io.StringIO()):
                    returned=main(["evaluate","--output-directory","new-output","--source-commit","abc123","--ci-evidence","ci.json","--preserved-campaign","v1-output"])
        study.run_fresh_revision.assert_called_once_with("new-output",source_commit="abc123",ci_evidence="ci.json",preserved_campaign="v1-output")
        renderer.assert_not_called()
        self.assertEqual(returned["status"],"completed_with_failures")

    def test_validate_routes_without_evaluation(self):
        study=types.ModuleType("thermotwin.studies.pinn_spatial_revision")
        study.run_fresh_revision=mock.Mock()
        study.validate_revision_evidence=mock.Mock(return_value={"valid":True})
        with mock.patch.dict(sys.modules,{study.__name__:study}):
            with contextlib.redirect_stdout(io.StringIO()):
                returned=main(["validate","--output-directory","saved-output"])
        study.validate_revision_evidence.assert_called_once_with("saved-output")
        study.run_fresh_revision.assert_not_called()
        self.assertEqual(returned,{"valid":True})

    def test_help_never_imports_scientific_coordinator(self):
        with mock.patch.dict(sys.modules,{"thermotwin.studies.pinn_spatial_revision":None}):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["--help"])
        self.assertEqual(raised.exception.code,0)


if __name__=="__main__":
    unittest.main()
