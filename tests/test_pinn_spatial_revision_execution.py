"""Constructed coordinator executions; no solver, observations, or fits run.

The filesystem/index path is real, while every scientific factory and provenance
boundary is mocked.  These tests must never open a named campaign or old data.
"""

import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from thermotwin.physics.planar_contact import PlanarContactConfig
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.pinn.planar_contact_revision import PlanarRevisionConfig
from thermotwin.reports.pinn_spatial_revision import main as revision_cli
from thermotwin.studies import pinn_spatial_revision as coordinator


class ConstructedModel(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.physics=PlanarContactConfig()
        self.mode='inverse'
        self.marker=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))


@contextlib.contextmanager
def constructed_execution(temporary,*,numerical_passed,first_neural_fit_fails=False,ci_fails=False):
    """Patch every scientific/provenance boundary before invoking the runner."""
    temporary=Path(temporary)
    config=PlanarRevisionConfig(architecture="soft_mlp",adam_epochs=1,lbfgs_iterations=0,
                              interior_points=4,boundary_points=4,interface_points=4)
    protocol={"constructed_test_only":True,
        "selected_inverse":{"config":asdict(config),"strategy":coordinator.INVERSE_SPEC["strategies"][0]},
        "one_d_selection":{"candidate":{"id":"constructed-no-scientific-inputs"}},
        "planned_denominators":{"two_d":6,"one_d":6,"one_d_method_fits":12,
                                "one_d_independent_paired_noise_blocks":3},
        "preservation":{"constructed_identity":"no-real-campaign-opened"}}
    source={"source_commit":"constructed-source","source_digest":"constructed-digest","source_files":[]}
    ci_path=temporary/"constructed-ci-input.json"
    ci_path.write_text(json.dumps({"constructed_test_only":True}))
    output=temporary/"constructed-output"
    coordinates=np.array([[.01,.003,10.],[.03,.003,20.]])
    observations=PlanarObservations(coordinates,np.array([299.99,299.98]))
    reference_values=np.array([299.9,299.8])
    reference_marker=object()

    def numerical(directory):
        directory=Path(directory);directory.mkdir()
        value={"passed":numerical_passed,"constructed_test_only":True}
        coordinator.write_json(directory/"constructed-reference.json",value)
        return value

    fit_calls=0
    def neural(*args,**kwargs):
        nonlocal fit_calls
        fit_calls+=1
        if first_neural_fit_fails and fit_calls==1:
            raise RuntimeError("constructed neural exception")
        return ConstructedModel(kwargs["config"]),{"status":"completed","constructed_test_only":True}

    def score(*args,**kwargs):
        # Deliberately failing supplied gates test retention and continuation;
        # none of these constants is a numerical performance observation.
        return {"constructed_test_only":True,"passed":False,"relative_h_error":.2,
                "observation_mse":1.,"field":{"rmse_K":.1,"maximum_error_K":.2},
                "physical":{"passed":False},"holdout_parameter_transfer":{"rmse_K":.1}}

    def one_d(directory,selection,namespace):
        directory=Path(directory);directory.mkdir()
        trials=[{"family":family,"index":index,"noise_block":index,"completed":True,
                 "methods":{"scalar":{"completed":True,"constructed_test_only":True},
                            "three_knot":{"completed":True,"constructed_test_only":True}}}
                for family in ("matched_pwl","cubic_mismatch") for index in range(3)]
        result={"status":"completed","constructed_test_only":True,"planned_trials":6,
                "completed_trials":6,"independent_paired_noise_blocks":3,"trials":trials}
        coordinator.write_json(directory/"result.json",result)
        return result

    def renderer(directory,result):
        path=Path(directory)/"REPORT.md"
        path.write_text("# Constructed execution fixture\n\nNo scientific computation occurred.\n")
        return {"report":str(path),"figures":[]}

    with contextlib.ExitStack() as stack:
        patched={}
        for name,kwargs in {
            "validated_revision_protocol":{"return_value":protocol},
            "runtime_context":{"return_value":{"constructed_runtime":True}},
            "source_context":{"return_value":source},
            "preserved_context":{"return_value":protocol["preservation"]},
            "verify_ci":({"side_effect":ValueError("constructed CI failure")} if ci_fails else {"return_value":{"constructed_ci":True}}),
            "numerical_preflight":{"side_effect":numerical},
            "coordinates_and_reference":{"return_value":(coordinates,reference_values,reference_marker)},
            "observations_from_reference":{"return_value":observations},
            "fit_delayed_inverse":{"side_effect":neural},
            "score_inverse":{"side_effect":score},
            "conventional_fit":{"return_value":{"h":900.,"success":True,"constructed_test_only":True}},
            "baseline_score":{"side_effect":score},
            "run_one_d_evaluation":{"side_effect":one_d},
        }.items():
            patched[name]=stack.enter_context(mock.patch.object(coordinator,name,**kwargs))
        patched["renderer"]=stack.enter_context(mock.patch("thermotwin.reports.pinn_spatial_revision.render_revision_report",side_effect=renderer))
        patched["predict"]=stack.enter_context(mock.patch("thermotwin.pinn.planar_contact_revision.predict_planar_revision",
            side_effect=lambda model,points:np.full(len(points),299.9)))
        patched["signal"]=stack.enter_context(mock.patch.object(coordinator.signal,"signal",return_value="constructed-old-handler"))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        yield types.SimpleNamespace(output=output,ci_path=ci_path,protocol=protocol,mocks=patched,
            kwargs={"source_commit":"constructed-source","ci_evidence":str(ci_path),
                    "preserved_campaign":str(temporary/"never-opened-original-campaign")})


class RevisionExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reference_failure_prevents_every_observation_and_fit_factory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with constructed_execution(temporary,numerical_passed=False) as fixture:
                result=coordinator.run_fresh_revision(fixture.output,**fixture.kwargs)
                self.assertEqual(result["status"],"blocked_reference")
                self.assertEqual(len(result["two_d_trials"]),6)
                for row in result["two_d_trials"]:
                    self.assertEqual(set(row["methods"]),{"revised","conventional","nominal"})
                    for method in row["methods"].values():
                        self.assertEqual(method["status"],"not_run")
                        self.assertEqual(method["reason"],"numerical_reference_failed")
                one_d=result["one_d_result"]
                self.assertEqual(one_d["status"],"not_run")
                self.assertEqual(len(one_d["trials"]),6)
                self.assertEqual(one_d["independent_paired_noise_blocks"],3)
                self.assertTrue(all(row["status"]=="not_run" for row in one_d["trials"]))
                for name in ("coordinates_and_reference","observations_from_reference","fit_delayed_inverse",
                             "score_inverse","conventional_fit","baseline_score","run_one_d_evaluation","predict"):
                    fixture.mocks[name].assert_not_called()
                fixture.mocks["renderer"].assert_called_once()
                self.assertGreater(coordinator.validate_revision_evidence(fixture.output)["validated_files"],3)
                self.assertEqual(fixture.mocks["signal"].call_count,2)

    def test_neural_exception_retains_six_cases_other_methods_and_sealed_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            with constructed_execution(temporary,numerical_passed=True,first_neural_fit_fails=True) as fixture:
                result=coordinator.run_fresh_revision(fixture.output,**fixture.kwargs)
                self.assertEqual(result["status"],"completed")
                trials=result["two_d_trials"]
                self.assertEqual([row["index"] for row in trials],list(range(6)))
                self.assertEqual([row["true_h"] for row in trials],[h for h in coordinator.INVERSE_SPEC["evaluation_h"] for _ in range(2)])
                self.assertEqual(trials[0]["methods"]["revised"]["status"],"failed")
                self.assertIn("constructed neural exception",trials[0]["methods"]["revised"]["error"])
                self.assertTrue(all(row["methods"]["revised"]["status"]=="completed" for row in trials[1:]))
                for row in trials:
                    self.assertEqual(row["methods"]["conventional"]["status"],"completed")
                    self.assertEqual(row["methods"]["nominal"]["status"],"completed")
                self.assertEqual(fixture.mocks["coordinates_and_reference"].call_count,6)
                self.assertEqual(fixture.mocks["observations_from_reference"].call_count,6)
                self.assertEqual(fixture.mocks["fit_delayed_inverse"].call_count,6)
                self.assertEqual(fixture.mocks["score_inverse"].call_count,5)
                self.assertEqual(fixture.mocks["conventional_fit"].call_count,6)
                self.assertEqual(fixture.mocks["baseline_score"].call_count,12)
                fixture.mocks["run_one_d_evaluation"].assert_called_once()
                self.assertEqual(result["one_d_result"]["independent_paired_noise_blocks"],3)
                self.assertEqual(result["planned_denominators"]["one_d_method_fits"],12)
                one_d_trials=result["one_d_result"]["trials"]
                self.assertEqual(len(one_d_trials),6)
                self.assertEqual([row["noise_block"] for row in one_d_trials[:3]],
                                 [row["noise_block"] for row in one_d_trials[3:]])
                validated=coordinator.validate_revision_evidence(fixture.output)
                self.assertGreater(validated["validated_files"],20)
                fixture.mocks["renderer"].assert_called_once()
                # Calling the CLI after a completed coordinator return cannot
                # render again or invalidate its just-sealed artifact index.
                with mock.patch.object(coordinator,"run_fresh_revision",return_value=result) as runner:
                    with contextlib.redirect_stdout(io.StringIO()):
                        revision_cli(["evaluate","--output-directory",str(fixture.output),"--source-commit","constructed-source",
                                      "--ci-evidence",str(fixture.ci_path),"--preserved-campaign",fixture.kwargs["preserved_campaign"]])
                    runner.assert_called_once()
                fixture.mocks["renderer"].assert_called_once()
                self.assertEqual(coordinator.validate_revision_evidence(fixture.output),validated)
                (fixture.output/"REPORT.md").write_text("constructed tamper\n")
                with self.assertRaisesRegex(ValueError,"evidence changed"):
                    coordinator.validate_revision_evidence(fixture.output)

    def test_rejected_ci_cannot_create_output_or_reach_scientific_factories(self):
        with tempfile.TemporaryDirectory() as temporary:
            with constructed_execution(temporary,numerical_passed=True,ci_fails=True) as fixture:
                with self.assertRaisesRegex(ValueError,"constructed CI failure"):
                    coordinator.run_fresh_revision(fixture.output,**fixture.kwargs)
                self.assertFalse(fixture.output.exists())
                for name in ("numerical_preflight","coordinates_and_reference","observations_from_reference",
                             "fit_delayed_inverse","conventional_fit","run_one_d_evaluation","renderer"):
                    fixture.mocks[name].assert_not_called()

    def test_scoring_failure_preserves_completed_fit_and_optimizer_failure_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            with constructed_execution(temporary,numerical_passed=True) as fixture:
                fixture.mocks['score_inverse'].side_effect=RuntimeError('constructed scoring error')
                fixture.mocks['conventional_fit'].return_value={'h':900.,'success':False}
                fixture.mocks['baseline_score'].side_effect=lambda *args,**kwargs:{'passed':True,'gates':{}}
                result=coordinator.run_fresh_revision(fixture.output,**fixture.kwargs)
                for row in result['two_d_trials']:
                    directory=fixture.output/f"two-d-trial-{row['index']:02d}"
                    self.assertTrue((directory/'revised-fit.json').is_file())
                    self.assertTrue((directory/'revised-final.pt').is_file())
                    self.assertEqual(row['methods']['revised']['status'],'failed')
                    conventional=row['methods']['conventional']
                    self.assertEqual(conventional['status'],'optimizer_failed')
                    self.assertFalse(conventional['score']['passed'])
                    self.assertFalse(conventional['score']['gates']['execution'])

    def test_interrupted_one_d_restores_completed_rows_and_partial_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with constructed_execution(temporary,numerical_passed=True) as fixture:
                def interrupted(directory,*args):
                    saved=Path(directory)/'matched_pwl-trial-00'
                    saved.mkdir(parents=True)
                    coordinator.write_json(saved/'result.json',{
                        'family':'matched_pwl','index':0,'completed':True,'methods':{}})
                    raise SystemExit('constructed interruption')
                fixture.mocks['run_one_d_evaluation'].side_effect=interrupted
                with self.assertRaisesRegex(SystemExit,'constructed interruption'):
                    coordinator.run_fresh_revision(fixture.output,**fixture.kwargs)
                partial=json.loads((fixture.output/'partial-result.json').read_text())
                self.assertEqual(partial['one_d_result']['status'],'interrupted')
                self.assertEqual(len(partial['one_d_result']['trials']),6)
                self.assertEqual(partial['one_d_result']['completed_trials'],1)
                self.assertTrue(partial['one_d_result']['trials'][0]['completed'])
                inventory=json.loads((fixture.output/'incomplete-evidence-index.json').read_text())
                records={record['path']:record for record in inventory['files']}
                self.assertIn('one-d/matched_pwl-trial-00/result.json',records)
                self.assertIn('two-d-trial-05/revised-final.pt',records)
                self.assertEqual(inventory['digest'],coordinator.canonical_digest(inventory['files']))
                self.assertFalse((fixture.output/'evidence-index.json').exists())

    def test_partial_recovery_skips_truncated_files_and_recovers_other_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            bad=out/'one-d/matched_pwl-trial-00'
            good=out/'one-d/matched_pwl-trial-01'
            bad.mkdir(parents=True); good.mkdir()
            (out/'one-d/result.json').write_text('{"trials":[')
            (bad/'result.json').write_text('{"family":')
            (bad/'scalar-raw-fit.json').write_text('{"fit_completed":true}')
            coordinator.write_json(good/'result.json',{'family':'matched_pwl','index':1,'completed':True,'methods':{}})
            result={'one_d_result':{'status':'not_run','trials':[
                {'family':family,'index':index,'completed':False,'status':'not_run'}
                for family in ('matched_pwl','cubic_mismatch') for index in range(3)]}}
            coordinator.restore_partial_one_d(out,result)
            saved=result['one_d_result']
            self.assertEqual(saved['completed_trials'],1)
            self.assertEqual(len(saved['recovery_errors']),2)
            self.assertEqual(saved['trials'][0]['status'],'interrupted')
            self.assertTrue(saved['trials'][0]['methods']['scalar']['fit_completed'])
            self.assertTrue(saved['trials'][1]['completed'])
            self.assertEqual(len(saved['trials']),6)

    def test_conventional_wrapper_records_real_timers_without_running_a_solver(self):
        observations=PlanarObservations(np.array([[.02,.003,10.]]),np.array([300.]))
        optimizer=types.SimpleNamespace(x=np.log([900.]),success=True,nfev=2,message='constructed',fun=np.array([.2]))
        with mock.patch.object(coordinator,'PlanarContactModel'),mock.patch.object(coordinator,'least_squares',return_value=optimizer):
            fit=coordinator.conventional_fit(observations)
        self.assertTrue(fit['success'])
        self.assertAlmostEqual(fit['h'],900.)
        self.assertGreaterEqual(fit['wall_seconds'],0.)
        self.assertGreaterEqual(fit['cpu_seconds'],0.)

    def test_unfrozen_protocol_blocks_public_runner_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary=Path(temporary)
            with mock.patch.object(coordinator,'PROTOCOL_PATH',temporary/'not-frozen.json'):
                with self.assertRaisesRegex(ValueError,'complete protocol is not frozen'):
                    coordinator.run_fresh_revision(temporary/'output',source_commit='unused',
                        ci_evidence='unused',preserved_campaign='unused')
            self.assertFalse((temporary/'output').exists())


if __name__=="__main__":
    unittest.main()
