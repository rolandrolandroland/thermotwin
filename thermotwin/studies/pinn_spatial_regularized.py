"""Four bounded fresh forward fits with one frozen continuous readout rule.

Calibration uses only each original common initialization and noisy training
evidence. The prerequisite gradient checks and the unchanged forward gates are
reported separately; every declared fit opportunity stays in the denominator.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

from . import pinn_spatial_continuity as common
from . import pinn_spatial_graded as graded
from . import pinn_spatial_step_control as previous
from . import pinn_spatial_cutoff_audit as cutoff

ROOT = common.ROOT
CASE_SECONDS, TOP_SECONDS, RSS_BYTES = previous.CASE_SECONDS, previous.TOP_SECONDS, previous.RSS_BYTES
WAVES, STOPS = previous.WAVES, previous.STOPS
_read, _case = previous._read, previous._case
GRADIENT_SEMANTICS = ('fixed-head partial derivative equals the regularized profiled derivative at a unique stationary head; '
    'frozen D and lambda have no hidden-parameter dependence')


def _api():
    from ..pinn import planar_contact_regularized
    return planar_contact_regularized


def _config(values, prior):
    config = asdict(_api().RegularizedConfig(**values))
    if set(config) != set(values):
        raise ValueError('The full regularized configuration must be explicit')
    if (config['adam_epochs'] != 320 or config['maximum_trial_solves'] != 480
        or config['max_backtracks'] != 4):
        raise ValueError('The frozen comparison requires 320 attempts, 480 trial solves and four backtracks')
    for key, value in prior.items():
        if config.get(key) != value:
            raise ValueError('The readout revision changed a frozen architecture, grid, objective weight or controller: '+key)
    return config


def _bound(path):
    path = Path(path)
    path = (ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    path.relative_to(ROOT)
    return path


def _method_gate(path, config, validator_path):
    path, validator_path = _bound(path), _bound(validator_path)
    decision = _read(path)
    followup = {'cases': list(common.CASES), 'maximum_attempts_per_case': 320,
        'maximum_trial_solves_per_case': 480, 'maximum_backtracks': 4,
        'acceptance_thresholds': common.GATES, 'parallel_workers': 2, 'worker_threads': 1,
        'case_wall_seconds': CASE_SECONDS, 'case_rss_bytes': RSS_BYTES,
        'supervisor_wall_seconds': TOP_SECONDS, 'automatic_retries': 0}
    if (decision.get('schema') != 'regularized_training_only_method_decision_v1'
        or decision.get('status') != 'approved_for_forward_development'
        or decision.get('method_checks_passed') is not True or decision.get('training_only') is not True
        or decision.get('reference_fields_used') is not False
        or decision.get('independent_diagnostics_used_for_selection') is not False
        or decision.get('selected_config') != config or decision.get('followup') != followup
        or decision.get('no_guarantee_of_forward_gate_success') is not True
        or decision.get('no_further_method_search') is not True):
        raise ValueError('The declared training-only method checks must pass before forward preparation')
    spec = importlib.util.spec_from_file_location('regularized_forward_prerequisite', validator_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    probe = module.verify_gradient_probe(_bound(decision['gradient_probe']['directory']))
    if probe != decision['gradient_probe'] or probe.get('method_checks_passed') is not True:
        raise ValueError('The sealed gradient-probe proof differs from the approved decision')
    return {'decision': {'path': str(path.relative_to(ROOT)), 'sha256': common.digest_file(path), 'value': decision},
        'validator': {'path': str(validator_path.relative_to(ROOT)), 'sha256': common.digest_file(validator_path)},
        'probe_context': probe, 'method_checks_passed': True}


def _cutoff_context(directory):
    """Read-only post-commit authentication; the old audit is never rerun."""
    directory = Path(directory).resolve()
    result = previous._validate_worker(directory)
    plan = _read(directory/'plan.json')
    if (result.get('status') != 'completed' or result.get('attempted_readout_solves') != 14
        or result.get('successful_readout_solves') != 14):
        raise ValueError('The completed fourteen-opportunity cutoff audit must be retained')
    if common._source_records(plan['source_sha256']) != plan['source_sha256']:
        raise RuntimeError('The frozen cutoff source changed')
    if common._source_records(plan['protected_files']) != plan['protected_files']:
        raise RuntimeError('A protected cutoff declaration changed')
    for path, records in plan['protected_trees'].items():
        if common._tree_records(path) != records:
            raise RuntimeError('A protected pre-audit experiment changed')
    for path, expected in plan['source_sha256'].items():
        if common.digest_file(directory/'source-snapshot'/path) != expected:
            raise RuntimeError('A cutoff source snapshot changed')
    compact_path = ROOT/'thermotwin/PINN_SPATIAL_CUTOFF_AUDIT_RESULT.json'
    compact = _read(compact_path)
    if (compact['plan_sha256'] != common.digest_file(directory/'plan.json')
        or compact['raw_result']['sha256'] != common.digest_file(directory/'result.json')
        or compact['artifact_index']['sha256'] != common.digest_file(directory/'artifact-index.json')):
        raise RuntimeError('The compact cutoff record does not bind the completed audit')
    return {'directory': str(directory), 'plan_sha256': common.digest_file(directory/'plan.json'),
        'artifact_index_sha256': common.digest_file(directory/'artifact-index.json'),
        'compact_result_sha256': common.digest_file(compact_path), 'files': common._tree_records(directory)}


def _preflight(config, prior_plan, progress=None):
    """Exactly four training-only initial design calibrations, no head solves."""
    api, grid_api = _api(), graded._model_api()
    instance, result = api.RegularizedConfig(**config), {}
    for case in common.CASES:
        physics = common.physics_for(case)
        model, initialization = api.matched_regularized_model(physics, instance, case['known_h'], common.SEEDS['neural'])
        grid = grid_api.graded_point_metadata(physics, instance, common.SEEDS['sampling'])
        old = prior_plan['preflight_by_case'][case['id']]
        if (initialization['common_weights_sha256'] != old['initialization']['common_weights_sha256']
            or initialization.get('log_column_present') is not False
            or initialization['probe_agreement']['passed'] is not True):
            raise ValueError('The regularized fit did not retain the original common initialization')
        if grid != old['training_grid']:
            raise ValueError('The regularized calibration changed the original graded grid')
        observations = common.paired_observations(case, prior_plan)
        points = grid_api.graded_training_points(physics, instance, common.SEEDS['sampling'])
        if progress:
            progress(case['id'], None)
        calibration = api.calibrate_readout(model, points, observations)
        _calibration_proof(calibration, config, grid)
        result[case['id']] = {'initialization': initialization, 'training_grid': grid,
            'readout_calibration': calibration, 'calibration_sha256': common.canonical_digest(calibration)}
        if progress:
            progress(case['id'], result[case['id']])
    return result


def _calibration_proof(calibration, config, grid):
    common._json_bytes(calibration)
    scales = np.asarray(calibration.get('column_scale'), dtype=np.float64)
    strength, sigma = calibration.get('fixed_lambda'), calibration.get('initial_sigma_max')
    if (calibration.get('schema') != 'fixed_initial_column_scale_ridge_v1'
        or calibration.get('relative_scale') != config['rank_rcond']
        or calibration.get('relative_scale') != config['regularization_relative_scale']
        or config['regularization_relative_scale'] != 1e-7
        or calibration.get('calibration_matrix_builds') != 1
        or calibration.get('calibration_spectral_calculations') != 1
        or type(calibration.get('calibration_matrix_builds')) is not int
        or type(calibration.get('calibration_spectral_calculations')) is not int
        or calibration.get('frozen_for_entire_fit') is not True
        or calibration.get('hidden_parameters_differentiated_through_calibration') is not False
        or calibration.get('sha256') != common.canonical_digest({key: value for key, value in calibration.items() if key != 'sha256'})
        or scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all() or np.any(scales < 1e-12)
        or graded._model_api()._array_digest(scales) != calibration.get('column_scale_sha256')
        or type(strength) not in (int, float) or type(sigma) not in (int, float)
        or not math.isfinite(strength) or not math.isfinite(sigma) or sigma <= 0
        or strength != config['regularization_relative_scale']*sigma
        or (calibration.get('initial_matrix_shape') or [None])[-1] != len(scales)
        or (calibration.get('bindings') or {}).get('config') != config
        or (calibration.get('bindings') or {}).get('points') != {
            name: value['sha256'] for name, value in grid['point_identities'].items()}):
        raise ValueError('The fixed initial training-only ridge calibration is inconsistent')
    return True


def _worker_plan(case, config, sources, preflight):
    return {'schema': 'regularized_single_case_worker_v8', 'case': case,
        'model_config_digest': common.canonical_digest(config), 'seeds': common.SEEDS,
        'source_digest': common.canonical_digest(sources),
        'calibration_sha256': preflight['calibration_sha256'],
        'maximum_wall_seconds': CASE_SECONDS, 'maximum_rss_bytes': RSS_BYTES,
        'retries': 0, 'threads': 1}


def _calibration_declaration(config, sources, runtime, method, paired_evidence):
    return {'schema': 'four_initial_training_calibrations_v1', 'model_config': config,
        'source_sha256': sources, 'runtime': runtime, 'method_checks': method,
        'seeds': common.SEEDS, 'maximum_calibration_opportunities': 4,
        'maximum_matrix_builds': 4, 'maximum_spectral_calculations': 4, 'head_solves': 0,
        'automatic_retries': 0, 'no_reference_fields': True,
        'paired_training_inputs': {case['id']: paired_evidence[case['id']]['observations.npz'] for case in common.CASES}}


def _protected_output_directories(prior, audit_directory, audit_plan, method):
    return sorted({str(Path(path).resolve()) for path in (
        prior, audit_directory, method['probe_context']['directory'], *audit_plan['protected_trees'])})


def _output_location(directory, protected):
    for name in protected:
        old = Path(name).resolve()
        if directory.is_relative_to(old) or old.is_relative_to(directory):
            raise ValueError('The new comparison must be outside every inherited protected experiment tree')
    for ancestor in directory.parents:
        if any((ancestor/name).is_file() for name in ('artifact-index.json', 'evidence-index.json', 'report-index.json', 'manifest.json')):
            raise ValueError('The new comparison must not alter a sealed ancestor directory')


def prepare_regularized(output_directory, *, model_config, method_decision_path,
        gradient_probe_validator_path, source_paths, step_control_campaign,
        cutoff_audit_directory, protected_files):
    cutoff._one_thread()
    directory = Path(output_directory).resolve()
    if directory.exists():
        raise FileExistsError('The once-only regularized preparation path already exists')
    prior = Path(step_control_campaign).resolve()
    prior_context = previous.validate_step_control(prior)
    prior_plan = _read(prior/'plan.json')
    config = _config(model_config, prior_plan['model_config'])
    method = _method_gate(method_decision_path, config, gradient_probe_validator_path)
    audit_context = _cutoff_context(cutoff_audit_directory)
    audit_plan = _read(Path(cutoff_audit_directory)/'plan.json')
    protected_output = _protected_output_directories(prior, cutoff_audit_directory, audit_plan, method)
    _output_location(directory, protected_output)
    sources = set(source_paths) | set(prior_plan['source_sha256']) | set(
        audit_plan['source_sha256']) | {
        'thermotwin/pinn/planar_contact_regularized.py',
        'thermotwin/studies/pinn_spatial_regularized.py',
        'tests/test_pinn_spatial_regularized.py', 'tests/test_planar_contact_regularized.py', method['validator']['path']}
    source_records = common._source_records(sources)
    protected = set(protected_files) | {
        'thermotwin/PINN_SPATIAL_OPTIMIZATION_PLAN.json', 'thermotwin/PINN_SPATIAL_OPTIMIZATION_PROBE.json',
        'thermotwin/PINN_SPATIAL_OPTIMIZATION_RESULT.json', 'thermotwin/PINN_SPATIAL_STEP_CONTROL_PLAN.json',
        'thermotwin/PINN_SPATIAL_STEP_CONTROL_PROBE.json', 'thermotwin/PINN_SPATIAL_STEP_CONTROL_RESULT.json',
        'thermotwin/PINN_SPATIAL_CUTOFF_AUDIT_PLAN.json', 'thermotwin/PINN_SPATIAL_CUTOFF_AUDIT_RESULT.json'}
    protected_records, runtime = common._source_records(protected), cutoff.runtime_context()
    directory.mkdir(parents=True, exist_ok=False)
    calibration_plan = _calibration_declaration(config, source_records, runtime, method, prior_plan['paired_evidence'])
    common._write(directory/'calibration-plan.json', calibration_plan)
    calibration_state = {'status': 'running', 'plan_sha256': common.digest_file(directory/'calibration-plan.json'),
        'attempted_cases': [], 'completed_cases': []}
    common._write(directory/'calibration-state.json', calibration_state)
    def retain_calibration(case_id, record):
        if record is None:
            calibration_state['attempted_cases'].append(case_id)
        else:
            common._write(directory/('calibration-'+case_id+'.json'), record)
            calibration_state['completed_cases'].append(case_id)
        common._write(directory/'calibration-state.json', calibration_state)
    try:
        preflight = _preflight(config, prior_plan, progress=retain_calibration)
        if (common._source_records(source_records) != source_records or cutoff.runtime_context() != runtime
            or calibration_state['attempted_cases'] != [case['id'] for case in common.CASES]
            or calibration_state['completed_cases'] != calibration_state['attempted_cases']):
            raise RuntimeError('Calibration source, runtime or once-only accounting changed')
    except BaseException as error:
        calibration_state.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        common._write(directory/'calibration-state.json', calibration_state)
        raise
    calibration_state['status'] = 'completed'
    common._write(directory/'calibration-state.json', calibration_state)
    workers = {}
    for case in common.CASES:
        worker = directory/case['id']
        worker.mkdir()
        common._write(worker/'plan.json', _worker_plan(case, config, source_records, preflight[case['id']]))
        workers[case['id']] = {'directory': case['id'], 'plan_sha256': common.digest_file(worker/'plan.json')}
        common._write(worker/'state.json', {'status': 'prepared', 'plan_sha256': workers[case['id']]['plan_sha256']})
    plan = {'schema': 'bounded_regularized_forward_v8',
        'candidate_id': 'continuous-regularized-readout-v8-log-off',
        'cases': list(common.CASES), 'model_config': config, 'source_sha256': source_records,
        'runtime': runtime, 'seeds': common.SEEDS, 'gates': common.GATES,
        'workers': workers, 'waves': [list(wave) for wave in WAVES], 'maximum_method_fits': 4,
        'reused_paired_datasets': 4, 'independent_noise_replications_claimed': False,
        'topology': {'maximum_concurrent_workers': 2, 'threads_per_worker': 1,
            'rss_bytes_per_worker': RSS_BYTES, 'aggregate_child_rss_bytes': 2*RSS_BYTES,
            'seconds_per_worker': CASE_SECONDS, 'top_seconds': TOP_SECONDS},
        'allowed_training_stop_reasons': list(STOPS), 'automatic_retries': 0,
        'paired_evidence': prior_plan['paired_evidence'], 'method_checks': method,
        'preserved_step_control': {'directory': str(prior), 'evidence': prior_context,
            'plan_sha256': common.digest_file(prior/'plan.json'), 'files': common._tree_records(prior)},
        'preserved_cutoff_audit': audit_context, 'protected_files': protected_records,
        'protected_output_directories': protected_output,
        'preflight_by_case': preflight,
        'calibration_preparation': {'plan_sha256': common.digest_file(directory/'calibration-plan.json'),
            'state_sha256': common.digest_file(directory/'calibration-state.json'),
            'receipts': {case['id']: common.digest_file(directory/('calibration-'+case['id']+'.json')) for case in common.CASES}},
        'calibration_accounting': {'preparation_matrix_builds': 4, 'preparation_spectral_calculations': 4,
            'worker_matrix_builds': 0, 'worker_spectral_calculations': 0,
            'initial_and_final_head_solves_per_fit': 2, 'maximum_trial_head_solves_per_fit': 480},
        'near_bottom_audit_digest': common.point_digest(graded.near_bottom_points()),
        'checkpoint_rule': 'training-returned terminal state retained before independent diagnostics and hidden-field score',
        'initialization_policy': 'fresh original common weights; frozen per-case initial training calibration; no continuation',
        'selection_boundary': 'full regularized training objective only; frozen gradient-method gate separate from unchanged forward acceptance'}
    common._write(directory/'plan.json', plan)
    for path in source_records:
        target = directory/'source-snapshot'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/path).read_bytes())
    common._write(directory/'state.json', {'status': 'prepared', 'plan_sha256': common.digest_file(directory/'plan.json')})
    verify_regularized_plan(plan, directory)
    return plan


def verify_regularized_plan(plan, output_directory=None):
    cutoff._one_thread()
    expected = {'schema': 'bounded_regularized_forward_v8', 'cases': list(common.CASES),
        'seeds': common.SEEDS, 'gates': common.GATES, 'waves': [list(wave) for wave in WAVES],
        'maximum_method_fits': 4, 'automatic_retries': 0, 'reused_paired_datasets': 4,
        'independent_noise_replications_claimed': False, 'allowed_training_stop_reasons': list(STOPS),
        'topology': {'maximum_concurrent_workers': 2, 'threads_per_worker': 1,
            'rss_bytes_per_worker': RSS_BYTES, 'aggregate_child_rss_bytes': 2*RSS_BYTES,
            'seconds_per_worker': CASE_SECONDS, 'top_seconds': TOP_SECONDS},
        'calibration_accounting': {'preparation_matrix_builds': 4, 'preparation_spectral_calculations': 4,
            'worker_matrix_builds': 0, 'worker_spectral_calculations': 0,
            'initial_and_final_head_solves_per_fit': 2, 'maximum_trial_head_solves_per_fit': 480}}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('The regularized scope, gates, budgets or denominator changed')
    if common._source_records(plan['source_sha256']) != plan['source_sha256']:
        raise RuntimeError('The regularized numerical source changed')
    if cutoff.runtime_context() != plan['runtime'] or common._source_records(plan['protected_files']) != plan['protected_files']:
        raise RuntimeError('The regularized runtime or prior tracked record changed')
    prior = plan['preserved_step_control']
    if (previous.validate_step_control(prior['directory']) != prior['evidence']
        or common._tree_records(prior['directory']) != prior['files']
        or common.digest_file(Path(prior['directory'])/'plan.json') != prior['plan_sha256']):
        raise RuntimeError('The completed V7/V6 or older experiment chain changed')
    if _cutoff_context(plan['preserved_cutoff_audit']['directory']) != plan['preserved_cutoff_audit']:
        raise RuntimeError('The completed cutoff audit changed')
    inherited = _read(Path(plan['preserved_cutoff_audit']['directory'])/'plan.json')
    if plan.get('protected_output_directories') != _protected_output_directories(
            prior['directory'], plan['preserved_cutoff_audit']['directory'], inherited, plan['method_checks']):
        raise ValueError('The inherited protected output boundaries changed')
    old_plan = _read(Path(prior['directory'])/'plan.json')
    _config(plan['model_config'], old_plan['model_config'])
    if plan['paired_evidence'] != old_plan['paired_evidence']:
        raise ValueError('The paired noisy data or scoring evidence changed')
    method = plan['method_checks']
    if _method_gate(method['decision']['path'], plan['model_config'], method['validator']['path']) != method:
        raise RuntimeError('The prospective method decision or gradient-probe proof changed')
    if common.point_digest(graded.near_bottom_points()) != plan['near_bottom_audit_digest']:
        raise RuntimeError('The report-only near-bottom points changed')
    if set(plan['preflight_by_case']) != {case['id'] for case in common.CASES}:
        raise ValueError('The calibration denominator changed')
    for case in common.CASES:
        record = plan['preflight_by_case'][case['id']]
        _calibration_proof(record['readout_calibration'], plan['model_config'], record['training_grid'])
        if common.canonical_digest(record['readout_calibration']) != record['calibration_sha256']:
            raise RuntimeError('A frozen initial calibration changed')
        old = old_plan['preflight_by_case'][case['id']]
        if (record['initialization']['common_weights_sha256'] != old['initialization']['common_weights_sha256']
            or record['training_grid'] != old['training_grid']):
            raise ValueError('Fresh initialization or graded-grid pairing changed')
    if output_directory is not None:
        directory = Path(output_directory).resolve()
        _output_location(directory, plan['protected_output_directories'])
        if _read(directory/'plan.json') != plan:
            raise RuntimeError('The regularized plan changed on disk')
        preparation = plan['calibration_preparation']
        expected_calibration_plan = _calibration_declaration(plan['model_config'], plan['source_sha256'],
            plan['runtime'], plan['method_checks'], plan['paired_evidence'])
        if (_read(directory/'calibration-plan.json') != expected_calibration_plan
            or common.digest_file(directory/'calibration-plan.json') != preparation['plan_sha256']
            or common.digest_file(directory/'calibration-state.json') != preparation['state_sha256']
            or _read(directory/'calibration-state.json') != {'status': 'completed',
                'plan_sha256': preparation['plan_sha256'], 'attempted_cases': [case['id'] for case in common.CASES],
                'completed_cases': [case['id'] for case in common.CASES]}):
            raise RuntimeError('The once-only four-calibration preparation proof changed')
        for case in common.CASES:
            receipt = directory/('calibration-'+case['id']+'.json')
            if (common.digest_file(receipt) != preparation['receipts'][case['id']]
                or _read(receipt) != plan['preflight_by_case'][case['id']]):
                raise RuntimeError('A completed calibration receipt changed')
        for name, expected_hash in plan['source_sha256'].items():
            if common.digest_file(directory/'source-snapshot'/name) != expected_hash:
                raise RuntimeError('A regularized pre-fit source snapshot changed')
        for case in common.CASES:
            record = plan['workers'][case['id']]
            if record['directory'] != case['id']:
                raise ValueError('A worker directory changed')
            path = directory/case['id']/'plan.json'
            if (common.digest_file(path) != record['plan_sha256'] or _read(path) !=
                _worker_plan(case, plan['model_config'], plan['source_sha256'], plan['preflight_by_case'][case['id']])):
                raise RuntimeError('A single-case regularized rule changed')
    return {'method_fit_count': 4, 'reused_dataset_count': 4,
        'method_checks_passed': True, 'source_sha256': plan['source_sha256']}


def _training_proof(raw, plan, case_id):
    controller = previous._training_proof(raw, plan, case_id)
    expected = plan['preflight_by_case'][case_id]
    if raw.get('readout_calibration') != expected['readout_calibration']:
        raise ValueError('The fit changed its frozen initial readout calibration')
    if (raw.get('initial_calibration_matrix_builds') != 0
        or raw.get('initial_calibration_spectral_calculations') != 0
        or type(raw.get('initial_calibration_matrix_builds')) is not int
        or type(raw.get('initial_calibration_spectral_calculations')) is not int):
        raise ValueError('A worker repeated the pre-fit calibration or spectral calculation')
    total, unpenalized, penalty = (raw.get(key) for key in
        ('final_objective', 'final_unpenalized_objective', 'final_regularization_penalty'))
    if (not all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in (total, unpenalized, penalty))
        or not math.isclose(total, unpenalized+penalty, rel_tol=1e-10, abs_tol=1e-10)):
        raise ValueError('The terminal full training objective omits or changes its penalty')
    stationarity = raw.get('final_head_stationarity')
    config = plan['model_config']
    if (type(stationarity) not in (int, float) or not math.isfinite(stationarity)
        or not 0 <= stationarity <= config['head_stationarity_tolerance']
        or raw.get('head_stationarity_required') is not True
        or raw.get('hard_singular_value_truncation') is not False
        or raw.get('hidden_truth_labels_used') is not False
        or raw.get('independent_diagnostics_used_for_training') is not False
        or raw.get('gradient_semantics') != GRADIENT_SEMANTICS):
        raise ValueError('The terminal stationary unique-readout method evidence is invalid')
    projections = raw.get('projections') or []
    trial_projections = [trial['projection'] for update in raw['hidden_update_history']
        for trial in update['trials'] if 'projection' in trial]
    if (len(projections) != len(trial_projections)+2 or projections[1:-1] != trial_projections
        or projections[0].get('stage') != 'initial_projection'
        or projections[-1].get('stage') != 'final_projection'
        or projections[-1].get('stationarity_relative') != stationarity):
        raise ValueError('The stationary head-solve history or final checkpoint evidence differs')
    for projection in projections:
        residual = projection.get('stationarity_relative')
        if (projection.get('head_policy') != 'unconditional unique stationary ridge readout'
            or projection.get('hard_singular_value_truncation') is not False
            or projection.get('rank_rcond') is not None or projection.get('rank') != projection.get('columns')
            or type(projection.get('rank')) is not int
            or projection.get('columns') != len(expected['readout_calibration']['column_scale'])
            or projection.get('head_solve_calls') != 1
            or projection.get('fixed_lambda') != expected['readout_calibration']['fixed_lambda']
            or projection.get('column_scale_sha256') != expected['readout_calibration']['column_scale_sha256']
            or projection.get('stationarity_tolerance') != config['head_stationarity_tolerance']
            or type(residual) not in (int, float) or not math.isfinite(residual)
            or not 0 <= residual <= config['head_stationarity_tolerance']):
            raise ValueError('A saved head solve violated the fixed stationary regularized rule')
        after, old_loss, ridge, algebraic = (projection.get(key) for key in
            ('objective_after', 'unpenalized_objective', 'regularization_penalty', 'algebraic_objective'))
        if (not all(_nonnegative(value) for value in (after, old_loss, ridge, algebraic))
            or not _objective_close(after, old_loss+ridge) or not _objective_close(after, algebraic)
            or not _objective_close(after, projection.get('objective_proposed'))):
            raise ValueError('A saved stationary-head objective disagrees with its components or algebraic solve')
    previous_objective = projections[0]['objective_after']
    for event in raw['hidden_update_history']:
        if not _objective_close(event.get('objective_before'), previous_objective):
            raise ValueError('The hidden objective history is not linked to the preceding stationary head')
        accepted_trials = []
        for trial in event['trials']:
            if type(trial.get('head_solve_attempted')) is not bool or type(trial.get('accepted')) is not bool:
                raise ValueError('The trial readout or acceptance accounting is missing')
            if 'projection' in trial:
                if trial['head_solve_attempted'] is not True or not _objective_close(trial.get('objective'), trial['projection']['objective_after']):
                    raise ValueError('A trial objective differs from its stationary readout')
            if trial['accepted']:
                if 'projection' not in trial:
                    raise ValueError('An accepted trial lacks its stationary readout')
                accepted_trials.append(trial)
        if event['accepted']:
            if (len(accepted_trials) != 1 or accepted_trials[0] is not event['trials'][-1]
                or not _objective_close(event.get('objective_after'), accepted_trials[0].get('objective'))
                or event['objective_after'] > event['objective_before']):
                raise ValueError('An accepted hidden update is not bound to its decreasing full objective')
        elif accepted_trials or not _objective_close(event.get('objective_after'), event.get('objective_before')):
            raise ValueError('A rejected hidden update changed the retained full objective')
        previous_objective = event['objective_after']
    if (not _objective_close(total, projections[-1]['objective_after'])
        or not _objective_close(previous_objective, projections[-1]['objective_after'])):
        raise ValueError('The final training objective differs from the retained stationary checkpoint')
    return {'controller': controller, 'calibration_verified': True,
        'calibration_sha256': expected['calibration_sha256'],
        'worker_calibration_matrix_builds': 0, 'worker_spectral_calculations': 0,
        'full_regularized_objective': total, 'unpenalized_objective': unpenalized,
        'regularization_penalty': penalty, 'final_head_stationarity': stationarity,
        'gradient_semantics': raw.get('gradient_semantics'), 'unique_stationary_readout_verified': True}


def _nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _objective_close(left, right):
    return _nonnegative(left) and _nonnegative(right) and math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-8)


def regularized_callbacks():
    api = _api()
    def fit(observations, **kwargs):
        kwargs['config'] = api.RegularizedConfig(**kwargs['config'])
        return api.fit_regularized(observations, **kwargs)
    return common.ForwardCallbacks(fit=fit, predict=api.predict_regularized,
        diagnose=api.diagnose_regularized, checkpoint=api.checkpoint_bytes)


def start_execution(output_directory):
    cutoff._one_thread()
    directory = Path(output_directory).resolve()
    plan, state = _read(directory/'plan.json'), _read(directory/'state.json')
    verify_regularized_plan(plan, directory)
    if state.get('status') != 'prepared' or state.get('plan_sha256') != common.digest_file(directory/'plan.json'):
        raise ValueError('execution has already started or plan changed')
    descriptor = os.open(directory/'execution.lock', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    execution = {'status': 'running', 'started_monotonic': time.monotonic(),
        'maximum_seconds': TOP_SECONDS, 'plan_sha256': common.digest_file(directory/'plan.json')}
    common._write(directory/'execution.json', execution)
    common._write(directory/'state.json', {'status': 'running', 'plan_sha256': execution['plan_sha256']})
    return execution


def _seal_worker(directory, row):
    common._write(directory/'result.json', row)
    index = {str(path.relative_to(directory)): {'sha256': common.digest_file(path), 'bytes': path.stat().st_size}
        for path in sorted(directory.rglob('*')) if path.is_file() and path not in (directory/'state.json', directory/'artifact-index.json')}
    common._write(directory/'artifact-index.json', index)
    common._write(directory/'state.json', {'status': 'completed', 'scientific_status': row['status'],
        'plan_sha256': common.digest_file(directory/'plan.json'),
        'artifact_index_sha256': common.digest_file(directory/'artifact-index.json')})


def run_case(output_directory, case_id, callbacks=None):
    started = time.monotonic()
    import torch
    cutoff._one_thread()
    top = Path(output_directory).resolve()
    plan, case = _read(top/'plan.json'), _case(case_id)
    verify_regularized_plan(plan, top)
    execution = _read(top/'execution.json')
    if execution.get('status') != 'running' or execution.get('plan_sha256') != common.digest_file(top/'plan.json'):
        raise ValueError('a matching top execution must be started first')
    if case_id in WAVES[1] and any(_read(top/previous/'state.json').get('status') != 'completed' for previous in WAVES[0]):
        raise ValueError('second wave requires both first-wave workers to be terminal')
    directory = top/case_id
    if _read(directory/'state.json').get('status') != 'prepared':
        raise ValueError('case was already consumed')
    descriptor = os.open(directory/'run.lock', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    top_sha = common.digest_file(top/'plan.json')
    row = dict(case, status='failed')
    previous_alarm, previous_term = signal.getsignal(signal.SIGALRM), signal.getsignal(signal.SIGTERM)
    def guard(event=None):
        if time.monotonic()-started >= CASE_SECONDS or time.monotonic()-execution['started_monotonic'] >= TOP_SECONDS:
            raise common.ResourceLimit('declared single-case or aggregate wall cap reached')
        if common._rss_bytes() > RSS_BYTES:
            raise common.ResourceLimit('declared single-case RSS cap reached')
        if common.digest_file(top/'plan.json') != top_sha:
            raise RuntimeError('top plan changed during worker execution')
        if event is not None:
            with (directory/'progress.jsonl').open('a') as handle:
                handle.write(json.dumps(event, allow_nan=False)+'\n')
    def alarm(signum, frame):
        try:
            guard()
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            raise
    def terminate(signum, frame):
        signal.setitimer(signal.ITIMER_REAL, 0.)
        raise common.ResourceLimit('single-case worker terminated; no retry')
    common._write(directory/'state.json', {'status': 'running', 'pid': os.getpid(), 'started_monotonic': started,
        'plan_sha256': plan['workers'][case_id]['plan_sha256']})
    signal.signal(signal.SIGALRM, alarm)
    signal.signal(signal.SIGTERM, terminate)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    try:
        guard()
        physics = common.physics_for(case)
        observations = common.paired_observations(case, plan)
        source = plan['paired_evidence'][case_id]['observations.npz']['path']
        common._atomic(directory/'observations.npz', Path(source).read_bytes())
        callbacks = callbacks or regularized_callbacks()
        model, raw = callbacks.fit(observations, physics=physics, config=plan['model_config'], known_h=case['known_h'],
            neural_seed=common.SEEDS['neural'], sampling_seed=common.SEEDS['sampling'], progress=guard,
            readout_calibration=plan['preflight_by_case'][case_id]['readout_calibration'])
        common._atomic(directory/'final_checkpoint.pt', callbacks.checkpoint(model))
        common._write(directory/'training-report.json', raw)
        guard()
        proof = _training_proof(raw, plan, case_id)
        verify_regularized_plan(plan, top)
        diagnostics = callbacks.diagnose(model, common.independent_points(physics))
        common._write(directory/'independent-diagnostics.json', diagnostics)
        for label, points in (('near-interface', common.near_interface_points(physics)), ('near-bottom', graded.near_bottom_points())):
            audit = {'status': 'not_applicable', 'role': 'report only', 'point_digest': common.point_digest(points)}
            if points:
                common._npz(directory/(label+'-audit-points.npz'), **points)
                try:
                    audit.update(status='completed', diagnostics=callbacks.diagnose(model, points))
                    if label == 'near-bottom':
                        values = points['interior']
                        audit['by_height'] = [{'z_over_H': float(height), 'point_count': int((values[:, 1] == height).sum()),
                            'diagnostics': callbacks.diagnose(model, {'interior': values[values[:, 1] == height]})}
                            for height in np.unique(values[:, 1])]
                    common._json_bytes(audit)
                except common.ResourceLimit:
                    raise
                except Exception as error:
                    audit = {'status': 'failed', 'role': 'report only', 'error': repr(error)}
            common._write(directory/(label+'-audit.json'), audit)
        coordinates, truth = common.paired_scoring_field(case, plan)
        prediction, predicted_obs = np.asarray(callbacks.predict(model, coordinates)), np.asarray(callbacks.predict(model, observations.coordinates))
        if prediction.shape != truth.shape or predicted_obs.shape != observations.temperature.shape or not np.isfinite(prediction).all() or not np.isfinite(predicted_obs).all():
            raise FloatingPointError('invalid or nonfinite terminal prediction')
        field = {'rmse_K': float(np.sqrt(np.mean((prediction-truth)**2))), 'maximum_error_K': float(np.max(abs(prediction-truth)))}
        mse = float(np.mean(((predicted_obs-observations.temperature)/observations.standard_deviation)**2))
        gates = common.acceptance(physics, mse, diagnostics['normalized_rms'], field)
        common._npz(directory/'field_scoring.npz', coordinates=coordinates, truth=truth, prediction=prediction)
        verify_regularized_plan(plan, top)
        row = dict(case, status='completed', config=plan['model_config'], gate=gates, field_metrics=field,
            normalized_observation_mse=mse, independent_residuals=diagnostics,
            method_proof=proof, checkpoint_sha256=common.digest_file(directory/'final_checkpoint.pt'))
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        row = dict(case, status='failed', error=repr(error), traceback=traceback.format_exc())
        common._write(directory/'failure.json', row)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)
    row.update(wall_seconds=time.monotonic()-started, peak_rss_bytes=common._rss_bytes())
    _seal_worker(directory, row)
    return row


def mark_unrun(output_directory, case_id, reason):
    """The parent can retain a never-started opportunity at its aggregate cap."""
    top = Path(output_directory).resolve()
    verify_regularized_plan(_read(top/'plan.json'), top)
    if _read(top/'state.json').get('status') != 'running':
        raise ValueError('not-run recording requires an active top execution')
    directory = top/_case(case_id)['id']
    if _read(directory/'state.json').get('status') != 'prepared' or (directory/'run.lock').exists():
        raise ValueError('only an untouched case may be recorded as not run')
    row = dict(_case(case_id), status='not_run', reason=str(reason))
    _seal_worker(directory, row)
    return row


def record_worker_exit(output_directory, case_id, return_code, reason):
    """Seal an exited child that could not seal itself; never refit or overwrite.

    The supervising parent must call this only after ``Popen.wait``/``poll``
    observed the child's exit. A normally sealed child is simply validated.
    """
    top = Path(output_directory).resolve()
    plan = _read(top/'plan.json')
    verify_regularized_plan(plan, top)
    if _read(top/'state.json').get('status') != 'running' or type(return_code) is not int:
        raise ValueError('exit recording requires an active execution and an observed integer exit code')
    directory = top/_case(case_id)['id']
    state = _read(directory/'state.json')
    if state.get('status') == 'completed':
        return _validate_worker(directory)
    if state.get('status') not in ('prepared', 'running'):
        raise ValueError('worker state is not a declared opportunity')
    row = dict(_case(case_id), status='failed', return_code=return_code,
        reason=str(reason), terminal_recorded_by='supervisor after observed child exit',
        wall_seconds=max(0., time.monotonic()-state['started_monotonic']) if 'started_monotonic' in state else None,
        peak_rss_bytes=None)
    common._write(directory/'failure.json', row)
    _seal_worker(directory, row)
    return row


def _validate_worker(directory):
    state, index = _read(directory/'state.json'), _read(directory/'artifact-index.json')
    if (state.get('status') != 'completed' or state.get('artifact_index_sha256') != common.digest_file(directory/'artifact-index.json')
        or state.get('plan_sha256') != common.digest_file(directory/'plan.json')):
        raise ValueError('worker is not terminal or its index changed')
    for name, record in index.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size != record['bytes'] or common.digest_file(path) != record['sha256']:
            raise ValueError('worker evidence changed')
    actual = {str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file() and path not in (directory/'state.json', directory/'artifact-index.json')}
    if actual != set(index):
        raise ValueError('worker evidence is not completely indexed')
    row = _read(directory/'result.json')
    if state.get('scientific_status') is not None and state['scientific_status'] != row.get('status'):
        raise ValueError('worker result status differs from its terminal state')
    return row


def _validated_rows(directory, plan):
    rows = []
    for case in common.CASES:
        row = _validate_worker(directory/case['id'])
        if any(row.get(key) != value for key, value in case.items()) or row.get('status') not in ('completed', 'failed', 'not_run'):
            raise ValueError('saved worker case identity or outcome differs')
        if row['status'] == 'completed':
            proof = _training_proof(_read(directory/case['id']/'training-report.json'), plan, case['id'])
            if row.get('method_proof') != proof or row.get('checkpoint_sha256') != common.digest_file(directory/case['id']/'final_checkpoint.pt'):
                raise ValueError('saved worker training proof or retained checkpoint differs')
        rows.append(row)
    return rows


def finalize_regularized(output_directory):
    directory = Path(output_directory).resolve()
    plan, state = _read(directory/'plan.json'), _read(directory/'state.json')
    if state.get('status') != 'running':
        raise ValueError('regularized execution is not active or was already finalized')
    verify_regularized_plan(plan, directory)
    if state.get('plan_sha256') != common.digest_file(directory/'plan.json'):
        raise ValueError('top state refers to another plan')
    rows = _validated_rows(directory, plan)
    execution = _read(directory/'execution.json')
    if execution.get('status') != 'running' or execution.get('plan_sha256') != state['plan_sha256']:
        raise ValueError('top execution provenance differs')
    wall = time.monotonic()-execution['started_monotonic']
    result = {'status': 'completed', 'cases': rows, 'plan_sha256': common.digest_file(directory/'plan.json'),
        'method_fit_count': 4, 'reused_paired_datasets': 4, 'no_winner_selected': True,
        'initial_method_checks_passed': True, 'calibration_accounting': plan['calibration_accounting'],
        'source_unchanged': True, 'source_sha256': plan['source_sha256'], 'wall_seconds': wall,
        'peak_worker_rss_bytes': max((row.get('peak_rss_bytes') or 0 for row in rows), default=0) or None,
        'forward_status': 'passed' if all(row['status'] == 'completed' and row['gate']['all'] for row in rows) else 'blocked_forward_gate',
        'inverse_opened': False, 'fresh_evaluation_opened': False, 'three_d_opened': False}
    common._write(directory/'result.json', result)
    execution.update(status='completed', wall_seconds=wall)
    common._write(directory/'execution.json', execution)
    index = {str(path.relative_to(directory)): {'sha256': common.digest_file(path), 'bytes': path.stat().st_size}
        for path in sorted(directory.rglob('*')) if path.is_file() and path not in (directory/'state.json', directory/'artifact-index.json')}
    common._write(directory/'artifact-index.json', index)
    common._write(directory/'state.json', {'status': 'completed', 'plan_sha256': result['plan_sha256'],
        'artifact_index_sha256': common.digest_file(directory/'artifact-index.json')})
    return result


def validate_regularized(output_directory):
    cutoff._one_thread()
    directory = Path(output_directory).resolve()
    result = _validate_worker(directory)
    plan = _read(directory/'plan.json')
    verify_regularized_plan(plan, directory)
    if result.get('plan_sha256') != common.digest_file(directory/'plan.json'):
        raise ValueError('top result refers to a different controller plan')
    if (result.get('cases') != _validated_rows(directory, plan) or result.get('method_fit_count') != 4
        or result.get('reused_paired_datasets') != 4 or result.get('source_sha256') != plan['source_sha256']
        or result.get('no_winner_selected') is not True or result.get('source_unchanged') is not True
        or result.get('initial_method_checks_passed') is not True
        or result.get('calibration_accounting') != plan['calibration_accounting']):
        raise ValueError('top result does not preserve the complete declared denominator or source')
    return {'plan_sha256': common.digest_file(directory/'plan.json'),
        'artifact_index_sha256': common.digest_file(directory/'artifact-index.json'), 'method_fit_count': 4,
        'initial_method_checks_passed': True, 'preparation_calibrations': 4,
        'worker_calibrations': 0}



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    for name in ('output-directory', 'config', 'method-decision', 'gradient-probe-validator',
                 'step-control-campaign', 'cutoff-audit-directory'):
        prepare.add_argument('--'+name, required=True)
    for name in ('source-path', 'protected-file'):
        prepare.add_argument('--'+name, action='append', required=True)
    for command in ('start', 'run-case', 'not-run', 'record-exit', 'finalize', 'validate'):
        child = commands.add_parser(command)
        child.add_argument('--output-directory', required=True)
        if command in ('run-case', 'not-run', 'record-exit'):
            child.add_argument('--case-id', choices=[case['id'] for case in common.CASES], required=True)
        if command in ('not-run', 'record-exit'):
            child.add_argument('--reason', required=True)
        if command == 'record-exit':
            child.add_argument('--return-code', type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare_regularized(args.output_directory, model_config=_read(args.config),
            method_decision_path=args.method_decision, gradient_probe_validator_path=args.gradient_probe_validator,
            source_paths=args.source_path, step_control_campaign=args.step_control_campaign,
            cutoff_audit_directory=args.cutoff_audit_directory, protected_files=args.protected_file)
    elif args.command == 'start':
        result = start_execution(args.output_directory)
    elif args.command == 'run-case':
        result = run_case(args.output_directory, args.case_id)
    elif args.command == 'not-run':
        result = mark_unrun(args.output_directory, args.case_id, args.reason)
    elif args.command == 'record-exit':
        result = record_worker_exit(args.output_directory, args.case_id, args.return_code, args.reason)
    elif args.command == 'finalize':
        result = finalize_regularized(args.output_directory)
    else:
        result = validate_regularized(args.output_directory)
    print(json.dumps({'command': args.command, 'status': result.get('status', 'prepared' if args.command == 'prepare' else 'validated')}), flush=True)


if __name__ == '__main__':
    main()
