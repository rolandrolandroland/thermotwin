"""Four isolated, bounded forward checks for a training-only step controller.

Each case runs once in its own main process. The declared two waves ensure a
resource failure cannot consume another case's opportunity. Every diagnostic
uses the retained final state; early termination uses training mechanics only.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from . import pinn_spatial_continuity as common
from . import pinn_spatial_graded as graded
from . import pinn_spatial_optimized as optimized

ROOT = common.ROOT
CASE_SECONDS = 3000.
TOP_SECONDS = 7200.
RSS_BYTES = 2*1024**3
WAVES = (('t01', 't02'), ('t03', 't04'))
STOPS = ('fixed_budget_exhausted', 'consecutive_failure_stop',
         'learning_rate_floor_stop', 'unchanged_proposal_stop', 'projection_budget')


def _read(path):
    return json.loads(Path(path).read_text())


def _api():
    from ..pinn import planar_contact_step_control
    return planar_contact_step_control


def _case(case_id):
    matches = [case for case in common.CASES if case['id'] == case_id]
    if len(matches) != 1:
        raise ValueError('unknown forward case')
    return matches[0]


def _config(values, prior):
    config = asdict(_api().StepControlConfig(**values))
    if set(config) != set(values):
        raise ValueError('the full controller configuration must be explicitly supplied')
    if (type(config['adam_epochs']) is not int or not 0 < config['adam_epochs'] <= 320
        or not 0 < config['maximum_trial_solves'] <= 480):
        raise ValueError('controller budgets exceed the declared 320 attempts / 480 trial solves')
    for key, value in prior.items():
        if key != 'adam_epochs' and config.get(key) != value:
            raise ValueError('controller changed frozen architecture, grid, cutoff, objective, or initial optimizer: '+key)
    return config


def _preflight(config, prior_plan):
    import torch
    torch.set_num_threads(1)
    api, grid_api = _api(), graded._model_api()
    instance = api.StepControlConfig(**config)
    result = {}
    for case in common.CASES:
        physics = common.physics_for(case)
        _, initialization = api.matched_step_control_model(physics, instance, case['known_h'], common.SEEDS['neural'])
        grid = grid_api.graded_point_metadata(physics, instance, common.SEEDS['sampling'])
        previous = prior_plan['optimization_provenance']['preflight_by_case'][case['id']]
        if initialization['common_weights_sha256'] != previous['initialization']['common_weights_sha256']:
            raise ValueError('controller initialization changed original common weights')
        if initialization.get('log_column_present') is not False or initialization['probe_agreement']['passed'] is not True:
            raise ValueError('controller initialization is not the verified original log-off field')
        if grid != previous['training_grid']:
            raise ValueError('controller changed the graded points or their weights')
        result[case['id']] = {'initialization': initialization, 'training_grid': grid}
    return result


def prepare_step_control(output_directory, *, model_config, probe_decision_path,
        source_paths, optimized_campaign, protected_files):
    """Freeze all four opportunities before any worker may start."""
    prior = Path(optimized_campaign).resolve()
    prior_context = optimized.validate_optimized(prior)
    prior_plan = _read(prior/'plan.json')
    config = _config(model_config, prior_plan['model_config'])
    decision = optimized._decision(probe_decision_path, config)
    records = _preflight(config, prior_plan)  # no optimization or hidden fields
    sources = set(source_paths) | set(prior_plan['source_sha256']) | {
        'thermotwin/pinn/planar_contact_step_control.py',
        'thermotwin/studies/pinn_spatial_step_control.py', 'tests/test_pinn_spatial_step_control.py'}
    source_records = common._source_records(sources)
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    workers = {}
    for case in common.CASES:
        worker = directory/case['id']
        worker.mkdir()
        payload = {'schema': 'step_control_single_case_worker_v7', 'case': case,
            'model_config_digest': common.canonical_digest(config), 'seeds': common.SEEDS,
            'source_digest': common.canonical_digest(source_records),
            'maximum_wall_seconds': CASE_SECONDS, 'maximum_rss_bytes': RSS_BYTES,
            'retries': 0, 'threads': 1}
        common._write(worker/'plan.json', payload)
        workers[case['id']] = {'directory': case['id'], 'plan_sha256': common.digest_file(worker/'plan.json')}
        common._write(worker/'state.json', {'status': 'prepared', 'plan_sha256': workers[case['id']]['plan_sha256']})
    plan = {'schema': 'bounded_step_control_forward_v7',
        'candidate_id': 'training-only-step-controller-v7-log-off',
        'cases': list(common.CASES), 'model_config': config, 'source_sha256': source_records,
        'runtime': common.runtime_context(), 'seeds': common.SEEDS, 'gates': common.GATES,
        'workers': workers, 'waves': [list(wave) for wave in WAVES],
        'maximum_method_fits': 4, 'reused_paired_datasets': 4,
        'independent_noise_replications_claimed': False,
        'topology': {'maximum_concurrent_workers': 2, 'threads_per_worker': 1,
            'rss_bytes_per_worker': RSS_BYTES, 'aggregate_child_rss_bytes': 2*RSS_BYTES,
            'seconds_per_worker': CASE_SECONDS, 'top_seconds': TOP_SECONDS},
        'allowed_training_stop_reasons': list(STOPS), 'automatic_retries': 0,
        'paired_evidence': prior_plan['paired_evidence'],
        'step_control_provenance': {'graded_comparison': prior_plan['optimization_provenance']['graded_comparison'],
            'prior_optimized_campaign': {'directory': str(prior), 'context': prior_context,
                'plan_sha256': common.digest_file(prior/'plan.json')}},
        'preserved_v6_attempt': {'directory': str(prior), 'evidence': prior_context,
            'plan_sha256': common.digest_file(prior/'plan.json'), 'files': common._tree_records(prior)},
        'protected_files': common._source_records(protected_files),
        'decision': decision, 'preflight_by_case': records,
        'near_bottom_audit_digest': common.point_digest(graded.near_bottom_points()),
        'checkpoint_rule': 'training-returned terminal state retained before any independent diagnostic or hidden-field score',
        'initialization_policy': 'fresh original common weights; no probe or prior-campaign continuation',
        'selection_boundary': 'training-only controller termination; unchanged acceptance; no winner or reserved evaluation'}
    common._write(directory/'plan.json', plan)
    for path in source_records:
        target = directory/'source-snapshot'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/path).read_bytes())
    common._write(directory/'state.json', {'status': 'prepared', 'plan_sha256': common.digest_file(directory/'plan.json')})
    verify_step_control_plan(plan, directory)
    return plan


def verify_step_control_plan(plan, output_directory=None):
    if (plan.get('schema') != 'bounded_step_control_forward_v7' or plan.get('cases') != list(common.CASES)
        or plan.get('seeds') != common.SEEDS or plan.get('gates') != common.GATES
        or plan.get('waves') != [list(wave) for wave in WAVES]
        or plan.get('maximum_method_fits') != 4 or plan.get('automatic_retries') != 0
        or plan.get('reused_paired_datasets') != 4 or plan.get('independent_noise_replications_claimed') is not False
        or plan.get('allowed_training_stop_reasons') != list(STOPS)):
        raise ValueError('step-control scope, waves, streams, gates, or stops changed')
    expected_topology = {'maximum_concurrent_workers': 2, 'threads_per_worker': 1,
        'rss_bytes_per_worker': RSS_BYTES, 'aggregate_child_rss_bytes': 2*RSS_BYTES,
        'seconds_per_worker': CASE_SECONDS, 'top_seconds': TOP_SECONDS}
    if plan.get('topology') != expected_topology:
        raise ValueError('step-control resource topology changed')
    if common._source_records(plan['source_sha256']) != plan['source_sha256']:
        raise RuntimeError('step-control numerical source changed')
    if common.runtime_context() != plan['runtime'] or common._source_records(plan['protected_files']) != plan['protected_files']:
        raise RuntimeError('step-control runtime or prior tracked record changed')
    prior = plan['preserved_v6_attempt']
    if optimized.validate_optimized(prior['directory']) != prior['evidence']:
        raise RuntimeError('the preserved V6 attempt or older chain changed')
    if common._tree_records(prior['directory']) != prior['files'] or common.digest_file(Path(prior['directory'])/'plan.json') != prior['plan_sha256']:
        raise RuntimeError('the complete V6 attempt changed')
    old_plan = _read(Path(prior['directory'])/'plan.json')
    _config(plan['model_config'], old_plan['model_config'])
    expected_provenance = {'graded_comparison': old_plan['optimization_provenance']['graded_comparison'],
        'prior_optimized_campaign': {'directory': prior['directory'], 'context': prior['evidence'], 'plan_sha256': prior['plan_sha256']}}
    if plan.get('step_control_provenance') != expected_provenance or plan.get('paired_evidence') != old_plan['paired_evidence']:
        raise ValueError('prior-campaign provenance or paired evidence changed')
    if optimized._decision(plan['decision']['path'], plan['model_config']) != plan['decision']:
        raise RuntimeError('training-only repair decision or probe changed')
    if common.point_digest(graded.near_bottom_points()) != plan['near_bottom_audit_digest']:
        raise RuntimeError('report-only near-bottom points changed')
    if output_directory is not None:
        directory = Path(output_directory).resolve()
        if _read(directory/'plan.json') != plan:
            raise RuntimeError('step-control plan changed on disk')
        for name, expected in plan['source_sha256'].items():
            if common.digest_file(directory/'source-snapshot'/name) != expected:
                raise RuntimeError('step-control pre-fit source snapshot changed')
        for case in common.CASES:
            record = plan['workers'][case['id']]
            if record['directory'] != case['id']:
                raise ValueError('single-case directory differs from the declared case')
            path = directory/record['directory']/'plan.json'
            if common.digest_file(path) != record['plan_sha256']:
                raise RuntimeError('single-case plan changed')
            worker = _read(path)
            expected = {'schema': 'step_control_single_case_worker_v7', 'case': case,
                'model_config_digest': common.canonical_digest(plan['model_config']), 'seeds': common.SEEDS,
                'source_digest': common.canonical_digest(plan['source_sha256']),
                'maximum_wall_seconds': CASE_SECONDS, 'maximum_rss_bytes': RSS_BYTES,
                'retries': 0, 'threads': 1}
            if worker != expected:
                raise RuntimeError('single-case rule differs from the top declaration')
    return {'method_fit_count': 4, 'reused_dataset_count': 4, 'source_sha256': plan['source_sha256']}


def start_execution(output_directory):
    directory = Path(output_directory).resolve()
    plan, state = _read(directory/'plan.json'), _read(directory/'state.json')
    verify_step_control_plan(plan, directory)
    if state.get('status') != 'prepared' or state.get('plan_sha256') != common.digest_file(directory/'plan.json'):
        raise ValueError('execution has already started or plan changed')
    descriptor = os.open(directory/'execution.lock', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    execution = {'status': 'running', 'started_monotonic': time.monotonic(),
        'maximum_seconds': TOP_SECONDS, 'plan_sha256': common.digest_file(directory/'plan.json')}
    common._write(directory/'execution.json', execution)
    common._write(directory/'state.json', {'status': 'running', 'plan_sha256': execution['plan_sha256']})
    return execution


def _training_proof(raw, plan, case_id):
    config = plan['model_config']
    history, transitions = raw.get('hidden_update_history', []), raw.get('controller_history', [])
    count = len(history)
    if raw.get('config') != config or raw.get('requested_updates') != config['adam_epochs']:
        raise ValueError('controller configuration or requested budget differs')
    if raw.get('completed_updates') != count or not 0 <= count <= config['adam_epochs'] or len(transitions) != count:
        raise ValueError('controller update and transition counts differ')
    reason = raw.get('stop_reason')
    if reason not in STOPS or (reason == 'fixed_budget_exhausted' and count != config['adam_epochs']):
        raise ValueError('undeclared or inconsistent training-only stop')
    trials = raw.get('trial_solve_count')
    if (type(trials) is not int or not 0 <= trials <= config['maximum_trial_solves']
        or raw.get('additional_initial_final_solves') != 2
        or raw.get('total_projection_solve_calls') != trials+2):
        raise ValueError('controller projection-solve accounting differs')
    initial_state = asdict(_api().StepControlState(config['learning_rate']))
    previous = initial_state
    total_trials = 0
    for step, (event, transition) in enumerate(zip(history, transitions), 1):
        before, after = transition.get('before'), transition.get('after')
        if total_trials+config['max_backtracks']+1 > config['maximum_trial_solves'] or previous['stopped']:
            raise ValueError('controller attempted an update after its training-only stop')
        if (transition != {'step': step, 'before': event.get('controller_before'),
                'after': event.get('controller_after'), 'transitions': event.get('controller_transitions')}
            or before != previous or event.get('step') != step or after.get('attempts') != step):
            raise ValueError('controller transition chain differs from actual updates')
        used = len(event.get('trials', [])) if event.get('proposed_hidden_displacement_finite') and event.get('proposed_optimizer_state_finite') else 0
        total_trials += used
        if (event.get('trial_solve_count') != used or used > config['max_backtracks']+1
            or after.get('trial_solves') != total_trials
            or not math.isfinite(after['learning_rate'])
            or not config['controller_minimum_lr'] <= after['learning_rate'] <= config['learning_rate']):
            raise ValueError('controller trial accounting or learning-rate bounds differ')
        if not event.get('accepted') and not (after['stopped'] or after['learning_rate'] < before['learning_rate']):
            raise ValueError('controller would repeat an unchanged rejected proposal')
        previous = after
    final = dict(previous)
    human_stops = {'consecutive_failure_stop': 'consecutive exhausted searches',
        'learning_rate_floor_stop': 'learning-rate floor reached',
        'unchanged_proposal_stop': 'controller cannot change the next proposal', 'projection_budget': 'projection budget'}
    if reason == 'projection_budget':
        if trials+config['max_backtracks']+1 <= config['maximum_trial_solves'] or previous['stopped']:
            raise ValueError('projection budget stop did not exhaust the declared full-search reservation')
        final.update(stopped=True, stop_reason='projection budget')
    if (total_trials != trials or raw.get('final_controller_state') != final
        or final['attempts'] != count or final['trial_solves'] != trials
        or (reason == 'fixed_budget_exhausted' and (final['stopped'] or final['stop_reason'] is not None))
        or (reason != 'fixed_budget_exhausted' and (not final['stopped'] or final['stop_reason'] != human_stops[reason]))
        or raw.get('optimizer_history') != config['optimizer_history'] or raw.get('optimizer_instances') != 1):
        raise ValueError('controller terminal state, stop reason, or optimizer mechanics differ')
    if ((reason == 'consecutive_failure_stop' and final['consecutive_failures'] < config['controller_max_failures'])
        or (reason == 'learning_rate_floor_stop' and (not transitions
            or transitions[-1]['before']['learning_rate'] != config['controller_minimum_lr']))
        or (reason == 'unchanged_proposal_stop' and (not transitions
            or final['learning_rate'] < transitions[-1]['before']['learning_rate']))):
        raise ValueError('controller early stop lacks its declared training-only condition')
    expected = plan['preflight_by_case'][case_id]
    actual, declared = raw.get('initialization') or {}, expected['initialization']
    if ({key: value for key, value in actual.items() if key != 'probe_agreement'}
        != {key: value for key, value in declared.items() if key != 'probe_agreement'}
        or (actual.get('probe_agreement') or {}).get('passed') is not True):
        raise ValueError('controller did not start from the declared fresh original weights')
    grid = expected['training_grid']
    if raw.get('graded_point_metadata') != grid:
        raise ValueError('controller graded-grid metadata differs')
    identities, counts = {}, {}
    for name, values in (raw.get('training_points') or {}).items():
        values = np.asarray(values, dtype=np.float64)
        identities[name] = {'shape': list(values.shape), 'sha256': graded._model_api()._array_digest(values)}
        counts[name] = len(values)
        if not np.array_equal(np.asarray((raw.get('training_point_weights') or {}).get(name)), np.full(len(values), 1/len(values))):
            raise ValueError('controller changed frozen row weights')
    if identities != grid['point_identities'] or counts != grid['point_counts']:
        raise ValueError('controller changed actual graded coordinates')
    return {'verified': True, 'actual_updates': count, 'stop_reason': reason, 'trial_solves': trials}


def step_control_callbacks():
    api = _api()
    def fit(observations, **kwargs):
        kwargs['config'] = api.StepControlConfig(**kwargs['config'])
        return api.fit_step_control(observations, **kwargs)
    return common.ForwardCallbacks(fit=fit, predict=api.predict_step_control,
        diagnose=api.diagnose_step_control, checkpoint=api.checkpoint_bytes)


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
    torch.set_num_threads(1)
    top = Path(output_directory).resolve()
    plan, case = _read(top/'plan.json'), _case(case_id)
    verify_step_control_plan(plan, top)
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
        callbacks = callbacks or step_control_callbacks()
        model, raw = callbacks.fit(observations, physics=physics, config=plan['model_config'], known_h=case['known_h'],
            neural_seed=common.SEEDS['neural'], sampling_seed=common.SEEDS['sampling'], progress=guard)
        common._atomic(directory/'final_checkpoint.pt', callbacks.checkpoint(model))
        common._write(directory/'training-report.json', raw)
        guard()
        proof = _training_proof(raw, plan, case_id)
        verify_step_control_plan(plan, top)
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
        verify_step_control_plan(plan, top)
        row = dict(case, status='completed', config=plan['model_config'], gate=gates, field_metrics=field,
            normalized_observation_mse=mse, independent_residuals=diagnostics,
            controller_proof=proof, checkpoint_sha256=common.digest_file(directory/'final_checkpoint.pt'))
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
    verify_step_control_plan(_read(top/'plan.json'), top)
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
    verify_step_control_plan(plan, top)
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
            if row.get('controller_proof') != proof or row.get('checkpoint_sha256') != common.digest_file(directory/case['id']/'final_checkpoint.pt'):
                raise ValueError('saved worker training proof or retained checkpoint differs')
        rows.append(row)
    return rows


def finalize_step_control(output_directory):
    directory = Path(output_directory).resolve()
    plan, state = _read(directory/'plan.json'), _read(directory/'state.json')
    if state.get('status') != 'running':
        raise ValueError('step-control execution is not active or was already finalized')
    verify_step_control_plan(plan, directory)
    if state.get('plan_sha256') != common.digest_file(directory/'plan.json'):
        raise ValueError('top state refers to another plan')
    rows = _validated_rows(directory, plan)
    execution = _read(directory/'execution.json')
    if execution.get('status') != 'running' or execution.get('plan_sha256') != state['plan_sha256']:
        raise ValueError('top execution provenance differs')
    wall = time.monotonic()-execution['started_monotonic']
    result = {'status': 'completed', 'cases': rows, 'plan_sha256': common.digest_file(directory/'plan.json'),
        'method_fit_count': 4, 'reused_paired_datasets': 4, 'no_winner_selected': True,
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


def validate_step_control(output_directory):
    directory = Path(output_directory).resolve()
    result = _validate_worker(directory)
    plan = _read(directory/'plan.json')
    verify_step_control_plan(plan, directory)
    if result.get('plan_sha256') != common.digest_file(directory/'plan.json'):
        raise ValueError('top result refers to a different controller plan')
    if (result.get('cases') != _validated_rows(directory, plan) or result.get('method_fit_count') != 4
        or result.get('reused_paired_datasets') != 4 or result.get('source_sha256') != plan['source_sha256']
        or result.get('no_winner_selected') is not True or result.get('source_unchanged') is not True):
        raise ValueError('top result does not preserve the complete declared denominator or source')
    return {'plan_sha256': common.digest_file(directory/'plan.json'),
        'artifact_index_sha256': common.digest_file(directory/'artifact-index.json'), 'method_fit_count': 4}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    for name in ('output-directory', 'config', 'probe-decision', 'optimized-campaign'):
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
        result = prepare_step_control(args.output_directory, model_config=_read(args.config),
            probe_decision_path=args.probe_decision, source_paths=args.source_path,
            optimized_campaign=args.optimized_campaign, protected_files=args.protected_file)
    elif args.command == 'start':
        result = start_execution(args.output_directory)
    elif args.command == 'run-case':
        result = run_case(args.output_directory, args.case_id)
    elif args.command == 'not-run':
        result = mark_unrun(args.output_directory, args.case_id, args.reason)
    elif args.command == 'record-exit':
        result = record_worker_exit(args.output_directory, args.case_id, args.return_code, args.reason)
    elif args.command == 'finalize':
        result = finalize_step_control(args.output_directory)
    else:
        result = validate_step_control(args.output_directory)
    print(json.dumps({'command': args.command, 'status': result.get('status', 'prepared' if args.command == 'prepare' else 'validated')}), flush=True)


if __name__ == '__main__':
    main()
