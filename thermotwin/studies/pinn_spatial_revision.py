"""Strict stage gates and fresh-data execution for the additive spatial revision.

Development evidence is assembled before a commit; named observations require
the resulting complete protocol in a clean, CI-passing source clone.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import traceback
import time

import numpy as np
from scipy.optimize import least_squares

from .pinn_spatial_validation import (
    ROOT, canonical_digest, digest_file, now, runtime_context, source_context, verify_ci, write_json,
    validate_evidence as validate_original_evidence,
)
from .pinn_spatial_revision_inverse import (
    INVERSE_SPEC, fit_delayed_inverse, numerical_preflight, coordinates_and_reference,
    observations_from_reference, score_inverse, semantic_seed, field_metrics,
    retain_checkpoint, retain_stage_evidence,
)
from .pinn_spatial_revision_1d import SPEC as ONE_D_SPEC, run_evaluation as run_one_d_evaluation
from ..pinn.planar_contact_revision import PlanarRevisionConfig
from ..physics.planar_contact import PlanarContactModel

PROTOCOL_PATH = ROOT/'thermotwin/PINN_SPATIAL_REVISION_V2_PROTOCOL.json'
CAMPAIGN = 'pinn_spatial_revision_v2_2026_10_reserved_1'
RUNTIME = {'implementation': 'CPython', 'python': '3.10.12',
           'packages': {'numpy': '2.2.6', 'scipy': '1.15.3', 'torch': '2.9.1', 'matplotlib': '3.10.8'}}


def preserved_context(campaign, *, expected=None):
    """Anchor old indexed evidence, old source records, and Phase F separately."""
    campaign = Path(campaign)
    indexed = validate_original_evidence(campaign)
    manifest = json.loads((campaign/'manifest.json').read_text())
    records = manifest['source_files']
    for item in records:
        if digest_file(ROOT/item['path']) != item['sha256']:
            raise RuntimeError('preserved v1 source/protocol changed: '+item['path'])
    gate_path = ROOT/'thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_GENERATION_GATE.json'
    gate = json.loads(gate_path.read_text())
    phase_f = gate['source_manifest']['files']
    if len(phase_f) != 117:
        raise RuntimeError('unexpected Phase F preservation manifest')
    for item in phase_f:
        if digest_file(ROOT/item['path']) != item['sha256']:
            raise RuntimeError('frozen Phase F source changed: '+item['path'])
    result = {'v1_indexed_evidence': indexed, 'v1_manifest_sha256': digest_file(campaign/'manifest.json'),
              'v1_source_record_count': len(records), 'v1_source_records_digest': canonical_digest(records),
              'phase_f_record_count': len(phase_f), 'phase_f_records_digest': canonical_digest(phase_f),
              'phase_f_gate_sha256': digest_file(gate_path)}
    if expected is not None and result != expected:
        raise RuntimeError('preserved evidence identities differ from the frozen protocol')
    return result


def verify_selection_sources(selection):
    records = selection.get('source_sha256')
    if not records:
        raise ValueError('selection lacks frozen source hashes')
    for path, value in records.items():
        if digest_file(ROOT/path) != value:
            raise ValueError('development selection source changed: '+path)


def verified_forward_selection(selection):
    if selection.get('status') != 'passed' or not selection.get('selected_config'):
        raise ValueError('known-h forward development has not passed')
    verify_selection_sources(selection)
    config = PlanarRevisionConfig(**selection['selected_config'])
    required = {(geometry, h) for geometry in ('full_contact', 'partial_contact') for h in (800., 1400.)}
    matched = set()
    for case in selection.get('cases', []):
        gate = case.get('gate', {})
        if canonical_digest(case.get('config')) == canonical_digest(asdict(config)) and gate.get('all') is True:
            matched.add((case['geometry'], float(case['known_h'])))
    if not required.issubset(matched):
        raise ValueError('the same selected configuration lacks all four full/partial forward gates')
    return config


def verified_inverse_selection(selection, forward_config):
    """Recheck the accepted inverse proof against its numerical source and field rule."""
    if (not isinstance(selection, dict) or selection.get('status') != 'passed'
        or selection.get('selected_strategy') not in INVERSE_SPEC['strategies']):
        raise ValueError('inverse development is not accepted')
    required_sources = {'thermotwin/studies/pinn_spatial_revision_inverse.py',
                        'thermotwin/pinn/planar_contact_revision.py',
                        'thermotwin/physics/planar_contact.py',
                        'thermotwin/pinn/planar_contact_inverse.py'}
    if set(selection.get('source_sha256', {})) != required_sources:
        raise ValueError('inverse development lacks the complete numerical source proof')
    verify_selection_sources(selection)
    if selection.get('source_unchanged') is not True:
        raise ValueError('inverse development changed source while fitting')
    if canonical_digest(selection.get('forward_config')) != canonical_digest(asdict(forward_config)):
        raise ValueError('inverse and forward selected configurations differ')
    strategy = selection['selected_strategy']
    rows = [row for row in selection.get('rows', []) if row.get('strategy') == strategy]
    if (len(rows) != 2 or {row.get('development_h') for row in rows} != {800., 1400.}
        or not all(row.get('passed') is True and row.get('score', {}).get('passed') is True for row in rows)):
        raise ValueError('selected inverse rule lacks its two accepted development cases')
    return strategy, rows


def verified_frozen_inverse(protocol, forward_config):
    """Bind the execution rule to the complete retained development evidence."""
    selected = protocol.get('selected_inverse', {})
    proof = selected.get('development_selection')
    strategy, rows = verified_inverse_selection(proof, forward_config)
    if (selected.get('strategy') != strategy
        or canonical_digest(selected.get('config')) != canonical_digest(asdict(forward_config))):
        raise ValueError('frozen inverse execution rule differs from accepted development')
    if selected.get('development_evidence_digest') != canonical_digest(proof):
        raise ValueError('frozen inverse development evidence digest differs')
    scorecards = [{'h': row['development_h'], 'score': row['score']} for row in rows]
    if canonical_digest(selected.get('development_scorecards')) != canonical_digest(scorecards):
        raise ValueError('frozen inverse development scorecards differ')


def build_frozen_protocol(*, one_d_selection, forward_selection, inverse_selection, preserved_campaign):
    """Construct a reviewable complete rule; no fresh dataset is opened."""
    config = verified_forward_selection(forward_selection)
    if (one_d_selection.get('status') != 'selected'
        or not one_d_selection.get('candidate_score', {}).get('passed')
        or not all(row['passed'] for row in one_d_selection.get('selected_fd_refinement', []))):
        raise ValueError('1D information-design development is not accepted')
    verify_selection_sources(one_d_selection)
    if one_d_selection['spec_sha256'] != canonical_digest(ONE_D_SPEC):
        raise ValueError('1D selected specification differs')
    selected_strategy, selected_rows = verified_inverse_selection(inverse_selection, config)
    protocol = {'schema_version': 1, 'campaign': CAMPAIGN,
        'selected_forward': forward_selection, 'selected_inverse': {
            'strategy': selected_strategy, 'config': asdict(config),
            'development_selection': inverse_selection,
            'development_evidence_digest': canonical_digest(inverse_selection),
            'development_scorecards': [{'h': row['development_h'], 'score': row['score']} for row in selected_rows]},
        'one_d_selection': one_d_selection, 'inverse_spec': INVERSE_SPEC, 'one_d_spec': ONE_D_SPEC,
        'preservation': preserved_context(preserved_campaign),
        'chronology': {'forward_before_inverse_development': True, 'all_selections_before_fresh_generation': True,
                       'committed_complete_rule_and_passing_exact_source_ci_before_fresh_generation': True},
        'execution': {'runtime_requirements': RUNTIME, 'threads': 1, 'rss_limit_bytes': 2*1024**3,
                      'wall_limit_seconds': 21600 if selected_strategy == 'alternating' else 10800,
                      'automatic_restart': False, 'retries': 0},
        'planned_denominators': {'two_d': 6, 'one_d': 6, 'one_d_method_fits': 12,
                                'one_d_independent_paired_noise_blocks': 3},
        'failure_policy': 'all planned rows retained; exceptions and gate failures distinct; no replacement trials',
        '3d': 'deferred: uniform-depth plate needs 2D; no new geometry supplied'}
    protocol['seed_schedule'] = seed_schedule(protocol)
    return protocol


def seed_schedule(protocol):
    from ..pinn.planar_contact_revision import _derived_seed
    from .pinn_spatial_revision_1d import _seed
    config = protocol['selected_inverse']['config']
    strategy = protocol['selected_inverse']['strategy']
    uses = []
    for case in range(6):
        namespace = f'{CAMPAIGN}/{case}'
        uses.append((f'two_d/{case}/noise', semantic_seed(CAMPAIGN, case, 'observation_noise')))
        uses.append((f'two_d/{case}/initial_field', semantic_seed(namespace, 0, 'neural')))
        stages = [('warm', semantic_seed(namespace, 0, 'sampling'), semantic_seed(namespace, 0, 'independent'), config['adam_epochs'])]
        if strategy == 'delayed_release':
            stages.append(('joint', semantic_seed(namespace, 1, 'sampling'), semantic_seed(namespace, 1, 'independent'), config['adam_epochs']))
        else:
            for cycle in range(INVERSE_SPEC['alternating_cycles']):
                stages.extend([(f'contact_{cycle}', semantic_seed(namespace, cycle, 'contact_sampling'),
                    semantic_seed(namespace, cycle, 'contact_independent'), INVERSE_SPEC['contact_only_adam_epochs']),
                    (f'field_{cycle}', semantic_seed(namespace, cycle, 'field_sampling'),
                     semantic_seed(namespace, cycle, 'field_independent'), config['adam_epochs'])])
        for stage, sampling, independent, epochs in stages:
            uses.append((f'two_d/{case}/{stage}/diagnostic', independent))
            for index in range((epochs-1)//config['refresh_adam_every']+1):
                uses.append((f'two_d/{case}/{stage}/adam/{index}', _derived_seed(sampling, 'adam', index)))
            uses.append((f'two_d/{case}/{stage}/lbfgs', _derived_seed(sampling, 'lbfgs', 0)))
    for trial in range(3):
        for experiment in range(protocol['one_d_selection']['candidate']['experiment_count']):
            uses.append((f'one_d/paired_family_noise/{trial}/{experiment}', _seed(CAMPAIGN+'/one-d', trial, experiment)))
    if len({seed for _, seed in uses}) != len(uses):
        raise ValueError('fresh random-stream collision; no observations may be generated')
    return {'streams': [{'semantic_key': key, 'seed': seed} for key, seed in uses], 'all_unique': True,
            'intentional_pairing': '1D matched/cubic families reuse each paired noise block; methods receive identical observed data; warm model weights continue across stages'}


def validated_revision_protocol():
    if not PROTOCOL_PATH.is_file():
        raise ValueError('v2 complete protocol is not frozen; accepted forward and inverse development are required before evaluation')
    protocol = json.loads(PROTOCOL_PATH.read_text())
    normalize = lambda value: json.loads(json.dumps(value, allow_nan=False))
    if (protocol.get('campaign') != CAMPAIGN or protocol['inverse_spec'] != normalize(INVERSE_SPEC)
        or protocol['one_d_spec'] != normalize(ONE_D_SPEC)):
        raise ValueError('frozen complete rule differs from the numerical implementation')
    forward_config = verified_forward_selection(protocol['selected_forward'])
    verified_frozen_inverse(protocol, forward_config)
    verify_selection_sources(protocol['one_d_selection'])
    if protocol['seed_schedule'] != seed_schedule(protocol):
        raise ValueError('frozen random-stream schedule differs')
    return protocol


def conventional_fit(observations):
    started, cpu = time.monotonic(), time.process_time()
    def prediction(h):
        trajectory = PlanarContactModel(nx=120, nz=30, contact_conductance=float(h)).solve(
            INVERSE_SPEC['field_times_s'], initial_temperature=300., max_step=.0625,
            te_temperature=lambda t: 300.-5.*(-np.expm1(-t/10.)),
            top_heat_flux=lambda t: 500.*(-np.expm1(-t/10.)))
        return trajectory.sample(*observations.coordinates.T)
    def residual(log_h):
        return (prediction(math.exp(float(log_h[0])))-observations.temperature)/observations.standard_deviation
    fit = least_squares(residual, [math.log(INVERSE_SPEC['initial_h'])],
                        bounds=np.log(INVERSE_SPEC['bounds']), max_nfev=40,
                        xtol=1e-8, ftol=1e-8, gtol=1e-8)
    return {'h': math.exp(float(fit.x[0])), 'success': bool(fit.success), 'nfev': int(fit.nfev),
            'message': str(fit.message), 'observation_mse': float(np.mean(fit.fun**2)),
            'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu}


def baseline_score(h, observations, coordinates, truth, true_h):
    def solve(holdout):
        ramp, drop, flux = (15., 7., 750.) if holdout else (10., 5., 500.)
        return PlanarContactModel(nx=120, nz=30, contact_conductance=float(h)).solve(
            INVERSE_SPEC['field_times_s'], initial_temperature=300., max_step=.0625,
            te_temperature=lambda t: 300.-drop*(-np.expm1(-t/ramp)),
            top_heat_flux=lambda t: flux*(-np.expm1(-t/ramp)))
    trajectory = solve(False)
    field = field_metrics(trajectory.sample(*coordinates.T), truth)
    observation = float(np.mean(((trajectory.sample(*observations.coordinates.T)-observations.temperature)/.02)**2))
    h_error = abs(h/true_h-1.)
    reduction = 1.-abs(h-true_h)/abs(INVERSE_SPEC['initial_h']-true_h)
    holdout_coords, holdout_truth, _ = coordinates_and_reference(true_h, holdout=True)
    holdout = field_metrics(solve(True).sample(*holdout_coords.T), holdout_truth)
    gates = {'observations': observation <= 2., 'field': field['rmse_K'] <= .05 and field['maximum_error_K'] <= .15,
             'parameter': h_error <= .1 and reduction >= .5,
             'holdout': holdout['rmse_K'] <= .05 and holdout['maximum_error_K'] <= .15}
    return {'contact_estimate': h, 'relative_h_error': h_error, 'initial_error_reduction': reduction,
            'observation_mse': observation, 'field': field, 'holdout_parameter_transfer': holdout,
            'gates': gates, 'passed': all(gates.values())}


def pending_two_d_rows(reason):
    return [{'index': index, 'true_h': h, 'methods': {
        method: {'status': 'not_run', 'reason': reason} for method in ('revised', 'conventional', 'nominal')}}
        for index, h in enumerate(h for h in INVERSE_SPEC['evaluation_h'] for _ in range(2))]


def restore_partial_one_d(out, result):
    """Restore completed saved trials if interruption precedes the 1D summary."""
    one_d = result['one_d_result']
    base = Path(out)/'one-d'
    errors = []
    def load(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as error:
            errors.append({'path': str(path.relative_to(out)), 'error': type(error).__name__+': '+str(error)})
            return None

    summary = base/'result.json'
    if summary.is_file():
        saved = load(summary)
        if isinstance(saved, dict) and isinstance(saved.get('trials'), list):
            result['one_d_result'] = saved
            return
    attempted = False
    for index, row in enumerate(one_d['trials']):
        directory = base/f"{row['family']}-trial-{row['index']:02d}"
        if not directory.is_dir():
            continue
        attempted = True
        path = directory/'result.json'
        saved = load(path) if path.is_file() else None
        if (isinstance(saved, dict) and saved.get('family') == row['family']
            and saved.get('index') == row['index']):
            one_d['trials'][index] = saved
            continue
        row.update(status='interrupted', completed=False, methods={})
        for method in ('scalar', 'three_knot'):
            raw_path = directory/f'{method}-raw-fit.json'
            raw = load(raw_path) if raw_path.is_file() else None
            row['methods'][method] = dict(raw or {}, completed=False,
                status='interrupted' if raw_path.is_file() else 'not_run',
                saved_raw_fit=str(raw_path.relative_to(out)) if raw_path.is_file() else None)
    if attempted:
        one_d['completed_trials'] = sum(bool(row.get('completed')) for row in one_d['trials'])
        one_d['status'] = 'interrupted'
    if errors:
        one_d['recovery_errors'] = errors


def artifact_inventory(out, *, incomplete=False):
    excluded = {'state.json', 'evidence-index.json', 'incomplete-evidence-index.json'}
    files = [{'path': str(path.relative_to(out)), 'bytes': path.stat().st_size, 'sha256': digest_file(path)}
             for path in sorted(out.rglob('*')) if path.is_file() and path.name not in excluded]
    value = {'files': files, 'digest': canonical_digest(files)}
    if incomplete:
        value['status'] = 'incomplete; inventory only, not accepted evaluation evidence'
    write_json(out/('incomplete-evidence-index.json' if incomplete else 'evidence-index.json'), value)
    return value


def run_fresh_revision(output_directory, *, source_commit, ci_evidence, preserved_campaign):
    protocol = validated_revision_protocol()
    runtime = runtime_context(protocol)
    context = source_context(source_commit)
    preserved_context(preserved_campaign, expected=protocol['preservation'])
    ci_path = Path(ci_evidence)
    ci = verify_ci(json.loads(ci_path.read_text()), source_commit)
    out = Path(output_directory).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(context, protocol=protocol, protocol_digest=canonical_digest(protocol), runtime=runtime,
                    ci=ci, started_utc=now(), preserved_campaign=str(Path(preserved_campaign).resolve()))
    write_json(out/'manifest.json', manifest)
    (out/'exact-source-ci.json').write_bytes(ci_path.read_bytes())
    state = {'status': 'numerical_preflight', 'source_commit': source_commit}
    result = {'status': 'running', 'source_commit': source_commit, 'protocol_digest': canonical_digest(protocol),
              'selection_summary': {'two_d_architecture': protocol['selected_inverse']['config']['architecture'],
                  'inverse_strategy': protocol['selected_inverse']['strategy'],
                  'one_d_design': protocol['one_d_selection']['candidate']},
              'planned_denominators': protocol['planned_denominators'],
              'resources': protocol.get('execution', {}),
              'two_d_trials': pending_two_d_rows('not_started'),
              'one_d_result': {'status': 'not_run', 'planned_trials': 6,
                  'independent_paired_noise_blocks': 3,
                  'trials': [{'family': family, 'index': index, 'completed': False, 'status': 'not_run'}
                             for family in ('matched_pwl', 'cubic_mismatch') for index in range(3)]}}
    def progress(event):
        state.update(updated_utc=now(), progress=event)
        write_json(out/'state.json', state)
        print(json.dumps(dict(utc=now(), **event), allow_nan=False), flush=True)
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    old_signal = signal.signal(signal.SIGTERM, interrupted)
    try:
        progress({'stage': 'numerical_preflight'})
        numerical = numerical_preflight(out/'two-d-reference')
        result['numerical_reference'] = numerical
        if not numerical['passed']:
            result['two_d_trials'] = pending_two_d_rows('numerical_reference_failed')
            result['status'] = 'blocked_reference'
        else:
            config = PlanarRevisionConfig(**protocol['selected_inverse']['config'])
            state['status'] = 'two_d_evaluation_running'
            for row in result['two_d_trials']:
                index, true_h = row['index'], row['true_h']
                directory = out/f"two-d-trial-{index:02d}"
                directory.mkdir()
                coords, truth, trajectory = coordinates_and_reference(true_h)
                observed = observations_from_reference(trajectory, semantic_seed(CAMPAIGN, index, 'observation_noise'))
                write_json(directory/'observations.json', observed.payload())
                np.savez_compressed(directory/'reference.npz', coordinates=coords, temperature=truth)
                try:
                    row['methods']['revised'] = {'status': 'running'}
                    write_json(out/'partial-result.json', result)
                    progress({'stage': 'inverse', 'trial': index})
                    model, fit = fit_delayed_inverse(observed, config=config,
                        strategy=protocol['selected_inverse']['strategy'], seed_namespace=f'{CAMPAIGN}/{index}',
                        progress=lambda event: progress(dict(stage='optimizer', trial=index, optimizer=event)),
                        completed_stage=lambda model, report, stage: retain_stage_evidence(directory, model, report, stage))
                    write_json(directory/'revised-fit.json', fit)
                    retain_checkpoint(directory/'revised-final.pt', model)
                    row['methods']['revised'] = {'status': 'fit_completed_scoring_pending',
                        'report': dict({key: fit[key] for key in ('wall_seconds', 'cpu_seconds') if key in fit},
                                       timing_scope='optimizer stages and independent physics checks')}
                    write_json(directory/'result.json', row)
                    score = score_inverse(model, fit, observed, coords, truth, true_h)
                    from ..pinn.planar_contact_revision import predict_planar_revision
                    np.savez_compressed(directory/'revised-predictions.npz', coordinates=coords,
                        temperature=predict_planar_revision(model, coords),
                        observation_coordinates=observed.coordinates,
                        observation_predictions=predict_planar_revision(model, observed.coordinates))
                    row['methods']['revised'].update(status=fit['status'], score=score)
                except Exception as error:
                    row['methods']['revised'].update(status='failed', error=type(error).__name__+': '+str(error),
                                                      traceback=traceback.format_exc())
                for method in ('conventional', 'nominal'):
                    try:
                        row['methods'][method] = {'status': 'running'}
                        fit = conventional_fit(observed) if method == 'conventional' else {
                            'h': INVERSE_SPEC['initial_h'], 'success': True, 'fit_performed': False}
                        write_json(directory/f'{method}-fit.json', fit)
                        row['methods'][method].update(status='fit_completed_scoring_pending', fit=fit)
                        score = baseline_score(fit['h'], observed, coords, truth, true_h)
                        score.setdefault('gates', {})['execution'] = bool(fit['success'])
                        score['passed'] = bool(score['passed'] and fit['success'])
                        row['methods'][method] = {'status': 'completed' if fit['success'] else 'optimizer_failed',
                                                  'fit': fit, 'score': score}
                    except Exception as error:
                        row['methods'][method].update(status='failed', error=type(error).__name__+': '+str(error))
                write_json(directory/'result.json', row)
                write_json(out/'partial-result.json', result)
                if source_context(source_commit)['source_digest'] != context['source_digest']:
                    raise RuntimeError('frozen source changed during evaluation')
                preserved_context(preserved_campaign, expected=protocol['preservation'])
            state['status'] = 'one_d_evaluation_running'
            progress({'stage': 'one_d'})
            run_one_d_evaluation(out/'one-d', protocol['one_d_selection'], CAMPAIGN+'/one-d')
            result['one_d_result'] = json.loads((out/'one-d/result.json').read_text())
            result['status'] = 'completed' if result['one_d_result']['status'] == 'completed' else 'stopped_one_d'
        if source_context(source_commit)['source_digest'] != context['source_digest']:
            raise RuntimeError('frozen source changed during evaluation')
        preserved_context(preserved_campaign, expected=protocol['preservation'])
        result['completed_utc'] = now()
        write_json(out/'result.json', result)
        from ..reports.pinn_spatial_revision import render_revision_report
        render_revision_report(out, result)
        artifact_inventory(out)
        state['status'] = result['status']
        progress({'stage': 'campaign', 'event': result['status']})
        return result
    except BaseException as error:
        state.update(status='failed_preserve_evidence', error=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
        write_json(out/'state.json', state)
        result['status'] = state['status']
        for row in result['two_d_trials']:
            for method in row['methods'].values():
                if method['status'] in ('running', 'fit_completed_scoring_pending'):
                    method.update(status='interrupted', reason=state['error'])
        try:
            restore_partial_one_d(out, result)
        except Exception as recovery_error:
            result['recovery_error'] = type(recovery_error).__name__+': '+str(recovery_error)
        try:
            write_json(out/'partial-result.json', result)
        finally:
            artifact_inventory(out, incomplete=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_signal)


def validate_revision_evidence(output_directory):
    # The shared validator requires the same portable manifest/CI/report files,
    # rejects extra or escaped paths, and only checks artifact integrity.
    return validate_original_evidence(output_directory)
