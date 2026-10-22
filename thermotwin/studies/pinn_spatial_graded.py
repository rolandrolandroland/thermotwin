"""Paired log-input ablation on one frozen bottom-graded training grid.

Preparation freezes both four-case plans before either arm may fit. Separate
main processes execute log-on and log-off, so the inherited resource guards
remain effective. Finalization retains all eight method fits and selects no
winner, checkpoint, inverse rule, or reserved evaluation cohort.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from . import pinn_spatial_continuity as common

ROOT = common.ROOT
ARMS = ('log_on', 'log_off')
ARM_FLAGS = {'log_on': True, 'log_off': False}
UPDATES = 80
WALL_SECONDS = 5400.
RSS_BYTES = 2*1024**3


def _read(path):
    return json.loads(Path(path).read_text())


def _model_api():
    from ..pinn import planar_contact_graded
    return planar_contact_graded


def near_bottom_points():
    """A new report-only whole-plate audit, including full-contact cases."""
    xx, zz, tt = np.meshgrid((np.arange(32)+.37)/32,
        np.array([.001, .002, .005, .01, .02, .05]),
        np.array([.4, 1.1, 4.5, 14., 45., 105.])/120., indexing='ij')
    return {'interior': np.column_stack((xx.ravel(), zz.ravel(), tt.ravel()))}


def _configs(base_config):
    api = _model_api()
    base = asdict(api.GradedConfig(**(base_config or {})))
    if (base['adam_epochs'] != UPDATES or base['projection_every'] != 1
        or base['interior_points'] != 8192 or base['bottom_points'] != 512):
        raise ValueError('both graded arms require 80 updates, 8192 interior points, and 512 points per bottom family')
    defaults = asdict(api.GradedConfig())
    if any(base[key] != value for key, value in defaults.items() if key != 'bottom_log_feature'):
        raise ValueError('the complete graded configuration is fixed apart from the paired log switch')
    return {arm: dict(base, bottom_log_feature=flag) for arm, flag in ARM_FLAGS.items()}


def _preflight(configs):
    import torch
    torch.set_num_threads(1)
    api = _model_api()
    prepared = {arm: api.GradedConfig(**config) for arm, config in configs.items()}
    records = {}
    for case in common.CASES:
        physics = common.physics_for(case)
        initialization = api.verify_matched_initialization(physics, prepared['log_on'],
            prepared['log_off'], case['known_h'], common.SEEDS['neural'])
        grids = {arm: api.graded_point_metadata(physics, config, common.SEEDS['sampling'])
                 for arm, config in prepared.items()}
        if initialization['probe_agreement']['passed'] is not True:
            raise ValueError('constructed initial function probes disagree')
        if initialization['on']['common_weights_sha256'] != initialization['off']['common_weights_sha256']:
            raise ValueError('common initial parameter identities differ')
        if grids['log_on'] != grids['log_off']:
            raise ValueError('the two log arms do not use identical graded points and weights')
        records[case['id']] = {'initialization': initialization, 'training_grid': grids}
    return records


def _validate_old_campaign(directory):
    from ..reports.pinn_spatial_continuity import validate_campaign
    context = validate_campaign(directory)
    old_plan = _read(Path(directory)/'plan.json')
    common.verify_plan(old_plan)
    return old_plan, context


def prepare_comparison(output_directory, *, source_paths, paired_v2_directory,
        v2_development_record, preserved_campaign, protected_campaigns,
        protected_files, base_config=None):
    """Write both arm plans, initialization/grid proofs, then the top plan."""
    configs = _configs(base_config)
    protected_campaigns = tuple(Path(path).resolve() for path in protected_campaigns)
    if len(protected_campaigns) != 2 or len(set(protected_campaigns)) != 2:
        raise ValueError('the complete distinct V3 and V4 campaigns must both be protected')
    protected = {}
    paths = set(source_paths) | {'thermotwin/studies/pinn_spatial_graded.py',
        'thermotwin/pinn/planar_contact_graded.py', 'tests/test_pinn_spatial_graded.py',
        'thermotwin/studies/pinn_spatial_continuity.py', 'thermotwin/physics/planar_contact.py',
        'thermotwin/pinn/planar_contact_inverse.py'}
    for directory in protected_campaigns:
        old_plan, context = _validate_old_campaign(directory)
        protected[str(directory)] = context
        paths.update(old_plan['source_sha256'])
    protected_file_records = common._source_records(protected_files)
    records = _preflight(configs)  # no optimization, observations, or hidden fields
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    subplans = {}
    for arm in ARMS:
        plan = common.prepare_development(directory/arm, model_config=configs[arm],
            source_paths=sorted(paths), paired_v2_directory=paired_v2_directory,
            protected_directories=protected_campaigns, maximum_wall_seconds=WALL_SECONDS,
            maximum_rss_bytes=RSS_BYTES, candidate_id='paired-bottom-graded-v5-'+arm.replace('_', '-'),
            v2_development_record=v2_development_record, preserved_campaign=preserved_campaign)
        subplans[arm] = {'relative_directory': arm,
            'plan_sha256': common.digest_file(directory/arm/'plan.json'),
            'paired_evidence_digest': common.canonical_digest(plan['paired_evidence'])}
    if len({record['paired_evidence_digest'] for record in subplans.values()}) != 1:
        raise ValueError('paired observation/reference bindings differ across arms')
    sources = common._source_records(sorted(paths))
    plan = {'schema': 'paired_graded_log_ablation_v5', 'candidate_id': 'paired-bottom-graded-v5',
        'created_utc': common.datetime.now(common.timezone.utc).isoformat(),
        'scope': 'development only; log-input comparison on reused observations; no inverse, reserved cohort, or 3D',
        'cases': list(common.CASES), 'arms': list(ARMS), 'model_configs': configs,
        'fixed_variable_projection_updates': UPDATES, 'maximum_method_fits': 8,
        'reused_paired_datasets': 4, 'independent_noise_replications_claimed': False,
        'topology': {'execution': 'two concurrent isolated main processes',
            'per_process_threads': 1, 'aggregate_child_threads': 2,
            'per_process_rss_limit_bytes': RSS_BYTES, 'aggregate_child_rss_limit_bytes': 2*RSS_BYTES,
            'per_arm_campaign_wall_seconds': WALL_SECONDS,
            'sum_of_arm_wall_caps_seconds': 2*WALL_SECONDS,
            'case_order_within_each_arm': [case['id'] for case in common.CASES]},
        'automatic_retries': 0, 'winner_selection': 'none; retain both complete trajectories and every outcome',
        'subplans': subplans, 'source_sha256': sources, 'runtime': common.runtime_context(),
        'protected_campaigns': protected, 'protected_files': protected_file_records,
        'preflight_by_case': records,
        'primary_gates': common.GATES, 'seeds': common.SEEDS,
        'near_bottom_audit': {'role': 'report only after final checkpoint; no acceptance or selection use',
            'normalized_x': '(arange(32)+.37)/32', 'normalized_heights': [.001, .002, .005, .01, .02, .05],
            'times_s': [.4, 1.1, 4.5, 14., 45., 105.],
            'point_digest': common.point_digest(near_bottom_points()), 'point_count': 1152},
        'checkpoint_rule': 'inherit optimizer-returned final state and raw-before-independent-score retention',
        'failure_rule': 'retain failed/nonfinite and unrun rows; no retry, imputation to success, or winner selection'}
    common._write(directory/'plan.json', plan)
    for path in sources:
        target = directory/'source-snapshot'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/path).read_bytes())
    common._write(directory/'state.json', {'status': 'prepared',
        'plan_sha256': common.digest_file(directory/'plan.json')})
    verify_comparison_plan(plan, directory)
    return plan


def verify_comparison_plan(plan, output_directory=None):
    """Check source, old seals, both subplans, fixed budgets, and pairing proof."""
    if plan.get('schema') != 'paired_graded_log_ablation_v5' or plan.get('arms') != list(ARMS):
        raise ValueError('unrecognized paired graded comparison plan')
    if (plan.get('cases') != list(common.CASES) or plan.get('maximum_method_fits') != 8
        or plan.get('fixed_variable_projection_updates') != UPDATES or plan.get('automatic_retries') != 0
        or plan.get('reused_paired_datasets') != 4 or plan.get('primary_gates') != common.GATES
        or plan.get('seeds') != common.SEEDS or set(plan.get('subplans', {})) != set(ARMS)):
        raise ValueError('paired comparison cases, budgets, streams, or gates changed')
    topology = plan['topology']
    if (topology.get('execution') != 'two concurrent isolated main processes'
        or topology.get('per_process_threads') != 1 or topology.get('aggregate_child_threads') != 2
        or topology.get('per_process_rss_limit_bytes') != RSS_BYTES
        or topology.get('aggregate_child_rss_limit_bytes') != 2*RSS_BYTES
        or topology.get('per_arm_campaign_wall_seconds') != WALL_SECONDS):
        raise ValueError('paired concurrent execution topology or resource limits changed')
    if common._source_records(plan['source_sha256']) != plan['source_sha256']:
        raise RuntimeError('graded comparison source changed')
    if common._source_records(plan['protected_files']) != plan['protected_files']:
        raise RuntimeError('protected tracked experiment record changed')
    if common.runtime_context() != plan['runtime']:
        raise RuntimeError('graded comparison runtime changed')
    for old_directory, expected in plan['protected_campaigns'].items():
        _, actual = _validate_old_campaign(old_directory)
        if actual != expected:
            raise RuntimeError('protected earlier campaign changed')
    if common.point_digest(near_bottom_points()) != plan['near_bottom_audit']['point_digest']:
        raise RuntimeError('whole-plate near-bottom audit points changed')
    configs = plan['model_configs']
    on, off = dict(configs['log_on']), dict(configs['log_off'])
    if on.pop('bottom_log_feature') is not True or off.pop('bottom_log_feature') is not False or on != off:
        raise ValueError('log arms differ beyond their declared input switch')
    if on.get('adam_epochs') != UPDATES or on.get('projection_every') != 1:
        raise ValueError('paired optimizer update budget changed')
    for case in common.CASES:
        record = plan['preflight_by_case'][case['id']]
        proof = record['initialization']
        if proof['probe_agreement']['passed'] is not True or proof['on']['common_weights_sha256'] != proof['off']['common_weights_sha256']:
            raise ValueError('initial function/parameter pairing proof is invalid')
        if record['training_grid']['log_on'] != record['training_grid']['log_off']:
            raise ValueError('graded grid pairing proof is invalid')
    if output_directory is not None:
        directory = Path(output_directory).resolve()
        if _read(directory/'plan.json') != plan:
            raise RuntimeError('top plan on disk differs')
        for arm, record in plan['subplans'].items():
            path = directory/record['relative_directory']/'plan.json'
            if common.digest_file(path) != record['plan_sha256']:
                raise RuntimeError('paired subplan changed')
            subplan = _read(path)
            common.verify_plan(subplan)
            if subplan['model_config'] != configs[arm] or subplan['source_sha256'] != plan['source_sha256']:
                raise RuntimeError('paired subplan field rule or source differs')
            if subplan['maximum_wall_seconds'] != WALL_SECONDS or subplan['maximum_rss_bytes'] != RSS_BYTES:
                raise RuntimeError('paired subplan resource caps differ')
            if common.canonical_digest(subplan['paired_evidence']) != record['paired_evidence_digest']:
                raise RuntimeError('paired observation/reference bindings changed')
    return {'source_sha256': plan['source_sha256'], 'case_count': 4,
            'method_fit_count': 8, 'reused_dataset_count': 4}


def _case_for_model(model):
    for case in common.CASES:
        if model.physics == common.physics_for(case) and float(model.known_h) == case['known_h']:
            return case
    raise ValueError('diagnostic model does not belong to a declared arm')


def graded_callbacks(output_directory):
    api = _model_api()
    directory = Path(output_directory)
    def fit(observations, **kwargs):
        kwargs['config'] = api.GradedConfig(**kwargs['config'])
        return api.fit_graded(observations, **kwargs)
    def diagnose(model, points):
        primary = api.diagnose_graded(model, points)
        if 'top' in points:
            case_directory = directory/_case_for_model(model)['id']
            if not (case_directory/'final_checkpoint.pt').is_file() or not (case_directory/'training-report.json').is_file():
                raise RuntimeError('whole-plate audit requires retained final checkpoint and raw report')
            bottom = near_bottom_points()
            common._npz(case_directory/'near-bottom-audit-points.npz', **bottom)
            audit = {'status': 'completed', 'role': 'report only',
                'point_digest': common.point_digest(bottom), 'point_count': len(bottom['interior'])}
            try:
                audit['diagnostics'] = api.diagnose_graded(model, bottom)
                audit['by_height'] = []
                values = bottom['interior']
                for height in np.unique(values[:, 1]):
                    group = values[values[:, 1] == height]
                    audit['by_height'].append({'z_over_H': float(height), 'point_count': len(group),
                        'diagnostics': api.diagnose_graded(model, {'interior': group})})
                common._json_bytes(audit)
            except common.ResourceLimit:
                raise
            except Exception as error:
                audit = {'status': 'failed', 'role': 'report only',
                    'point_digest': common.point_digest(bottom), 'error': repr(error)}
            common._write(case_directory/'near-bottom-audit.json', audit)
        return primary
    return common.ForwardCallbacks(fit=fit, predict=api.predict_graded,
        diagnose=diagnose, checkpoint=api.checkpoint_bytes)


def run_arm(output_directory, arm, callbacks=None):
    if arm not in ARMS:
        raise ValueError('unknown log arm')
    directory = Path(output_directory).resolve()
    plan = _read(directory/'plan.json')
    state = _read(directory/'state.json')
    if state.get('status') != 'prepared' or state.get('plan_sha256') != common.digest_file(directory/'plan.json'):
        raise ValueError('top comparison is changed or already finalized')
    if (directory/f'process-exit-{arm}.json').exists():
        raise ValueError('a process exit was already recorded; this arm cannot be retried')
    verify_comparison_plan(plan, directory)
    subdirectory = directory/plan['subplans'][arm]['relative_directory']
    # Each process mutates only its own subcampaign; no racy shared state writes.
    return common.run_development(subdirectory, callbacks or graded_callbacks(subdirectory))


def record_arm_exit(output_directory, arm, return_code):
    """Retain a parent-observed native process failure without retrying it."""
    if arm not in ARMS or type(return_code) is not int:
        raise ValueError('an explicit arm and integer process return code are required')
    directory = Path(output_directory).resolve()
    plan = _read(directory/'plan.json')
    verify_comparison_plan(plan, directory)
    path = directory/f'process-exit-{arm}.json'
    if path.exists() or _read(directory/'state.json').get('status') != 'prepared':
        raise ValueError('process exit was already recorded or comparison was finalized')
    subdirectory = directory/plan['subplans'][arm]['relative_directory']
    record = {'arm': arm, 'return_code': return_code,
        'subcampaign_status': _read(subdirectory/'state.json').get('status'),
        'no_retry': True, 'raw_artifacts_retained': True}
    common._write(path, record)
    return record


def _completed_arm(directory):
    from ..reports.pinn_spatial_continuity import validate_campaign
    context = validate_campaign(directory)
    return _read(directory/'final-forward-selection.json'), context


def _verify_training_pairing(directory, plan, arm, row):
    path = directory/arm/row['id']/'training-report.json'
    if not path.is_file():
        if row.get('status') == 'completed':
            raise ValueError('completed method lacks its raw training proof')
        return {'status': 'not_recorded_after_failure'}
    raw = _read(path)
    expected = plan['preflight_by_case'][row['id']]
    initialization = expected['initialization']['on' if arm == 'log_on' else 'off']
    actual_initialization = raw.get('initialization') or {}
    common_actual = {key: value for key, value in actual_initialization.items() if key != 'probe_agreement'}
    common_expected = {key: value for key, value in initialization.items() if key != 'probe_agreement'}
    actual_probe = actual_initialization.get('probe_agreement') or {}
    declared_probe = initialization.get('probe_agreement') or {}
    probe_matches = (actual_probe.get('passed') is True
        and actual_probe.get('relative_tolerance') == declared_probe.get('relative_tolerance')
        and actual_probe.get('absolute_tolerance') == declared_probe.get('absolute_tolerance'))
    # Common weights/state identities are exact; repeated floating probe residuals
    # are evidence of tolerance agreement, not a claim of bitwise function equality.
    if (common_actual != common_expected or not probe_matches
        or raw.get('graded_point_metadata') != expected['training_grid'][arm]):
        raise ValueError('executed initialization or graded points differ from the paired preflight')
    if raw.get('config') != plan['model_configs'][arm] or len(raw.get('hidden_update_history', [])) != UPDATES:
        raise ValueError('executed log switch or fixed optimizer budget differs')
    return {'status': 'verified', 'common_weights_sha256': initialization['common_weights_sha256'],
        'shared_function_sha256': initialization['shared_function_sha256'],
        'point_digest': expected['training_grid'][arm]['point_digest'], 'optimizer_updates': UPDATES}


def finalize_comparison(output_directory):
    """Seal both completed subcampaigns; no metric can choose a winner."""
    directory = Path(output_directory).resolve()
    plan = _read(directory/'plan.json')
    state = _read(directory/'state.json')
    if state.get('status') != 'prepared' or state.get('plan_sha256') != common.digest_file(directory/'plan.json'):
        raise ValueError('top comparison is changed or already finalized')
    verify_comparison_plan(plan, directory)
    arms, cases = {}, {}
    for arm in ARMS:
        subdirectory = directory/plan['subplans'][arm]['relative_directory']
        result, evidence = _completed_arm(subdirectory)  # refuses any still-running arm
        arms[arm] = {'directory': str(subdirectory), 'scientific_status': result['status'],
            'results': result, **evidence,
            'process_exit': _read(directory/f'process-exit-{arm}.json')
                if (directory/f'process-exit-{arm}.json').exists() else None}
        cases[arm] = {row['id']: row for row in result['cases']}
        if len(cases[arm]) != 4 or set(cases[arm]) != {case['id'] for case in common.CASES}:
            raise ValueError('an arm omitted or duplicated a declared case')
    paired = []
    for case in common.CASES:
        methods, proofs = {}, {}
        for arm in ARMS:
            row = cases[arm][case['id']]
            if row.get('geometry') != case['geometry'] or row.get('known_h') != case['known_h']:
                raise ValueError('paired case metadata differs')
            methods[arm] = row
            proofs[arm] = _verify_training_pairing(directory, plan, arm, row)
        paired.append(dict(case, methods=methods, pairing={'predeclared_matched_initialization': True,
            'predeclared_identical_graded_grid': True, 'executed_proofs': proofs}))
    verify_comparison_plan(plan, directory)
    result = {'schema': 'paired_graded_log_ablation_v5_result', 'status': 'completed',
        'plan_sha256': common.digest_file(directory/'plan.json'), 'arms': arms,
        'paired_case_rows': paired, 'method_fit_count': 8, 'reused_paired_datasets': 4,
        'independent_noise_replications_claimed': False, 'no_winner_selected': True,
        'no_retries': True, 'inverse_opened': False, 'fresh_evaluation_opened': False,
        'three_d_opened': False, 'source_unchanged': True,
        'source_sha256': plan['source_sha256'],
        'arm_wall_seconds': {arm: value['results'].get('wall_seconds') for arm, value in arms.items()},
        'arm_peak_rss_bytes': {arm: value['results'].get('peak_rss_bytes') for arm, value in arms.items()}}
    common._write(directory/'result.json', result)
    index = {str(path.relative_to(directory)): {'sha256': common.digest_file(path), 'bytes': path.stat().st_size}
        for path in sorted(directory.rglob('*')) if path.is_file() and path not in (directory/'state.json', directory/'artifact-index.json')}
    common._write(directory/'artifact-index.json', index)
    common._write(directory/'state.json', {'status': 'completed', 'plan_sha256': result['plan_sha256'],
        'artifact_index_sha256': common.digest_file(directory/'artifact-index.json')})
    return result


def validate_comparison(output_directory):
    directory = Path(output_directory).resolve()
    state = _read(directory/'state.json')
    if state.get('status') != 'completed':
        raise ValueError('paired comparison is not completed')
    index = _read(directory/'artifact-index.json')
    if state.get('artifact_index_sha256') != common.digest_file(directory/'artifact-index.json'):
        raise ValueError('paired comparison index differs')
    for name, record in index.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('invalid paired evidence path')
        if path.stat().st_size != record['bytes'] or common.digest_file(path) != record['sha256']:
            raise ValueError('paired evidence changed: '+name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file()
        and path not in (directory/'state.json', directory/'artifact-index.json')}
    if actual != set(index):
        raise ValueError('paired evidence index does not cover its file set')
    plan = _read(directory/'plan.json')
    verify_comparison_plan(plan, directory)
    result = _read(directory/'result.json')
    if state.get('plan_sha256') != common.digest_file(directory/'plan.json') or result.get('plan_sha256') != state['plan_sha256']:
        raise ValueError('paired comparison plan digest differs')
    for arm in ARMS:
        _, evidence = _completed_arm(directory/plan['subplans'][arm]['relative_directory'])
        if evidence['artifact_index_sha256'] != result['arms'][arm]['artifact_index_sha256']:
            raise ValueError('paired arm index differs from the final result')
    return {'artifact_index_sha256': common.digest_file(directory/'artifact-index.json'),
        'plan_sha256': state['plan_sha256'], 'files': len(index), 'method_fit_count': 8,
        'reused_dataset_count': 4}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--output-directory', required=True)
    prepare.add_argument('--paired-v2-directory', required=True)
    prepare.add_argument('--v2-development-record', required=True)
    prepare.add_argument('--preserved-campaign', required=True)
    prepare.add_argument('--protected-campaign', action='append', required=True)
    prepare.add_argument('--protected-file', action='append', required=True)
    prepare.add_argument('--source-path', action='append', required=True)
    prepare.add_argument('--base-config')
    run = commands.add_parser('run-arm')
    run.add_argument('--output-directory', required=True)
    run.add_argument('--arm', choices=ARMS, required=True)
    record_exit = commands.add_parser('record-exit')
    record_exit.add_argument('--output-directory', required=True)
    record_exit.add_argument('--arm', choices=ARMS, required=True)
    record_exit.add_argument('--return-code', type=int, required=True)
    finalize = commands.add_parser('finalize')
    finalize.add_argument('--output-directory', required=True)
    validate = commands.add_parser('validate')
    validate.add_argument('--output-directory', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare_comparison(args.output_directory, source_paths=args.source_path,
            paired_v2_directory=args.paired_v2_directory, v2_development_record=args.v2_development_record,
            preserved_campaign=args.preserved_campaign, protected_campaigns=args.protected_campaign,
            protected_files=args.protected_file, base_config=_read(args.base_config) if args.base_config else None)
    elif args.command == 'run-arm':
        result = run_arm(args.output_directory, args.arm)
    elif args.command == 'record-exit':
        result = record_arm_exit(args.output_directory, args.arm, args.return_code)
    elif args.command == 'finalize':
        result = finalize_comparison(args.output_directory)
    else:
        result = validate_comparison(args.output_directory)
    print(json.dumps({'command': args.command, 'status': result.get('status', 'prepared' if args.command == 'prepare' else 'validated')}, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
