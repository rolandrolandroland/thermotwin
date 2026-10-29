"""Probe-selected optimizer revision on the unchanged log-off graded problem.

The learning rate, optimizer-history policy, and fixed budget must come from an
explicit training-only decision artifact. This module provides no selected
defaults, continuation from probe checkpoints, or acceptance-based tuning.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from . import pinn_spatial_continuity as common
from . import pinn_spatial_graded as graded

ROOT = common.ROOT
WALL_SECONDS = 4*60*60.
RSS_BYTES = 2*1024**3


def _read(path):
    return json.loads(Path(path).read_text())


def _api():
    from ..pinn import planar_contact_optimized
    return planar_contact_optimized


def _bound_path(path):
    path = Path(path)
    return (ROOT/path).resolve() if not path.is_absolute() else path.resolve()


def _decision(path, config):
    path = _bound_path(path)
    relative = str(path.relative_to(ROOT))
    decision = _read(path)
    if (decision.get('status') != 'approved_for_forward_development'
        or decision.get('training_only') is not True
        or decision.get('reference_fields_used') is not False
        or decision.get('independent_diagnostics_used_for_selection') is not False):
        raise ValueError('an explicit approved training-only optimizer decision is required')
    if decision.get('selected_config') != config:
        raise ValueError('optimizer configuration differs from the probe decision')
    evidence = decision.get('probe_evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('optimizer decision lacks bound probe evidence')
    paths = set()
    for record in evidence:
        artifact = _bound_path(record['path'])
        name = str(artifact.relative_to(ROOT))
        if name in paths or common.digest_file(artifact) != record['sha256']:
            raise ValueError('probe evidence identity differs or is duplicated')
        paths.add(name)
    return {'path': relative, 'sha256': common.digest_file(path), 'decision': decision}


def _config(model_config, old_config):
    if not isinstance(model_config, dict):
        raise ValueError('the complete optimizer configuration must be explicitly supplied')
    config = asdict(_api().OptimizedConfig(**model_config))
    if set(config) != set(model_config):
        raise ValueError('optimizer configuration must not rely on undeclared defaults')
    if config.get('bottom_log_feature') is not False or config.get('optimizer_history') not in ('persistent', 'reset'):
        raise ValueError('the selected rule requires log-off and an explicit supported optimizer-history policy')
    if config.get('adam_epochs', 0) < 1:
        raise ValueError('the optimizer revision requires a positive explicit fixed update budget')
    mutable = {'adam_epochs', 'learning_rate', 'max_backtracks'}
    for key, value in old_config.items():
        if key not in mutable and config.get(key) != value:
            raise ValueError('optimizer revision changed frozen model, objective, grid, or rank cutoff: '+key)
    return config


def _preflight(config, old_plan):
    import torch
    torch.set_num_threads(1)
    api = _api()
    graded_api = graded._model_api()
    instance = api.OptimizedConfig(**config)
    records = {}
    for case in common.CASES:
        physics = common.physics_for(case)
        _, initialization = api.matched_optimized_model(physics, instance, case['known_h'], common.SEEDS['neural'])
        grid = graded_api.graded_point_metadata(physics, instance, common.SEEDS['sampling'])
        prior = old_plan['preflight_by_case'][case['id']]
        if initialization['common_weights_sha256'] != prior['initialization']['off']['common_weights_sha256']:
            raise ValueError('fresh optimizer initialization differs from original log-off common weights')
        if initialization.get('log_column_present') is not False or initialization['probe_agreement']['passed'] is not True:
            raise ValueError('optimizer initialization is not the verified log-off field')
        if grid != prior['training_grid']['log_off']:
            raise ValueError('optimizer revision changed the graded points or their weights')
        records[case['id']] = {'initialization': initialization, 'training_grid': grid}
    return records


def prepare_optimized(output_directory, *, model_config, probe_decision_path,
        source_paths, graded_comparison, paired_v2_directory, v2_development_record,
        preserved_campaign, protected_campaigns, protected_files):
    """Prepare one four-case rule only after the training-only decision exists."""
    graded_comparison = Path(graded_comparison).resolve()
    old_context = graded.validate_comparison(graded_comparison)
    old_plan = _read(graded_comparison/'plan.json')
    old_config = old_plan['model_configs']['log_off']
    config = _config(model_config, old_config)
    decision = _decision(probe_decision_path, config)
    records = _preflight(config, old_plan)  # constructed initialization/grid probes; no optimization
    protected_campaigns = tuple(Path(path).resolve() for path in protected_campaigns)
    if len(protected_campaigns) != 2 or len(set(protected_campaigns)) != 2:
        raise ValueError('both complete V3 and V4 campaigns must remain protected')
    old_campaigns = {}
    sources = set(source_paths) | set(old_plan['source_sha256']) | {
        'thermotwin/studies/pinn_spatial_optimized.py', 'thermotwin/pinn/planar_contact_optimized.py',
        'tests/test_pinn_spatial_optimized.py'}
    for directory in protected_campaigns:
        prior_plan, context = graded._validate_old_campaign(directory)
        sources.update(prior_plan['source_sha256'])
        old_campaigns[str(directory)] = context
    file_records = common._source_records(protected_files)
    directory = Path(output_directory).resolve()
    plan = common.prepare_development(directory, model_config=config, source_paths=sorted(sources),
        paired_v2_directory=paired_v2_directory,
        protected_directories=(*protected_campaigns, graded_comparison),
        maximum_wall_seconds=WALL_SECONDS, maximum_rss_bytes=RSS_BYTES,
        candidate_id='training-probe-selected-optimized-v6-log-off',
        v2_development_record=v2_development_record, preserved_campaign=preserved_campaign)
    plan.update(schema='probe_selected_optimized_log_off_v6',
        optimization_provenance={'decision': decision,
            'graded_comparison': {'directory': str(graded_comparison), 'context': old_context,
                'plan_sha256': common.digest_file(graded_comparison/'plan.json')},
            'old_campaigns': old_campaigns, 'protected_files': file_records,
            'preflight_by_case': records, 'original_log_off_config': old_config,
            'initialization_policy': 'fresh original seed and original common weights; no probe or campaign checkpoint continuation',
            'selection_boundary': 'training-only probe objectives; no independent acceptance diagnostics or hidden reference fields'},
        fixed_optimizer_updates=config['adam_epochs'], optimizer_history=config['optimizer_history'],
        reused_paired_datasets=4, independent_noise_replications_claimed=False,
        topology={'execution': 'single isolated main process; four cases in declared order',
            'threads': 1, 'rss_limit_bytes': RSS_BYTES, 'campaign_wall_seconds': WALL_SECONDS},
        near_bottom_audit={'point_digest': common.point_digest(graded.near_bottom_points()),
            'role': 'inherited report-only whole-plate audit after checkpoint; unchanged six height strata'},
        no_inverse_or_reserved_or_three_d=True)
    common._write(directory/'plan.json', plan)
    for name in plan['source_sha256']:
        target = directory/'source-snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    common._write(directory/'state.json', {'status': 'prepared', 'plan_sha256': common.digest_file(directory/'plan.json')})
    verify_optimized_plan(plan, directory)
    return plan


def verify_optimized_plan(plan, output_directory=None):
    common.verify_plan(plan)
    if (plan.get('schema') != 'probe_selected_optimized_log_off_v6'
        or plan.get('cases') != list(common.CASES) or plan.get('maximum_fits') != 4
        or plan.get('maximum_wall_seconds') != WALL_SECONDS or plan.get('maximum_rss_bytes') != RSS_BYTES
        or plan.get('reused_paired_datasets') != 4 or plan.get('retries') != 0):
        raise ValueError('optimized campaign scope, denominator, or resource caps changed')
    provenance = plan['optimization_provenance']
    if _decision(provenance['decision']['path'], plan['model_config']) != provenance['decision']:
        raise RuntimeError('optimizer decision or its evidence changed')
    old = provenance['graded_comparison']
    if graded.validate_comparison(old['directory']) != old['context']:
        raise RuntimeError('the preserved graded comparison changed')
    if common.digest_file(Path(old['directory'])/'plan.json') != old['plan_sha256']:
        raise RuntimeError('the preserved graded comparison plan changed')
    if common._source_records(provenance['protected_files']) != provenance['protected_files']:
        raise RuntimeError('a protected tracked experiment record changed')
    for path, expected in provenance['old_campaigns'].items():
        _, context = graded._validate_old_campaign(path)
        if context != expected:
            raise RuntimeError('a prior forward campaign changed')
    config = _config(plan['model_config'], provenance['original_log_off_config'])
    if plan.get('fixed_optimizer_updates') != config['adam_epochs'] or plan.get('optimizer_history') != config['optimizer_history']:
        raise ValueError('optimizer rule differs from the fixed probe-selected configuration')
    if common.point_digest(graded.near_bottom_points()) != plan['near_bottom_audit']['point_digest']:
        raise RuntimeError('report-only near-bottom audit changed')
    if output_directory is not None and _read(Path(output_directory)/'plan.json') != plan:
        raise RuntimeError('executed optimizer plan differs from its supplied proof')
    return {'source_sha256': plan['source_sha256'], 'case_count': 4, 'reused_dataset_count': 4}


def _training_proof(raw, plan, case_id):
    expected = plan['optimization_provenance']['preflight_by_case'][case_id]
    if raw.get('config') != plan['model_config']:
        raise ValueError('executed optimizer configuration differs from the frozen rule')
    if len(raw.get('hidden_update_history', [])) != plan['fixed_optimizer_updates']:
        raise ValueError('executed optimizer update count differs from its fixed final budget')
    expected_instances = 1 if plan['optimizer_history'] == 'persistent' else plan['fixed_optimizer_updates']
    if raw.get('optimizer_history') != plan['optimizer_history'] or raw.get('optimizer_instances') != expected_instances:
        raise ValueError('executed optimizer-history mechanics differ from the probe-selected policy')
    actual = raw.get('initialization') or {}
    declared = expected['initialization']
    if ({key: value for key, value in actual.items() if key != 'probe_agreement'}
        != {key: value for key, value in declared.items() if key != 'probe_agreement'}
        or (actual.get('probe_agreement') or {}).get('passed') is not True):
        raise ValueError('executed fresh initialization differs from the declared original log-off state')
    grid = expected['training_grid']
    if raw.get('graded_point_metadata') != grid:
        raise ValueError('executed graded grid metadata differs')
    identities, counts = {}, {}
    for name, values in (raw.get('training_points') or {}).items():
        values = np.asarray(values, dtype=np.float64)
        identities[name] = {'shape': list(values.shape), 'sha256': graded._model_api()._array_digest(values)}
        counts[name] = len(values)
        weights = np.asarray((raw.get('training_point_weights') or {}).get(name), dtype=np.float64)
        if not np.array_equal(weights, np.full(len(values), 1/len(values))):
            raise ValueError('executed training row weights differ from the frozen uniform means')
    if identities != grid['point_identities'] or counts != grid['point_counts']:
        raise ValueError('executed training coordinates differ from the frozen graded points')
    return {'verified': True, 'fresh_original_common_weights_sha256': declared['common_weights_sha256'],
        'point_digest': grid['point_digest'], 'optimizer_updates': plan['fixed_optimizer_updates'],
        'optimizer_history': plan['optimizer_history']}


def optimized_callbacks(output_directory, plan):
    api = _api()
    inherited = graded.graded_callbacks(output_directory)
    directory = Path(output_directory)
    def fit(observations, **kwargs):
        kwargs['config'] = api.OptimizedConfig(**kwargs['config'])
        return api.fit_optimized(observations, **kwargs)
    def diagnose(model, points):
        if 'top' in points:
            case_id = graded._case_for_model(model)['id']
            # The generic runner has retained both files before this callback.
            if not (directory/case_id/'final_checkpoint.pt').is_file():
                raise RuntimeError('optimizer diagnostics require the retained final checkpoint')
            _training_proof(_read(directory/case_id/'training-report.json'), plan, case_id)
            return inherited.diagnose(model, points)
        return api.diagnose_optimized(model, points)
    return common.ForwardCallbacks(fit=fit, predict=api.predict_optimized,
        diagnose=diagnose, checkpoint=api.checkpoint_bytes)


def run_optimized(output_directory, callbacks=None):
    directory = Path(output_directory).resolve()
    plan = _read(directory/'plan.json')
    state = _read(directory/'state.json')
    if state.get('status') != 'prepared' or state.get('plan_sha256') != common.digest_file(directory/'plan.json'):
        raise ValueError('optimized campaign is changed or already consumed')
    verify_optimized_plan(plan, directory)
    result = common.run_development(directory, callbacks or optimized_callbacks(directory, plan))
    verify_optimized_plan(plan, directory)
    return result


def validate_optimized(output_directory):
    from ..reports.pinn_spatial_continuity import validate_campaign
    directory = Path(output_directory).resolve()
    evidence = validate_campaign(directory)
    plan = _read(directory/'plan.json')
    verify_optimized_plan(plan, directory)
    result = _read(directory/'final-forward-selection.json')
    if len(result.get('cases', [])) != 4 or {row['id'] for row in result['cases']} != {case['id'] for case in common.CASES}:
        raise ValueError('optimized campaign omitted or duplicated a declared case')
    for row in result['cases']:
        if row.get('status') == 'completed':
            _training_proof(_read(directory/row['id']/'training-report.json'), plan, row['id'])
    return dict(evidence, optimizer_history=plan['optimizer_history'],
        fixed_optimizer_updates=plan['fixed_optimizer_updates'], reused_dataset_count=4)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    for flag in ('output-directory', 'config', 'probe-decision', 'graded-comparison', 'paired-v2-directory',
                 'v2-development-record', 'preserved-campaign'):
        prepare.add_argument('--'+flag, required=True)
    for flag in ('source-path', 'protected-campaign', 'protected-file'):
        prepare.add_argument('--'+flag, action='append', required=True)
    for command in ('run', 'validate'):
        commands.add_parser(command).add_argument('--output-directory', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare_optimized(args.output_directory, model_config=_read(args.config),
            probe_decision_path=args.probe_decision, source_paths=args.source_path,
            graded_comparison=args.graded_comparison, paired_v2_directory=args.paired_v2_directory,
            v2_development_record=args.v2_development_record, preserved_campaign=args.preserved_campaign,
            protected_campaigns=args.protected_campaign, protected_files=args.protected_file)
    elif args.command == 'run':
        result = run_optimized(args.output_directory)
    else:
        result = validate_optimized(args.output_directory)
    print(json.dumps({'command': args.command, 'status': result.get('status', 'prepared' if args.command == 'prepare' else 'validated')}, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
