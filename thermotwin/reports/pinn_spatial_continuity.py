"""Evidence-only comparison for the paired 2D continuity development repair.

The renderer validates sealed outputs and displays supplied gate decisions.
It cannot train a model, select a checkpoint, or change an acceptance rule.
All rendered files belong to a separate report directory.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _fmt(value):
    value = _number(value)
    return 'not recorded' if value is None else f'{value:.6g}'


def _pass(value):
    return 'pass' if value is True else 'fail' if value is False else 'not recorded'


def _json(path):
    return json.loads(Path(path).read_text())


def validate_campaign(directory):
    """Validate the complete sealed file set without scientific recomputation."""
    directory = Path(directory).resolve()
    state = _json(directory/'state.json')
    if state.get('status') != 'completed':
        raise ValueError('campaign has not completed')
    index_path = directory/'artifact-index.json'
    index = _json(index_path)
    if not isinstance(index, dict) or not {'plan.json', 'final-forward-selection.json'}.issubset(index):
        raise ValueError('campaign artifact index is incomplete')
    if state.get('artifact_index_sha256') != digest_file(index_path):
        raise ValueError('campaign index differs from terminal state')
    for name, record in index.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('invalid campaign evidence path: '+name)
        if path.stat().st_size != record['bytes'] or digest_file(path) != record['sha256']:
            raise ValueError('campaign evidence changed: '+name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file()
              and path.name not in ('state.json', 'artifact-index.json')}
    if actual != set(index):
        raise ValueError('campaign index does not cover its file set')
    result = _json(directory/'final-forward-selection.json')
    plan_sha = digest_file(directory/'plan.json')
    if state.get('plan_sha256') != plan_sha or result.get('plan_sha256') != plan_sha:
        raise ValueError('campaign plan digest differs')
    return {'directory': str(directory), 'files': len(index),
            'artifact_index_sha256': digest_file(index_path), 'plan_sha256': plan_sha}


def _bound_prior_selection(plan, path):
    path = Path(path).resolve()
    for protected in plan.get('protected_indices', []):
        root = Path(protected['directory']).resolve()
        if path.is_relative_to(root):
            relative = str(path.relative_to(root))
            records = [row for row in protected['files'] if row['path'] == relative]
            if len(records) != 1 or digest_file(path) != records[0]['sha256']:
                raise ValueError('old v2 selection differs from the paired plan')
            return root
    raise ValueError('old v2 selection is not bound by the paired plan')


def _residuals(row, units='normalized_rms'):
    return (row.get('independent_residuals') or {}).get(units) or {}


def _training_summary(raw):
    projections = raw.get('projections') or []
    final = projections[-1] if projections else {}
    return {'wall_seconds': raw.get('wall_seconds'),
        'coefficient_count': raw.get('coefficient_count'),
        'projection_count': len(projections) if 'projections' in raw else None,
        'final_projection_rank': final.get('rank'), 'final_projection_columns': final.get('columns'),
        'final_coefficient_norm': final.get('coefficient_norm_returned'),
        'final_projection_accepted': final.get('accepted'),
        'hidden_backtrack_count': raw.get('hidden_backtrack_count'),
        'hidden_rejected_updates': raw.get('hidden_rejected_updates'),
        'optimizer_resets': raw.get('optimizer_resets'),
        'hidden_accepted_objective_increases': raw.get('hidden_accepted_objective_increases')}


def _candidate_label(plan):
    explicit = plan.get('report_label')
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    identifier = str(plan.get('candidate_id') or '')
    versions = set(re.findall(r'(?:^|[^a-z0-9])(v[0-9]+)(?=$|[^a-z0-9])', identifier.lower()))
    return next(iter(versions)).upper() if len(versions) == 1 else 'Candidate'


def _method_label(comparison, version):
    return 'V2' if version == 'v2' else comparison['candidate_label']


def load_comparison(campaign_directory, prior_selection):
    evidence = validate_campaign(campaign_directory)
    directory = Path(campaign_directory).resolve()
    plan = _json(directory/'plan.json')
    prior_root = _bound_prior_selection(plan, prior_selection)
    result = _json(directory/'final-forward-selection.json')
    prior = _json(prior_selection)
    previous = {row['id']: row for row in prior.get('cases', [])}
    current = {row['id']: row for row in result.get('cases', [])}
    if len(current) != len(result.get('cases', [])) or len(previous) != len(prior.get('cases', [])):
        raise ValueError('duplicate case identifier in supplied evidence')
    rows = []
    for declared in plan['cases']:
        case_id = declared['id']
        saved = current.get(case_id, dict(declared, status='not_recorded'))
        old = previous.get(case_id, dict(declared, status='not_recorded'))
        for entry in (saved, old):
            if entry.get('geometry') != declared['geometry'] or entry.get('known_h') != declared['known_h']:
                raise ValueError('paired geometry or conductance differs for '+case_id)
        raw_path = directory/case_id/'training-report.json'
        audit_path = directory/case_id/'near-interface-audit.json'
        raw = _json(raw_path) if raw_path.is_file() else {}
        audit = _json(audit_path) if audit_path.is_file() else {'status': 'not_recorded', 'role': 'report only'}
        rows.append({'id': case_id, 'geometry': declared['geometry'], 'known_h': declared['known_h'],
            'v2': old, 'candidate': saved, 'training': _training_summary(raw), 'supplemental_audit': audit})
    return {'schema': 'paired_2d_continuity_development_report', 'campaign_status': result.get('status'),
        'candidate_id': plan.get('candidate_id'), 'candidate_label': _candidate_label(plan),
        'source_unchanged': result.get('source_unchanged'), 'source_sha256': result.get('source_sha256'),
        'plan_sha256': evidence['plan_sha256'], 'campaign_evidence': evidence,
        'prior_selection': {'path': str(Path(prior_selection).resolve()), 'sha256': digest_file(prior_selection)},
        'prior_root': str(prior_root), 'model_config': plan.get('model_config'),
        'wall_seconds': result.get('wall_seconds'), 'peak_rss_bytes': result.get('peak_rss_bytes'),
        'inverse_opened': result.get('inverse_opened'), 'fresh_evaluation_opened': result.get('fresh_evaluation_opened'),
        'rows': rows}


def _plots(output, comparison):
    with tempfile.TemporaryDirectory(prefix='thermotwin-continuity-plot-') as cache:
        os.environ.setdefault('MPLCONFIGDIR', cache)
        os.environ.setdefault('XDG_CACHE_HOME', cache)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        rows = comparison['rows']
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.5), constrained_layout=True)
        panels = ((axes[0, 0], lambda row: _residuals(row).get('interior'), 'Independent heat-equation RMS (normalized)', .05),
                  (axes[0, 1], lambda row: (row.get('field_metrics') or {}).get('rmse_K'), 'Hidden field RMS error (K)', .05),
                  (axes[1, 0], lambda row: _residuals(row).get('bottom_contact'), 'Bottom-contact RMS (normalized)', .05),
                  (axes[1, 1], lambda row: _residuals(row, 'physical_rms').get('interface_1_temperature'), 'Second interface temperature jump RMS (K)', .005))
        x = np.arange(len(rows))
        for axis, metric, title, reference in panels:
            for offset, version, color in ((-.19, 'v2', '#64748b'), (.19, 'candidate', '#0f766e')):
                values = [_number(metric(row[version])) for row in rows]
                axis.bar(x+offset, [np.nan if value is None else value for value in values], .36,
                         label=_method_label(comparison, version), color=color)
                for position, value, row in zip(x+offset, values, rows):
                    if value is None:
                        label = 'N/A' if version in row and row['geometry'] == 'full_contact' and axis is axes[1, 1] else row[version].get('status', 'not recorded')
                        axis.text(position, .01, label, transform=axis.get_xaxis_transform(),
                                  rotation=90, va='bottom', ha='center', fontsize=7)
            axis.axhline(reference, color='#b91c1c', linestyle='--', linewidth=1., label='Declared reference limit')
            axis.set_title(title, fontsize=10)
            axis.set_xticks(x, [f"{row['id']}\n{row['geometry'].replace('_contact','')} h={row['known_h']:g}" for row in rows], fontsize=8)
            axis.set_xlim(-.65, len(rows)-.35)
            axis.set_ylim(bottom=0)
            axis.grid(axis='y', alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Paired 2D forward development: saved metrics only\nSupplied gate decisions are reported separately; no new acceptance calculation', fontsize=13)
        names = ['comparison.png', 'comparison.pdf']
        fig.savefig(output/names[0], dpi=170)
        fig.savefig(output/names[1])
        plt.close(fig)
        return names


def _markdown(comparison):
    lines = [f"# Paired 2D continuity repair: {comparison['candidate_label']} development", '',
        f"Recorded campaign result: **{comparison.get('campaign_status', 'not recorded')}**.", '',
        'This report displays saved gate outcomes and metrics. It does not recompute acceptance, choose checkpoints, train models, or open new evaluation data. All four arms reuse the original v2 observations and reference fields.', '',
        f"Plan SHA256: `{comparison['plan_sha256']}`. Campaign index SHA256: `{comparison['campaign_evidence']['artifact_index_sha256']}`.", '',
        '## Saved outcomes', '',
        '| Case | Geometry | h W/(m² K) | Method | Status | Supplied overall gate | PDE normalized RMS | Field RMS K | Field max K | Observation MSE | Training wall s |',
        '| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in comparison['rows']:
        for version in ('v2', 'candidate'):
            entry = row[version]
            field = entry.get('field_metrics') or {}
            wall = row['training'].get('wall_seconds') if version == 'candidate' else entry.get('wall_seconds')
            lines.append(f"| {row['id']} | {row['geometry']} | {_fmt(row['known_h'])} | {_method_label(comparison, version)} | {entry.get('status', 'not recorded')} | {_pass((entry.get('gate') or {}).get('all'))} | {_fmt(_residuals(entry).get('interior'))} | {_fmt(field.get('rmse_K'))} | {_fmt(field.get('maximum_error_K'))} | {_fmt(entry.get('normalized_observation_mse'))} | {_fmt(wall)} |")
    lines.extend(['', 'Missing measurements and failed or unrun arms remain marked as such. A missing metric is never rendered as zero. Values are shown at six significant digits; full precision remains in the source JSON.', '',
        '## Independent boundary physics', '',
        'All values below are saved normalized RMS values. Full-contact bottom insulation is not applicable.', '',
        '| Case | Method | Bottom contact | Bottom insulated | Top flux | Left side | Right side |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
    for row in comparison['rows']:
        for version in ('v2', 'candidate'):
            residual = _residuals(row[version])
            insulated = 'not applicable' if row['geometry'] == 'full_contact' else _fmt(residual.get('bottom_insulated'))
            lines.append(f"| {row['id']} | {_method_label(comparison, version)} | {_fmt(residual.get('bottom_contact'))} | {insulated} | {_fmt(residual.get('top'))} | {_fmt(residual.get('left'))} | {_fmt(residual.get('right'))} |")
    lines.extend(['', '## Independent interface continuity', '',
        'These are the primary unchanged diagnostic points. Full-contact arms have no internal interfaces. Shared traces are still checked explicitly; exact construction does not replace the independent heat-equation gate.', '',
        '| Case | Method | Interface | Temperature RMS K | Fourier-flux RMS W/m² | Temperature normalized RMS | Flux normalized RMS |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |'])
    for row in comparison['rows']:
        if row['geometry'] == 'full_contact':
            continue
        for version in ('v2', 'candidate'):
            normalized, physical = _residuals(row[version]), _residuals(row[version], 'physical_rms')
            for interface in (0, 1):
                prefix = f'interface_{interface}_'
                lines.append(f"| {row['id']} | {_method_label(comparison, version)} | {interface} | {_fmt(physical.get(prefix+'temperature'))} | {_fmt(physical.get(prefix+'flux'))} | {_fmt(normalized.get(prefix+'temperature'))} | {_fmt(normalized.get(prefix+'flux'))} |")
    lines.extend(['', '## Recorded solver conditioning and cost', '',
        'Rank and coefficient norm refer to the final recorded readout projection, not a selected best projection. Backtracks and rejected updates are the supplied training totals. Training wall time excludes subsequent independent diagnostics and field scoring.', '',
        '| Case | Training wall s | Projection count | Final rank / columns | Final coefficient norm | Final projection accepted | Hidden backtracks | Rejected hidden updates | Optimizer resets | Accepted objective increases |',
        '| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |'])
    for row in comparison['rows']:
        train = row['training']
        rank = f"{_fmt(train.get('final_projection_rank'))} / {_fmt(train.get('final_projection_columns'))}"
        lines.append(f"| {row['id']} | {_fmt(train.get('wall_seconds'))} | {_fmt(train.get('projection_count'))} | {rank} | {_fmt(train.get('final_coefficient_norm'))} | {_pass(train.get('final_projection_accepted'))} | {_fmt(train.get('hidden_backtrack_count'))} | {_fmt(train.get('hidden_rejected_updates'))} | {_fmt(train.get('optimizer_resets'))} | {_fmt(train.get('hidden_accepted_objective_increases'))} |")
    lines.extend(['', f"Recorded complete campaign elapsed time: **{_fmt(comparison.get('wall_seconds'))} s**. Recorded peak process RSS: **{_fmt(comparison.get('peak_rss_bytes'))} bytes**. Source unchanged: **{comparison.get('source_unchanged', 'not recorded')}**.", '',
        '## Supplemental near-interface audit — report only', '',
        'This separate audit never changes the primary gates, training objective, or checkpoint selection. Its coordinates are closer to the interfaces than the original acceptance grid; full-contact arms are not applicable. A failed audit is reported independently.', '',
        '| Case | Audit status | PDE normalized RMS | Interface 0 temperature K | Interface 0 flux W/m² | Interface 1 temperature K | Interface 1 flux W/m² |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
    for row in comparison['rows']:
        audit = row['supplemental_audit']
        diagnostics = audit.get('diagnostics') or {}
        normalized = diagnostics.get('normalized_rms') or {}
        physical = diagnostics.get('physical_rms') or {}
        display = lambda key: 'not applicable' if audit.get('status') == 'not_applicable' else _fmt(physical.get(key))
        pde = 'not applicable' if audit.get('status') == 'not_applicable' else _fmt(normalized.get('interior'))
        lines.append(f"| {row['id']} | {audit.get('status', 'not recorded')} | {pde} | {display('interface_0_temperature')} | {display('interface_0_flux')} | {display('interface_1_temperature')} | {display('interface_1_flux')} |")
        if audit.get('error'):
            lines.extend(['', f"{row['id']} audit error: `{audit['error']}`.", ''])
    lines.extend(['', '## Interpretation limits and provenance', '',
        'This is a known-conductance forward development comparison. It establishes no inverse recovery, unseen-cohort success, PINN advantage, or 3D validation. The repair changes both interface representation and PDE solution conditioning, so their individual effects are not isolated.', '',
        f"Inverse opened: **{comparison.get('inverse_opened', 'not recorded')}**. Fresh evaluation opened: **{comparison.get('fresh_evaluation_opened', 'not recorded')}**.", '',
        f"Candidate identifier: `{comparison.get('candidate_id', 'not recorded')}`. Old v2 selection SHA256: `{comparison['prior_selection']['sha256']}`. The selection is bound to the protected evidence records in the saved candidate plan.", '',
        'The comparison image displays the supplied values with the declared reference limits. The saved coordinator gate is the sole source of the reported pass/fail labels.', '',
        '![Saved metric comparison](comparison.png)', ''])
    for row in comparison['rows']:
        error = row['candidate'].get('error') or row['candidate'].get('reason')
        if error:
            lines.extend([f"{row['id']} recorded failure or nonexecution reason: `{error}`.", ''])
    return '\n'.join(lines)


def render_report(campaign_directory, *, prior_selection, output_directory):
    comparison = load_comparison(campaign_directory, prior_selection)
    output = Path(output_directory).resolve()
    for protected in (Path(campaign_directory).resolve(), Path(comparison['prior_root'])):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('report output must be separate from preserved campaign directories')
    output.mkdir(parents=True, exist_ok=False)
    names = _plots(output, comparison)
    (output/'REPORT.md').write_text(_markdown(comparison))
    _write(output/'result-summary.json', comparison)
    names += ['REPORT.md', 'result-summary.json']
    records = [{'path': name, 'sha256': digest_file(output/name), 'bytes': (output/name).stat().st_size}
               for name in sorted(names)]
    index = {'files': records, 'digest': canonical_digest(records),
             'campaign_artifact_index_sha256': comparison['campaign_evidence']['artifact_index_sha256'],
             'plan_sha256': comparison['plan_sha256'], 'renderer_sha256': digest_file(__file__)}
    _write(output/'report-index.json', index)
    if validate_campaign(campaign_directory) != comparison['campaign_evidence']:
        raise ValueError('campaign changed while rendering')
    return {'report_directory': str(output), 'report_index_digest': index['digest'],
            'report_index_sha256': digest_file(output/'report-index.json'),
            'campaign_status': comparison['campaign_status'], 'files': len(records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Render sealed paired 2D development evidence without fitting.')
    parser.add_argument('--campaign-directory', required=True)
    parser.add_argument('--prior-selection', required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(render_report(args.campaign_directory,
        prior_selection=args.prior_selection, output_directory=args.output_directory), allow_nan=False))


if __name__ == '__main__':
    main()
