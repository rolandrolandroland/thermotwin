"""Execute or validate the frozen spatial PINN study and render its report."""

import argparse
import json
from pathlib import Path


def _number(value, scale=1.):
    return '—' if value is None else f'{value*scale:.4g}'


def render_planar_field(directory, trials):
    import matplotlib.pyplot as plt
    import numpy as np
    root = directory/'bounded-2d'
    reference_path = root/'evaluation/reference/hidden_reference_fields.npz'
    if not reference_path.exists() or not trials:
        return
    with np.load(reference_path) as data:
        reference = data['training_temperature'][-1]
    panels = [('Truth', reference)]
    for name in ('conventional', 'inverse_pinn', 'data_only'):
        artifact = trials[0].get('methods', {}).get(name, {}).get('artifacts', {}).get('predictions')
        if artifact and (root/artifact).is_file():
            with np.load(root/artifact) as data:
                panels.append((name.replace('_', ' '), data['training_field_temperature'][-1]))
    if len(panels) == 1:
        return
    figure, axes = plt.subplots(2, len(panels), figsize=(3.5*len(panels), 4.5),
                                constrained_layout=True, squeeze=False)
    minimum, maximum = float(reference.min()), float(reference.max())
    error_limit = max(.05, max(float(np.max(np.abs(field-reference))) for _, field in panels))
    for column, (name, field) in enumerate(panels):
        image = axes[0, column].imshow(field, origin='lower', extent=[0, 40, 0, 4], aspect='auto',
                                      vmin=minimum, vmax=maximum, cmap='viridis')
        error = axes[1, column].imshow(field-reference, origin='lower', extent=[0, 40, 0, 4], aspect='auto',
                                      vmin=-error_limit, vmax=error_limit, cmap='coolwarm')
        axes[0, column].set_title(name)
        for row in (0, 1):
            axes[row, column].set_xlabel('x (mm)')
            axes[row, column].set_ylabel('z (mm)')
    figure.colorbar(image, ax=list(axes[0]), label='Temperature (K)', shrink=.8)
    figure.colorbar(error, ax=list(axes[1]), label='Error relative to truth (K)', shrink=.8)
    figure.suptitle('Evaluation trial 0 at 120 s — one illustration; all trials remain in the table')
    figure.savefig(directory/'planar-final-field.png', dpi=160)
    plt.close(figure)


def render_report(directory, result):
    """Render saved results without making any scientific acceptance decision."""
    directory = Path(directory)
    stages = result['stages']
    fresh = stages.get('fresh_1d', {})
    planar = stages.get('bounded_2d', {})
    lines = ['# Spatial PINN validation', '',
        f"Campaign: `{result['campaign']}`. Source: `{result['source_commit']}`.", '',
        f"Execution status: **{result['status']}**. Completion is not scientific acceptance.", '',
        'This is synthetic software validation. It does not establish hardware performance '
        'or a general advantage for PINNs.', '', '## Fresh 1D validation', '']
    for label, key in (('Planned trials', 'denominator'), ('Completed trials', 'completed_count'),
                       ('Operationally accepted', 'operational_accepted_count'),
                       ('Denser-grid accepted', 'denser_grid_accepted_count'),
                       ('Knot-shape recovered', 'knot_shape_recovered_count'),
                       ('In-support transfer accepted', 'in_support_transfer_accepted_count'),
                       ('Combined gate accepted', 'combined_gate_accepted_count')):
        lines.append(f"- {label}: {fresh.get(key, 'not available')}")
    if fresh.get('blocked_reason'):
        lines += ['', f"Stopped before fitting: `{fresh['blocked_reason']}`."]
    lines += ['', 'The independent cubic truth, three fixed training regimes and material remain '
              'predeclared. Noise and neural initializations are fresh. Curve shape and truth-blind '
              'operational criteria are reported separately.', '', '## Verified 2D and bounded inverse benchmark', '',
              f"2D status: `{planar.get('status', 'not run')}`.", '',
              'The plate has a single unknown contact conductance. The data-only network has no '
              'contact estimate; the known-contact PINN is an oracle diagnostic. Frozen-parameter '
              'transfer uses the conventional forward solver under a new schedule.', '']
    if planar:
        compact = {key: planar[key] for key in ('summary', 'contact_recovery_claim_allowed',
                   'development_selection') if key in planar}
        lines += ['```json', json.dumps(compact, sort_keys=True, indent=2, allow_nan=False), '```', '']
    trials = planar.get('trials', [])
    if trials:
        lines += ['Every planned method and trial, including failures:', '',
                  '| Trial | Method | Status | Field RMS (K) | Max (K) | h error (%) | Error reduction (%) | Holdout RMS (K) |',
                  '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for trial in trials:
            for name, method in trial.get('methods', {}).items():
                metrics = method.get('metrics', {})
                field = metrics.get('hidden_field', {})
                holdout = (metrics.get('holdout_transfer') or {}).get('field', {})
                lines.append(f"| {trial['index']} | {name} | {metrics.get('status', method.get('status', 'unknown'))} "
                    f"| {_number(field.get('rmse_kelvin'))} | {_number(field.get('max_abs_kelvin'))} "
                    f"| {_number(metrics.get('relative_contact_error'), 100)} "
                    f"| {_number(metrics.get('fraction_absolute_error_reduction'), 100)} "
                    f"| {_number(holdout.get('rmse_kelvin'))} |")
        lines += ['', 'Parameter-error reduction is relative to the frozen initial guess, '
                  'not relative to the conventional fitted baseline. A broad 10% tolerance '
                  'alone does not establish learned recovery. Full gates and all errors are '
                  'retained in `bounded-2d/result.json`.', '']
    geometry = stages.get('geometry_3d_assessment', {})
    lines += ['## 3D assessment', '', geometry.get('basis', 'Not assessed.'), '',
              'The benchmark plate is synthetic. Its full-depth contact, insulated depth faces, '
              'uniform materials and loads justify 2D. Localized heating, a finite-depth contact '
              'or individual module legs would require a new geometry-specific study.', '',
              'The manifests record the frozen source, protocol, runtime and seeds. '
              '`evidence-index.json` hashes every final artifact; validate it before sharing results.', '']
    (directory/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    # Plot only recorded trial-level summaries; raw 1D/2D histories remain intact.
    if fresh.get('trials') or trials:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    if fresh.get('trials'):
        names, counts = [], []
        for label, key in (('Operational', 'operational_accepted_count'),
                           ('Denser residuals', 'denser_grid_accepted_count'),
                           ('Knot shape', 'knot_shape_recovered_count'),
                           ('Transfer', 'in_support_transfer_accepted_count'),
                           ('Combined', 'combined_gate_accepted_count')):
            names.append(label)
            counts.append(fresh.get(key, 0))
        figure, axis = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
        axis.bar(names, counts, color='#267e92')
        axis.set_ylim(0, fresh['denominator'])
        axis.set_yticks(range(fresh['denominator']+1))
        axis.set_ylabel(f"Accepted trials / {fresh['denominator']}")
        axis.set_title('Fresh 1D synthetic validation: separate acceptance criteria')
        figure.savefig(directory/'fresh-1d-gates.png', dpi=160)
        plt.close(figure)
    if trials:
        render_planar_field(directory, trials)


def main():
    from ..studies.pinn_spatial_validation import run_spatial_campaign, validate_evidence
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    run = subcommands.add_parser('run')
    run.add_argument('--output-directory', required=True)
    run.add_argument('--source-commit', required=True)
    run.add_argument('--ci-evidence', required=True)
    validate = subcommands.add_parser('validate')
    validate.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    if args.command == 'run':
        result = run_spatial_campaign(args.output_directory, source_commit=args.source_commit,
                                      ci_evidence=args.ci_evidence)
        print(json.dumps({'status': result['status'], 'output_directory': args.output_directory}))
    else:
        print(json.dumps(validate_evidence(args.output_directory), sort_keys=True))


if __name__ == '__main__':
    main()
