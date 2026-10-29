"""Render the three final study figures from compact, authenticated inputs.

No archive, ThermoTwin numerical module, fitting or simulation is needed.
Writes PNG, PDF, SVG and a file-hash manifest into a NEW output directory.
"""

import argparse
import json
from pathlib import Path

from tools import verify_prospective_phase_f_result as audit
from tools.operating_decision_final_figure_data import DATA_DOMAIN


LABELS = {'prospective_selector': 'Selector', 'stop_now': 'Stop', 'fixed_thermal': 'Thermal',
          'fixed_voltage': 'Voltage', 'fixed_face_temperature': 'Face'}
COLORS = {'prospective_selector': '#242B36', 'stop_now': '#7D8794', 'fixed_thermal': '#009E73',
          'fixed_voltage': '#0072B2', 'fixed_face_temperature': '#CC79A7', 'selection_failure': '#B22222'}
MODEL_COLORS = {'four_state': '#0072B2', 'five_state': '#D55E00'}
MODEL_LABELS = {'four_state': 'Four-state model', 'five_state': 'Five-state model'}


def validate_inputs(data):
    audit.require(data['artifact_digest'] == audit.digest(DATA_DOMAIN, {k: v for k, v in data.items() if k != 'artifact_digest'}), 'figure inputs were modified')
    audit.require(data['figure_2']['primary_success'] is False and not data['figure_2']['qualifying_fixed_policies'], 'primary interpretation differs')
    audit.require([r['procedure'] for r in data['figure_2']['procedures']] == list(audit.PROCEDURES), 'missing comparison procedure')
    audit.require(all(len(r['final_interval_widths_kelvin']) == 300 for r in data['figure_2']['procedures']), 'interval-width denominator differs')
    audit.require(data['figure_3']['map_cell_count'] == len(data['figure_3']['cells']) == 48 and
                  data['figure_3']['frozen_design_correction_q_applied'] is False, 'development/reserved scope differs')


def title(ax, text):
    ax.set_title(text, loc='left', fontsize=11, fontweight='bold', pad=13)


def temperature_panel(ax, band, values):
    low = min(min(v) for v in values)
    high = max(max(v) for v in values)
    lower = min(low, band['upper']) - .12
    upper = max(high, band['upper']) + .17
    ax.set_ylim(lower, upper)
    ax.axhspan(max(lower, band['lower']), band['upper'], color='#009E73', alpha=.07)
    ax.axhline(band['upper'], color='#505050', ls='--', lw=1.1, label='Upper operating limit')
    ax.set_xlabel('Time in target schedule (s)')
    ax.set_ylabel('Cold-face temperature (K)')


def decision_story(data, plt):
    story = data['figure_1']
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.4))
    fig.subplots_adjust(left=.065, right=.98, top=.86, bottom=.16, hspace=.57, wspace=.35)
    fig.suptitle('An illustrative case: uncertainty narrows, but the decision remains unresolved', x=.04, y=.975, ha='left', fontsize=17, fontweight='bold')
    fig.text(.04, .925, f'Synthetic reserved block {story["block"]} · temperature-dependent contact · thermal acquisition selected · post hoc illustration', fontsize=11, color='#444444')
    channel = 'cold_exchanger_temperature'
    ax = axes[0, 0]
    obs = story['initial_observations'][channel]
    ax.scatter(obs['time'], obs['value'], s=10, color='#6C7177', alpha=.65, label='Saved measurements', zorder=4)
    for curve in story['initial_model_curves']:
        points = curve['acquisition'][channel]
        ax.plot(points['time'], points['value'], color=MODEL_COLORS[curve['model']], lw=1.7, label=MODEL_LABELS[curve['model']])
    title(ax, 'A  Similar fits to the initial measured channel')
    ax.set(xlabel='Initial acquisition time (s)', ylabel='Cold exchanger (K)')
    ax.legend(fontsize=8, frameon=False, loc='best')

    ax = axes[0, 1]
    values = []
    for curve in story['initial_model_curves']:
        ax.plot(curve['target_time'], curve['target_cold_face'], color=MODEL_COLORS[curve['model']], lw=1.8, label=MODEL_LABELS[curve['model']])
        values.append(curve['target_cold_face'])
    temperature_panel(ax, story['operating_band_kelvin'], values)
    title(ax, 'B  Initial forecasts of the hidden cold face')
    ax.legend(fontsize=8, frameon=False, loc='lower right')

    ax = axes[0, 2]
    evaluations = [v for v in story['action_evaluations'] if v['policy_name'] != 'stop_now']
    xs = list(range(3))
    vals = [v['utility_per_cost'] for v in evaluations]
    ax.bar(xs, vals, color=[COLORS[v['policy_name']] for v in evaluations], width=.60)
    for x, value in zip(xs, vals):
        ax.text(x, value + .002, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    ax.axhline(story['minimum_utility_per_cost'], color='#4D4D4D', lw=1, ls='--', label='Frozen value threshold')
    ax.set_xticks(xs, [LABELS[v['policy_name']] for v in evaluations])
    ax.set_ylabel('Expected width reduction / normalized cost (K)')
    ax.set_ylim(min(0., min(vals)) - .004, max(vals) * 1.27)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    title(ax, 'C  Select the highest frozen value per cost')

    ax = axes[1, 0]
    obs = story['added_run_observations'][channel]
    ax.scatter(obs['time'], obs['value'], s=10, color='#6C7177', alpha=.65, zorder=4, label='Saved added-run measurements')
    for curve in story['initial_model_curves']:
        points = curve['added_run_forecast'][channel]
        ax.plot(points['time'], points['value'], color=MODEL_COLORS[curve['model']], lw=1.6, label=MODEL_LABELS[curve['model']])
    title(ax, 'D  First of three added thermal runs')
    ax.set(xlabel='Added 0.6 A / 30 s pulse: within-run time (s)', ylabel='Cold exchanger (K)')
    ax.legend(fontsize=8, frameon=False, loc='best')

    ax = axes[1, 1]
    values = [story['saved_target_truth']['cold_face']]
    for curve in story['verified_model_curves']:
        ax.plot(curve['target_time'], curve['target_cold_face'], color=MODEL_COLORS[curve['model']], lw=1.6, label=MODEL_LABELS[curve['model']])
        values.append(curve['target_cold_face'])
    ax.plot(story['saved_target_truth']['time'], story['saved_target_truth']['cold_face'], color='#111111', lw=2, ls=':', label='Saved target truth')
    temperature_panel(ax, story['operating_band_kelvin'], values)
    title(ax, 'E  Verified forecasts and the saved target reveal')
    ax.legend(fontsize=8, frameon=False, loc='lower right')

    ax = axes[1, 2]
    intervals = [('Initial envelope', story['initial_interval'], '#657181'),
                 ('Verified after thermal', story['raw_after_interval'], COLORS['fixed_thermal']),
                 ('Final calibrated interval', story['final_interval'], '#B57B00')]
    for y, (label, interval, color) in zip((2, 1, 0), intervals):
        ax.hlines(y, interval['lower'], interval['upper'], color=color, lw=4)
        ax.plot([interval['lower'], interval['upper']], [y, y], '|', color=color, ms=12, mew=1.6)
    ax.axvline(0, color='#444444', ls='--', lw=1.1, label='Decision boundary')
    ax.axvline(story['saved_target_truth']['margin'], color='#111111', lw=1.2, ls=':', label='Saved true margin')
    ax.set_yticks((2, 1, 0), [v[0] for v in intervals], fontsize=8)
    ax.set(xlabel='Operating margin (K)', ylim=(-.5, 2.5))
    ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.02))
    title(ax, 'F  Calibration leaves an abstention')
    before = story['initial_interval']['upper'] - story['initial_interval']['lower']
    after = story['raw_after_interval']['upper'] - story['raw_after_interval']['lower']
    fig.text(.04, .082, f'Raw envelope width {before:.3f} → {after:.3f} K; selector correction q = {story["final_correction_kelvin"]:.4f} K per endpoint. '
             f'True margin = {story["saved_target_truth"]["margin"]:.3f} K; calibrated decision: ABSTAIN.', fontsize=10)
    fig.text(.04, .043, 'Illustrative calibration-induced abstention, one of 27 such cases among 137 abstentions. No refitting or new truth was generated for this figure.\n'
             'Initial acquisition fits include profiled sensor offsets; added-run forecasts use zero-mean new-run offsets. Full operating band: 285.0–301.7 K.', fontsize=9, color='#4A4A4A', linespacing=1.5)
    return fig


def comparison(data, plt):
    rows = data['figure_2']['procedures']
    fig, axes = plt.subplots(3, 2, figsize=(14.3, 12.9))
    fig.subplots_adjust(left=.065, right=.98, top=.905, bottom=.125, hspace=.57, wspace=.28)
    fig.suptitle('Reserved comparison: energy savings did not establish primary success', x=.04, y=.98, ha='left', fontsize=17, fontweight='bold')
    fig.text(.04, .945, 'Synthetic study · 100 paired blocks · 300 cases per procedure · no procedure qualified on the predeclared quality requirements', fontsize=11, color='#444444')
    labels = [LABELS[r['procedure']] for r in rows]
    colors = [COLORS[r['procedure']] for r in rows]
    xs = list(range(5))
    def ticks(ax):
        ax.set_xticks(xs, labels, fontsize=9)
    ax = axes[0, 0]
    rates = [100 * r['observed']['decision_coverage']['estimate'] for r in rows]
    ax.bar(xs, rates, color=colors, width=.62, alpha=.85, label='Observed definitive decisions')
    ax.scatter(xs, [100 * r['decision_coverage_lower'] for r in rows], s=35, marker='D', color='black', zorder=5, label='Simultaneous lower bound')
    for x, rate, row in zip(xs, rates, rows):
        ax.text(x, rate + 2, f'{row["observed"]["definitive_decisions"]}/300', ha='center', fontsize=9)
    ax.axhline(70, color='#A84835', ls='--', lw=1, label='Required lower bound: 70%')
    ax.set(ylabel='Definitive decisions (%)', ylim=(0, 102))
    ticks(ax); title(ax, 'A  Decision coverage falls short')
    ax.legend(frameon=False, fontsize=8, loc='upper left', ncols=1)

    ax = axes[0, 1]
    observed = [100 * r['observed']['false_approval_risk']['estimate'] for r in rows]
    ax.bar(xs, observed, color=colors, width=.62, alpha=.85, label='Observed false approvals / approvals')
    ax.scatter(xs, [100 * r['approval_risk_bounds']['upper'] for r in rows], marker='D', s=35, color='black', zorder=5, label='Simultaneous upper bound')
    for x, rate, row in zip(xs, observed, rows):
        ax.text(x, rate + 1.2, f'{row["observed"]["false_approvals"]}/{row["observed"]["approvals"]}', ha='center', fontsize=9)
    ax.axhline(10, color='#A84835', ls='--', lw=1, label='Required upper bound: 10%')
    ax.set(ylabel='False approvals conditional on approval (%)', ylim=(0, 55))
    ticks(ax); title(ax, 'B  Approval-risk assurance remains inconclusive')
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    ax = axes[1, 0]
    ax.bar([x - .17 for x in xs], [100 * r['observed']['case_interval_coverage']['estimate'] for r in rows], width=.32, color='#7E9CB6', label='Individual intervals (300)')
    ax.bar([x + .17 for x in xs], [100 * r['block_interval_coverage']['estimate'] for r in rows], width=.32, color='#303F53', label='All three intervals in a block (100)')
    for x, row in zip(xs, rows):
        case = 100 * row['observed']['case_interval_coverage']['estimate']
        block = 100 * row['block_interval_coverage']['estimate']
        ax.text(x - .17, case + 2, f'{case:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(x + .17, block + 2, f'{block:.0f}', ha='center', va='bottom', fontsize=8)
    ticks(ax); ax.set(ylabel='Observed interval coverage (%)', ylim=(0, 111))
    ax.set_yticks((0, 25, 50, 75, 100))
    title(ax, 'C  Interval coverage is a separate endpoint')
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.16), ncols=2)

    ax = axes[1, 1]
    energy = [r['observed']['realized_resources']['total_diagnostic_energy']['mean'] for r in rows]
    ax.bar(xs, energy, width=.62, color=colors)
    for x, value in zip(xs, energy):
        ax.text(x, value + 2, f'{value:.2f}', ha='center', fontsize=9)
    ticks(ax); ax.set(ylabel='Mean simulated diagnostic energy (J)', ylim=(0, 116))
    title(ax, 'D  Lower energy than the measurement comparators')
    ax.text(.01, .96, 'Acquisition + verification; stop uses the least energy', transform=ax.transAxes, fontsize=8, va='top')

    ax = axes[2, 0]
    bp = ax.boxplot([r['final_interval_widths_kelvin'] for r in rows], positions=xs, widths=.52,
                    whis=(0, 100), showfliers=False, patch_artist=True, manage_ticks=False,
                    medianprops={'color': '#111111', 'linewidth': 1.6}, whiskerprops={'color': '#646464'}, capprops={'color': '#646464'})
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color, alpha=.65)
    ticks(ax); ax.set_ylabel('Final calibrated margin interval width (K)')
    title(ax, 'E  Uncertainty retained for all 300 cases')

    ax = axes[2, 1]
    ax.axis('off'); title(ax, 'F  Instruments, schedule and software computation')
    body = []
    for row in rows:
        res = row['observed']['realized_resources']
        body.append([LABELS[row['procedure']], f'{res["diagnostic_run_count"]["mean"]:.2f}',
                     f'{res["energized_schedule_time_seconds"]["mean"]:.1f}', f'{res["extra_sensor_count"]["mean"]:.2f}'])
    table = ax.table(cellText=body, colLabels=['Procedure', 'Mean runs', 'Energized s', 'Extra sensors'],
                     cellLoc='center', bbox=[0, .40, 1, .60], colWidths=[.28, .23, .25, .24])
    table.auto_set_font_size(False); table.set_fontsize(8.5)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#D4D8DD')
        if i == 0: cell.set_facecolor('#E9EEF2'); cell.set_text_props(weight='bold')
    comp = data['figure_2']['compute']
    execution = comp['execution_invocation']
    ax.text(0, .31, f'Whole reserved benchmark: {comp["completed_block_cpu_seconds"]/3600:.2f} CPU h; '
            f'{execution["wall_seconds_this_invocation"]/3600:.2f} h execution wall.\n'
            f'Sampled execution peak: {execution["process_tree_peak_rss_bytes"]/1024**3:.3f} GiB; four workers.\n'
            'Includes every comparator and predictive draw.\nSingle-case deployed latency was not benchmarked.', transform=ax.transAxes, fontsize=9, va='top', linespacing=1.65)
    fig.text(.04, .068, 'Observed false rejections: zero for every procedure. Bounds use the frozen 73-direction simultaneous allocation; the 300 cases are not independent replicates.\n'
             'Approval-risk upper bounds represent uncertainty, not observed error rates. Boxes show medians/IQRs; whiskers span the complete min–max range.', fontsize=9.2, linespacing=1.6)
    fig.text(.04, .025, 'Diagnostic joules are simulated acquisition/verification energy, excluding computer electricity, resets, sensor electronics and hardware validation. Energized time is not full bench time.', fontsize=8.8, color='#4A4A4A')
    return fig


def measurement_map(data, plt):
    from matplotlib.patches import Patch, Rectangle
    values = data['figure_3']
    fig = plt.figure(figsize=(15.2, 9.2))
    ax = fig.add_axes([.225, .235, .75, .51])
    fig.suptitle('Development measurement map: choices respond to sensor quality and assumed costs', x=.035, y=.975, ha='left', fontsize=17, fontweight='bold')
    fig.text(.035, .925, '48 declared cells · 20 paired development blocks / 60 cases per cell · development padding only, before final calibration', fontsize=11, color='#444444')
    handles = [Patch(facecolor=COLORS[p], label=LABELS[p]) for p in audit.POLICIES]
    handles.append(Patch(facecolor='white', edgecolor='#555555', hatch='////', label='Abstaining cases within each action'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.60, .895), ncols=5, frameon=False, fontsize=10)
    cells = {(v['sensor_scenario'], v['cost_scenario']): v for v in values['cells']}
    sensor_labels = [
        'Nominal\nσV = 0.002 V; σface = 0.02 K\nProbe capacity = 5 J/K',
        'Higher voltage noise\nσV = 0.004 V (2×)',
        'Higher face noise\nσface = 0.04 K (2×)',
        'Higher probe loading\nProbe capacity = 10 J/K (2×)']
    groups = [('Balanced', '⅓ / ⅓ / ⅓'), ('Energy weighted', '0.70 / 0.20 / 0.10'),
              ('Time weighted', '0.20 / 0.70 / 0.10'), ('Instrument weighted', '0.15 / 0.15 / 0.70')]
    for g, (name, weights) in enumerate(groups):
        ax.add_patch(Rectangle((g * 3, 0), 3, 4, facecolor='#F1F3F5' if g % 2 == 0 else '#FFFFFF', zorder=0))
        ax.text(g * 3 + 1.5, 4.45, name + '\nE / T / I = ' + weights, ha='center', va='center', fontsize=10, fontweight='bold')
    for j in range(12):
        ax.text(j + .5, 4.08, ('¼×', '1×', '4×')[j % 3], ha='center', va='center', fontsize=10)
    for i, sensor in enumerate(values['sensor_scenarios']):
        y = 3.5 - i
        ax.text(-.22, y, sensor_labels[i], ha='right', va='center', fontsize=10, linespacing=1.45)
        for j, cost in enumerate(values['cost_scenarios']):
            cell = cells[sensor['name'], cost['name']]
            left, total_width = j + .06, .88
            ax.add_patch(Rectangle((j + .02, y - .37), .96, .78, facecolor='none', edgecolor='#D5D9DD', linewidth=.6))
            for item in cell['actions']:
                count = item['selected']
                if not count: continue
                width = total_width * count / 60
                ax.add_patch(Rectangle((left, y + .02), width, .30, facecolor=COLORS[item['action']], edgecolor='white', linewidth=.35))
                abstain_width = total_width * item['abstentions'] / 60
                if abstain_width:
                    ax.add_patch(Rectangle((left + width - abstain_width, y + .02), abstain_width, .30,
                                           facecolor=(1, 1, 1, .22), edgecolor='#3B3B3B', hatch='////', linewidth=0))
                left += width
            counts = {v['action']: v['selected'] for v in cell['actions']}
            ax.text(j + .50, y - .09, f'S{counts["stop_now"]} T{counts["fixed_thermal"]} V{counts["fixed_voltage"]} F{counts["fixed_face_temperature"]}',
                    ha='center', va='center', fontsize=7)
            ax.text(j + .50, y - .27, f'D {cell["definitive"]}/60', ha='center', va='center', fontsize=9)
            if cell['ineligible_actions']['fixed_face_temperature']:
                ax.plot(j + .88, y + .36, marker='^', color='#B22222', markersize=4)
            if cell['primary']:
                ax.add_patch(Rectangle((j + .02, y - .37), .96, .78, facecolor='none', edgecolor='#202020', linewidth=2))
    ax.set(xlim=(0, 12), ylim=(0, 4)); ax.axis('off')
    fig.text(.225, .835, 'Face-instrument price relative to voltage: ¼× / 1× / 4×. Weight triples below are energy / time / instrumentation.', fontsize=9, color='#555555')
    fig.text(.035, .168, 'Each small bar totals 60 cases; counts below are Stop / Thermal / Voltage / Face selections (S/T/V/F). Hatching shows abstention; D is the definitive total.\n'
             'Selection, verification and pipeline failures: 0/60 in every cell. ▲ Face action ineligible in 1/60 cases throughout the high-loading row; every case remains counted.', fontsize=9.6, linespacing=1.65)
    fig.text(.035, .092, 'Black outline: primary nominal / balanced / 1× setting. Nominal maps reuse D1 observations; stress rows used actual predeclared noise/loading computations.\n'
             'Cells share development cases and are not independent replications. These counts have no final calibration q or reserved risk guarantee.', fontsize=9.6, linespacing=1.65)
    fig.text(.035, .027, 'Reserved confirmation covers only the nominal / balanced / 1× setting, reported in Figure 2. Cost scenarios assume 80 s reset per added run; probe response is 2.5 s.', fontsize=9.4, color='#444444')
    return fig


def render(args):
    data = json.loads(args.inputs.read_bytes())
    validate_inputs(data)
    audit.require(not args.output_directory.exists(), 'refuse to reuse a figure output directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit.require(matplotlib.__version__ == '3.10.8', 'use the recorded Matplotlib 3.10.8 renderer')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .15,
                         'axes.axisbelow': True, 'pdf.fonttype': 42,
                         'svg.hashsalt': 'thermotwin-operating-decision-final-figures-v1'})
    args.output_directory.mkdir(parents=True)
    records = []
    for stem, build in (('figure_1_decision_story', decision_story), ('figure_2_reserved_comparison', comparison), ('figure_3_development_map', measurement_map)):
        fig = build(data, plt)
        outputs = []
        for extension in ('png', 'pdf', 'svg'):
            path = args.output_directory / (stem + '.' + extension)
            metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {'Date': None} if extension == 'svg' else {}
            fig.savefig(path, dpi=180, facecolor='white', metadata=metadata)
            outputs.append({'path': path.name, 'format': extension, **audit.file_record(path)})
        records.append({'figure': stem, 'size_inches': list(fig.get_size_inches()), 'outputs': outputs})
        plt.close(fig)
    manifest = {'schema_version': 1, 'version': 'operating_decision_final_figures_v1',
                'figure_input_artifact_digest': data['artifact_digest'], 'figure_input_file': audit.file_record(args.inputs),
                'renderer_file': audit.file_record(__file__), 'matplotlib_version': matplotlib.__version__,
                'reserved_scientific_result_digest': data['reserved_scientific_result_digest'],
                'development_scientific_result_digest': data['development_scientific_result_digest'],
                'figures': records, 'new_fits_simulations_or_observations_in_renderer': False,
                'primary_success': False, 'development_map_is_not_reserved_confirmation': True}
    manifest['artifact_digest'] = audit.digest('thermotwin.operating_decision.final_figure_manifest_v1', manifest)
    with (args.output_directory / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'directory': str(args.output_directory), 'figures': 3, 'export_files': 9,
                      'manifest_digest': manifest['artifact_digest']}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    render(parser.parse_args(argv))


if __name__ == '__main__':
    main()
