"""Checkpointed final evaluation with durable calibrated decisions before reveal."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, process_time
from typing import Mapping

from . import operating_decision_prospective_phase_e_execution as e
from .operating_decision_prospective import ProspectiveSelectorRule
from .operating_decision_prospective_calibration_freeze import load_calibration_freeze, frozen_corrections
from .operating_decision_prospective_phase_e import PHASE_E_SOURCE_PATHS
from .operating_decision_prospective_phase_e_analysis import analyze_primary_comparison
from .operating_decision_prospective_phase_e_calibration import (
    SECONDARY_CLEARANCES_KELVIN, calibrated_outcome,
)
from .operating_decision_prospective_phase_f import (
    PHASE_F_DOMAIN, PHASE_F_VERSION, PHASE_F_REHEARSAL_PARTITION, PHASE_F_SOURCE_PATHS,
    RULE_DIGEST, MATH_ENVIRONMENT, load_complete_rule, phase_f_protocol_digest, phase_f_protocol_payload,
)
from .operating_decision_prospective_phase_f_decisions import (
    BLINDED_FIELDS, blinded_inputs_from_saved, calibrated_saves, seal_saved_decisions,
    load_saved_decisions, persist_saved_decisions,
)
from .operating_decision_prospective_phase_f_worker import run_calibrated_family
from .operating_decision_prospective_pilot import PROSPECTIVE_CAMPAIGN, PROSPECTIVE_RESERVED_PARTITION
from .operating_decision_provenance import (
    create_source_manifest, source_manifest_payload, verify_source_manifest, verify_committed_paths,
)
from .operating_decision_realism import STAGE3_TRUTH_CONDITIONS
from .operating_decision_replication import CorrectedPartition


d6 = e.d6


def phase_f_partition(name: str) -> CorrectedPartition:
    sizes = {PHASE_F_REHEARSAL_PARTITION: 1, PROSPECTIVE_RESERVED_PARTITION: 100}
    if name not in sizes:
        raise ValueError('Phase F permits only its declared disposable or reserved partition')
    return CorrectedPartition(name=name, block_count=sizes[name], campaign=PROSPECTIVE_CAMPAIGN)


def phase_f_context(repository_root: Path | str, source_revision: str) -> dict:
    root = Path(repository_root).resolve(strict=True)
    accepted = load_calibration_freeze()
    load_complete_rule()
    if any(os.environ.get(k) != v for k, v in MATH_ENVIRONMENT.items()):
        raise ValueError('executing math environment differs from the complete rule freeze')
    # The original numerical dependencies must remain byte-identical.
    if create_source_manifest(root, PHASE_E_SOURCE_PATHS).digest != accepted['source_manifest_digest']:
        raise ValueError('Phase F changed the sealed Phase E numerical source')
    manifest = create_source_manifest(root, PHASE_F_SOURCE_PATHS)
    verify_source_manifest(manifest, root, PHASE_F_SOURCE_PATHS).assert_valid()
    if verify_committed_paths(root, PHASE_F_SOURCE_PATHS).commit != source_revision:
        raise ValueError('Phase F requires its clean committed executing HEAD')
    return {
        'source_revision': source_revision, 'source_manifest': source_manifest_payload(manifest),
        'source_manifest_digest': manifest.digest,
        'protocol_digest': phase_f_protocol_digest(source_revision=source_revision, source_manifest_digest=manifest.digest),
        'protocol': phase_f_protocol_payload(source_revision=source_revision, source_manifest_digest=manifest.digest),
        'runtime_identity': d6.phase_d_runtime_identity(),
    }


def decision_context(context: Mapping, partition_name: str) -> dict:
    phase_f_partition(partition_name)
    return {**{k: deepcopy(context[k]) for k in (
        'source_revision', 'source_manifest_digest', 'protocol_digest', 'runtime_identity')},
        'campaign': PROSPECTIVE_CAMPAIGN, 'partition': partition_name,
        'complete_rule_digest': RULE_DIGEST}


def before_path(root: Path, block: int, family: str) -> Path:
    if family not in STAGE3_TRUTH_CONDITIONS or type(block) is not int or not 0 <= block < 100:
        raise ValueError('invalid before-reveal file identity')
    return root / 'before-reveal' / f'block-{block:03d}' / f'{family}.json'


def _run_block(arguments: tuple) -> dict:
    block, name, context, root_string = arguments
    partition = phase_f_partition(name)
    if type(block) is not int or not 0 <= block < partition.block_count:
        raise ValueError('invalid Phase F block index')
    config = e._physical_config(partition)
    registry = d6.RandomStreamRegistry()
    started, cpu = perf_counter(), process_time()
    truth = d6.corrected_truth_for_block(partition, block, config, registry)
    truth_timing = d6._runtime_performance_payload(perf_counter()-started, process_time()-cpu, d6._peak_rss_bytes())
    cases, saved = [], {}
    for family in STAGE3_TRUTH_CONDITIONS:
        def sink(value, family=family):
            return persist_saved_decisions(
                before_path(Path(root_string), block, family), value,
                context=decision_context(context, name), block=block, family=family)
        case, sealed = run_calibrated_family(
            truth_condition=family, block=block, truth=truth, partition=partition,
            physical_config=config, registry=registry, selector_rule=ProspectiveSelectorRule(),
            frozen_rule=d6.phase_d6_frozen_selector_rule(),
            cost_scenario=d6.PRIMARY_PROSPECTIVE_COST_SCENARIO, before_reveal_sink=sink)
        if any(f['stage'].startswith('fixed_policy_reveal:') for f in case['pipeline_failures']):
            raise RuntimeError('material incident: target reveal or physical energy failed')
        e.calibration_case_records(case)
        cases.append(case)
        saved[family] = sealed
    audit = registry.audit()
    audit.assert_clean()
    return {'numerical_block': {
        'block': block, 'cases': cases,
        'corrected_random_stream_manifest': d6._stream_manifest_payload(registry.uses),
        'corrected_random_stream_audit': d6._strict_json_value(audit),
        'timing': {'truth_generation': truth_timing, 'block_wall_seconds': perf_counter()-started,
                   'block_cpu_seconds': process_time()-cpu, 'peak_rss_bytes': d6._peak_rss_bytes()},
    }, 'before_reveal_by_family': saved}


def scored_records(result: Mapping, *, context: Mapping, partition_name: str) -> dict:
    """Authenticate the saved decision first; reveal contributes only outcome labels."""
    records = {str(c): [] for c in SECONDARY_CLEARANCES_KELVIN}
    corrections = frozen_corrections()
    for case in result['numerical_block']['cases']:
        family, block = case['truth_condition'], case['block']
        raw = {p: v['saved_before_reveal'] for p, v in case['fixed_policy_results'].items()
               if v['saved_before_reveal'] is not None}
        inputs, actions = blinded_inputs_from_saved(raw, case['selected_policy_at_n16_primary'])
        expected, _ = seal_saved_decisions(
            calibrated_saves(inputs, actions), context=decision_context(context, partition_name),
            block=block, family=family)
        if result['before_reveal_by_family'].get(family) != expected:
            raise ValueError('before-reveal decision does not match the authenticated raw saves')
        for row in e.calibration_case_records(case):
            procedure = row['procedure']
            if {k: row[k] for k in BLINDED_FIELDS} != inputs[procedure]:
                raise ValueError('post-reveal input differs from the blinded decision input')
            for i, clearance in enumerate(SECONDARY_CLEARANCES_KELVIN):
                scored = calibrated_outcome(row, corrections[procedure], decision_clearance_kelvin=clearance)
                before = expected['saves']['procedures'][procedure]['decisions'][i]
                if {k: scored[k] for k in before} != before:
                    raise ValueError('post-reveal scoring changed a saved calibrated decision')
                records[str(clearance)].append(scored)
    return records


def validate_phase_f_block(result: Mapping, *, context: Mapping, partition_name: str) -> dict:
    d6._exact_mapping(result, {'numerical_block', 'before_reveal_by_family'}, 'Phase F block')
    base = result['numerical_block']
    d6._exact_mapping(base, {'block', 'cases', 'corrected_random_stream_manifest',
                           'corrected_random_stream_audit', 'timing'}, 'Phase F numerical block')
    partition = phase_f_partition(partition_name)
    block = base['block']
    if type(block) is not int or not 0 <= block < partition.block_count:
        raise ValueError('Phase F block index is outside the frozen partition')
    if (not isinstance(base['cases'], list)
        or [c.get('truth_condition') for c in base['cases']] != list(STAGE3_TRUTH_CONDITIONS)
        or set(result['before_reveal_by_family']) != set(STAGE3_TRUTH_CONDITIONS)):
        raise ValueError('Phase F requires every family and every before-reveal save')
    e._validate_phase_e_corrected_stream_records(base['corrected_random_stream_manifest'],
        base['corrected_random_stream_audit'], block=block, partition=partition)
    for case in base['cases']:
        e._validate_phase_e_case(case, block=block, partition=partition,
            physical_config=e._physical_config(partition), selector_rule=ProspectiveSelectorRule(),
            cost_scenario=d6.PRIMARY_PROSPECTIVE_COST_SCENARIO)
        if any(f['stage'].startswith('fixed_policy_reveal:') for f in case['pipeline_failures']):
            raise ValueError('material target incident is not ordinary attrition')
    scored_records(result, context=context, partition_name=partition_name)
    return deepcopy(dict(result))


def seal_phase_f_block(result: Mapping, *, context: Mapping, partition_name: str) -> tuple:
    result = validate_phase_f_block(result, context=context, partition_name=partition_name)
    payload = {'schema_version': 1, 'protocol_version': 'phase_f_block_v1',
               'context': decision_context(context, partition_name),
               'block': result['numerical_block']['block'], 'block_result': result,
               'scientific_block_digest': d6._digest(f'{PHASE_F_DOMAIN}.scientific_block_v1',
                    d6.phase_d_scientific_block_payload(result))}
    return d6._sealed_payload(payload, domain=f'{PHASE_F_DOMAIN}.block_v1')


def load_phase_f_block(path: Path | str, *, context: Mapping, partition_name: str) -> dict:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    expected, canonical = seal_phase_f_block(value['block_result'], context=context, partition_name=partition_name)
    if raw != canonical or value != expected:
        raise ValueError('Phase F checkpoint is altered or bound to another context')
    return expected


def _write_new(path: Path, raw: bytes, validator) -> dict:
    if path.exists() or path.with_name(path.name + '.partial').exists():
        raise FileExistsError('refuse to overwrite a final or unresolved partial artifact')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with partial.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    loaded = validator(partial)
    os.replace(partial, path)
    return loaded


def _save_block(path: Path, result: Mapping, *, context: Mapping, partition_name: str) -> dict:
    _, raw = seal_phase_f_block(result, context=context, partition_name=partition_name)
    return _write_new(path, raw, lambda p: load_phase_f_block(p, context=context, partition_name=partition_name))


def validate_calibration_evidence(archive: Path | str, original_source_root: Path | str) -> dict:
    """Run the original validator in its own interpreter import context."""
    accepted = load_calibration_freeze()
    archive, source = Path(archive).resolve(strict=True), Path(original_source_root).resolve(strict=True)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != accepted['evidence']['calibration_json']['sha256']:
        raise ValueError('calibration archive is not the accepted closed result')
    revision = subprocess.run(('git', 'rev-parse', 'HEAD'), cwd=source,
                              check=True, capture_output=True, text=True).stdout.strip()
    if revision != accepted['numerical_source_revision']:
        raise ValueError('original calibration validator must execute from its frozen source revision')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    completed = subprocess.run((sys.executable, '-m', 'thermotwin.reports.operating_decision_prospective_phase_e',
        '--validate-calibration', str(archive), '--output-directory', str(archive.parent)),
        cwd=source, env=env, check=True, capture_output=True, text=True)
    expected = 'Phase E calibration archive: VALID\nfinite_calibration_requires_phase_f_freeze\n'
    if completed.stdout != expected:
        raise ValueError('original full calibration validator did not return the expected acceptance')
    return {'accepted_calibration_digest': accepted['artifact_digest'], 'full_original_validator_passed': True}


def _check_before_files(root: Path, *, context: Mapping, partition_name: str,
                        completed: Mapping) -> None:
    partition = phase_f_partition(partition_name)
    parent = root / 'before-reveal'
    if not parent.exists():
        if completed:
            raise ValueError('completed blocks lack their durable before-reveal files')
        return
    for blockdir in parent.iterdir():
        expected = {f'block-{b:03d}': b for b in range(partition.block_count)}
        if blockdir.name not in expected or not blockdir.is_dir():
            raise ValueError('unexpected before-reveal block path')
        block = expected[blockdir.name]
        families = {f'{family}.json': family for family in STAGE3_TRUTH_CONDITIONS}
        for path in blockdir.iterdir():
            if path.name not in families or not path.is_file():
                raise ValueError('unexpected or partial before-reveal file requires review')
            family = families[path.name]
            value = load_saved_decisions(path, context=decision_context(context, partition_name), block=block, family=family)
            if block in completed and value != completed[block]['block_result']['before_reveal_by_family'][family]:
                raise ValueError('durable before-reveal file differs from its completed block')
    for block in completed:
        if {p.name for p in before_path(root, block, STAGE3_TRUTH_CONDITIONS[0]).parent.iterdir()} != {
            f'{f}.json' for f in STAGE3_TRUTH_CONDITIONS}:
            raise ValueError('completed block has an incomplete before-reveal inventory')


def phase_f_preflight(*, repository_root, source_revision, output_directory, partition_name,
                      calibration_archive, calibration_source_root) -> dict:
    partition = phase_f_partition(partition_name)
    d6.validate_executing_phase_d_runtime()
    context = phase_f_context(repository_root, source_revision)
    e._check_phase_e_corrected_stream_contract(partition)
    validation = validate_calibration_evidence(calibration_archive, calibration_source_root)
    root = Path(output_directory).resolve()
    if any((root / n).exists() for n in ('evaluation.json', 'evaluation.json.partial', 'rehearsal.json', 'incident.json')):
        raise FileExistsError('final output or unresolved incident already exists')
    expected = {f'block-{b:03d}.json': b for b in range(partition.block_count)}
    completed = {}
    if (root / 'blocks').exists():
        for path in (root / 'blocks').iterdir():
            if path.name not in expected:
                raise ValueError('unexpected or partial checkpoint requires incident review')
            block = expected[path.name]
            value = load_phase_f_block(path, context=context, partition_name=partition_name)
            if value['block'] != block:
                raise ValueError('checkpoint filename differs from its block identity')
            completed[block] = {'block_result': {'before_reveal_by_family':
                                                value['block_result']['before_reveal_by_family']}}
    _check_before_files(root, context=context, partition_name=partition_name, completed=completed)
    return {'context': context, 'validated_existing_blocks': sorted(completed),
            'partition': partition_name, 'block_count': partition.block_count,
            'calibration_evidence': validation, 'reserved_generation_authorized': False}


def validate_generation_gate(gate: Mapping, *, context: Mapping) -> dict:
    required = {'protocol_tests', 'exact_source_ci', 'full_original_calibration_validation',
                'disposable_roundtrip', 'exact_scientific_replay', 'constructed_100_block_roundtrip',
                'independent_saved_evidence_reload', 'process_tree_monitor_preflight'}
    expected = {'protocol_version': 'phase_f_generation_gate_v1',
                **decision_context(context, PROSPECTIVE_RESERVED_PARTITION),
                'reserved_generation_authorized': True}
    if (not isinstance(gate, Mapping) or any(gate.get(k) != v for k, v in expected.items())
        or not isinstance(gate.get('checks'), Mapping) or set(gate['checks']) != required
        or any(v is not True for v in gate['checks'].values())):
        raise ValueError('Phase F generation gate is incomplete or bound to another source')
    ci = gate.get('ci', {})
    if (ci.get('head_sha') != context['source_revision'] or ci.get('status') != 'completed'
        or ci.get('conclusion') != 'success'):
        raise ValueError('Phase F requires passing exact-source CI')
    resources = gate.get('resources', {})
    peak, samples = resources.get('process_tree_peak_rss_bytes'), resources.get('process_tree_sample_count')
    if (resources.get('source_revision') != context['source_revision']
        or resources.get('partition') != PHASE_F_REHEARSAL_PARTITION
        or resources.get('execution_completed') is not True
        or resources.get('resource_gate_passed') is not True
        or resources.get('process_tree_sampling_error') is not None
        or type(peak) is not int or not 0 < peak <= 4 * 1024**3
        or type(samples) is not int or samples <= 0):
        raise ValueError('Phase F gate lacks passing whole-rehearsal resource evidence')
    evidence = gate.get('rehearsal_evidence', {})
    for field in ('rehearsal_json_sha256', 'independent_validation_json_sha256',
                  'rehearsal_resources_sha256', 'constructed_archive_sha256',
                  'scientific_result_digest'):
        d6._validate_sha256(field, evidence.get(field))
    return deepcopy(dict(gate))


def load_committed_generation_gate(path, *, gate_repository_root, context):
    root, path = Path(gate_repository_root).resolve(strict=True), Path(path).resolve(strict=True)
    relative = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    committed = subprocess.run(('git', 'show', f'HEAD:{relative}'), cwd=root,
                               check=True, capture_output=True).stdout
    if raw != committed:
        raise ValueError('Phase F gate must match its committed bytes')
    subprocess.run(('git', 'merge-base', '--is-ancestor', context['source_revision'], 'HEAD'),
                   cwd=root, check=True, capture_output=True)
    return validate_generation_gate(json.loads(raw), context=context)


def _analysis_payload(context: Mapping, inventory: list, records: Mapping, *, partition_name: str) -> dict:
    count = phase_f_partition(partition_name).block_count
    return {'schema_version': 1, 'protocol_version': 'phase_f_evaluation_v1',
            'context': deepcopy(dict(context)), 'partition': partition_name,
            'block_file_records': inventory, 'calibrated_case_records_by_clearance': records,
            'primary_analysis': analyze_primary_comparison(records['0.0'], block_count=count),
            'secondary_analysis_by_clearance': {
                key: {'scientific_use': 'descriptive_only_no_confirmatory_inference',
                      'analysis': analyze_primary_comparison(value, block_count=count)}
                for key, value in records.items() if key != '0.0'},
            'reserved_sample_size_extended_after_outcomes': False}


def validate_phase_f_archive(path, *, repository_root) -> dict:
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    context = phase_f_context(repository_root, value['context']['source_revision'])
    name = value['partition']
    partition = phase_f_partition(name)
    inventory, records, completed = [], {str(c): [] for c in SECONDARY_CLEARANCES_KELVIN}, {}
    expected_files = {f'block-{b:03d}.json' for b in range(partition.block_count)}
    if {p.name for p in (path.parent / 'blocks').iterdir()} != expected_files:
        raise ValueError('evaluation has an incomplete or unexpected checkpoint inventory')
    for b in range(partition.block_count):
        blockpath = path.parent / 'blocks' / f'block-{b:03d}.json'
        block = load_phase_f_block(blockpath, context=context, partition_name=name)
        if block['block'] != b:
            raise ValueError('evaluation checkpoint block identity differs from its filename')
        completed[b] = {'block_result': {'before_reveal_by_family':
                                        block['block_result']['before_reveal_by_family']}}
        inventory.append({'block': b, 'path': f'blocks/{blockpath.name}',
                          'sha256': hashlib.sha256(blockpath.read_bytes()).hexdigest(),
                          'scientific_block_digest': block['scientific_block_digest']})
        rows = scored_records(block['block_result'], context=context, partition_name=name)
        for key in records:
            records[key].extend(rows[key])
    _check_before_files(path.parent, context=context, partition_name=name, completed=completed)
    expected = _analysis_payload(context, inventory, records, partition_name=name)
    if name == PROSPECTIVE_RESERVED_PARTITION:
        expected['generation_gate'] = validate_generation_gate(value.get('generation_gate'), context=context)
    expected['performance'] = value.get('performance')
    expected['scientific_result_digest'] = d6._digest(f'{PHASE_F_DOMAIN}.evaluation_scientific_v1', {
        'context': context, 'partition': name, 'scientific_block_digests': [r['scientific_block_digest'] for r in inventory],
        'calibrated_case_records_by_clearance': records, 'primary_analysis': expected['primary_analysis'],
        'secondary_analysis_by_clearance': expected['secondary_analysis_by_clearance']})
    sealed, canonical = d6._sealed_payload(expected, domain=f'{PHASE_F_DOMAIN}.evaluation_v1')
    if value != sealed or raw != canonical:
        raise ValueError('evaluation summary does not reproduce from authenticated raw checkpoints')
    return sealed


def validate_ci_evidence(ci: Mapping, *, source_revision: str) -> dict:
    if (not isinstance(ci, Mapping) or ci.get('head_sha') != source_revision
        or ci.get('status') != 'completed' or ci.get('conclusion') != 'success'
        or ci.get('workflow_id') != 348759149
        or ci.get('repository', {}).get('full_name') != 'rolandrolandroland/thermotwin'):
        raise ValueError('rehearsal requires completed successful CI for the exact source revision')
    return deepcopy(dict(ci))


def _record_incident(root: Path, *, context: Mapping, name: str, error: Exception, completed: list) -> None:
    record = {'schema_version': 1, 'context': decision_context(context, name),
              'error_type': type(error).__name__, 'message': str(error),
              'scientific_interpretation_stopped': True, 'completed_blocks_retained': sorted(completed)}
    with (root / 'incident.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _scan_checkpoints(root: Path, *, context: Mapping, name: str) -> dict:
    partition = phase_f_partition(name)
    if (root / 'incident.json').exists() or (root / 'evaluation.json.partial').exists():
        raise FileExistsError('unresolved incident or partial evaluation requires review')
    expected = {f'block-{b:03d}.json': b for b in range(partition.block_count)}
    completed = {}
    if (root / 'blocks').exists():
        for path in (root / 'blocks').iterdir():
            if path.name not in expected:
                raise ValueError('unexpected or partial checkpoint requires review')
            b = expected[path.name]
            value = load_phase_f_block(path, context=context, partition_name=name)
            if value['block'] != b:
                raise ValueError('checkpoint filename does not match its block')
            completed[b] = {'block_result': {'before_reveal_by_family':
                                             value['block_result']['before_reveal_by_family']}}
    _check_before_files(root, context=context, partition_name=name, completed=completed)
    return completed


def _execute_partition(*, root: Path, repository_root, context: Mapping, name: str,
                       gate: Mapping | None = None, progress=None) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'evaluation.json').exists():
        # Only the explicitly declared disposable pair can resume a completed
        # first computation. Reserved final archives are never regenerated.
        if name != PHASE_F_REHEARSAL_PARTITION:
            raise FileExistsError('reserved evaluation is complete and closed')
        return validate_phase_f_archive(root / 'evaluation.json', repository_root=repository_root)
    completed = _scan_checkpoints(root, context=context, name=name)
    count = phase_f_partition(name).block_count
    inventory, rows, block_cpu = {}, {}, {}
    started, cpu = perf_counter(), process_time()
    resumed = len(completed)

    def retain(b, value):
        path = root / 'blocks' / f'block-{b:03d}.json'
        inventory[b] = {'block': b, 'path': f'blocks/{path.name}',
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'scientific_block_digest': value['scientific_block_digest']}
        rows[b] = scored_records(value['block_result'], context=context, partition_name=name)
        block_cpu[b] = value['block_result']['numerical_block']['timing']['block_cpu_seconds']

    for b in completed:
        retain(b, load_phase_f_block(root / 'blocks' / f'block-{b:03d}.json', context=context, partition_name=name))
    del completed
    missing = [b for b in range(count) if b not in inventory]
    if progress:
        progress(f'{name}: {len(inventory)}/{count} authenticated blocks; {len(missing)} pending')
    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_run_block, (b, name, context, str(root))): b for b in missing}
            for future in as_completed(futures):
                b = futures.pop(future)
                try:
                    result = future.result()
                    checkpoint = _save_block(root / 'blocks' / f'block-{b:03d}.json', result,
                                             context=context, partition_name=name)
                    retain(b, checkpoint)
                    del result, checkpoint
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise
                if progress:
                    progress(f'{name}: validated block {len(inventory)}/{count}')
        all_rows = {str(c): [row for b in range(count) for row in rows[b][str(c)]]
                    for c in SECONDARY_CLEARANCES_KELVIN}
        ordered = [inventory[b] for b in range(count)]
        value = _analysis_payload(context, ordered, all_rows, partition_name=name)
        if name == PROSPECTIVE_RESERVED_PARTITION:
            value['generation_gate'] = validate_generation_gate(gate, context=context)
        value['performance'] = {'wall_seconds_this_invocation': perf_counter()-started,
                                'coordinator_cpu_seconds_this_invocation': process_time()-cpu,
                                'aggregate_completed_block_cpu_seconds': sum(block_cpu.values()),
                                'resumed_blocks': resumed}
        value['scientific_result_digest'] = d6._digest(f'{PHASE_F_DOMAIN}.evaluation_scientific_v1', {
            'context': context, 'partition': name,
            'scientific_block_digests': [r['scientific_block_digest'] for r in ordered],
            'calibrated_case_records_by_clearance': all_rows, 'primary_analysis': value['primary_analysis'],
            'secondary_analysis_by_clearance': value['secondary_analysis_by_clearance']})
        _, raw = d6._sealed_payload(value, domain=f'{PHASE_F_DOMAIN}.evaluation_v1')
        return _write_new(root / 'evaluation.json', raw,
                          lambda p: validate_phase_f_archive(p, repository_root=repository_root))
    except Exception as error:
        _record_incident(root, context=context, name=name, error=error, completed=list(inventory))
        raise


def constructed_archive_probe(seed: Mapping, *, output_directory) -> dict:
    """Repeated records test storage/analysis transport, never scientific replication."""
    root = Path(output_directory)
    path = root / 'constructed-100-block-archive.json'
    raw = d6._canonical_bytes(seed['block_result'])
    digest = hashlib.sha256(raw).hexdigest()
    with path.open('xb') as stream:
        stream.write(b'{"scientific_use":"constructed_resource_evidence_only","blocks":[')
        for i in range(100):
            if i:
                stream.write(b',')
            stream.write(raw)
        stream.write(b']}\n')
        stream.flush()
        os.fsync(stream.fileno())
    restored = json.loads(path.read_bytes())
    if restored['scientific_use'] != 'constructed_resource_evidence_only' or len(restored['blocks']) != 100:
        raise ValueError('constructed archive marker or count is invalid')
    if any(hashlib.sha256(d6._canonical_bytes(v)).hexdigest() != digest for v in restored['blocks']):
        raise ValueError('constructed raw archive failed byte roundtrip')
    del restored
    rows = []
    for b in range(100):
        for row in seed['calibrated_records']:
            rows.append({**deepcopy(row), 'block': b,
                         'device_token': f'constructed-only-{b}-{row["truth_family"]}'})
    analysis = analyze_primary_comparison(rows, block_count=100)
    if analyze_primary_comparison(json.loads(d6._canonical_bytes(rows)), block_count=100) != analysis:
        raise ValueError('constructed 1500-cell analysis failed roundtrip')
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            hasher.update(chunk)
    return {'scientific_use': 'constructed_resource_evidence_only', 'constructed_blocks': 100,
            'constructed_analysis_records': len(rows), 'roundtrip_passed': True,
            'bytes': path.stat().st_size, 'sha256': hasher.hexdigest(), 'seed_block_sha256': digest}


def run_phase_f_rehearsal(*, repository_root, source_revision, output_directory,
                          calibration_archive, calibration_source_root, ci_evidence, progress=None) -> dict:
    ci = validate_ci_evidence(ci_evidence, source_revision=source_revision)
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    e._claim_execution_lock(root / 'rehearsal.execution.lock', source_revision=source_revision)
    try:
        preflight = phase_f_preflight(repository_root=repository_root, source_revision=source_revision,
            output_directory=root, partition_name=PHASE_F_REHEARSAL_PARTITION,
            calibration_archive=calibration_archive, calibration_source_root=calibration_source_root)
        context = preflight['context']
        completed = []
        for index in (1, 2):
            completed.append(_execute_partition(root=root / f'run-{index}', repository_root=repository_root,
                context=context, name=PHASE_F_REHEARSAL_PARTITION, progress=progress))
            if progress:
                progress(f'Phase F independent disposable computation {index}/2 saved and validated')
        if completed[0]['scientific_result_digest'] != completed[1]['scientific_result_digest']:
            raise ValueError('Phase F exact scientific replay differs')
        first = load_phase_f_block(root / 'run-1' / 'blocks' / 'block-000.json',
                                   context=context, partition_name=PHASE_F_REHEARSAL_PARTITION)
        probe = constructed_archive_probe({'block_result': first['block_result'],
            'calibrated_records': completed[0]['calibrated_case_records_by_clearance']['0.0']}, output_directory=root)
        record = {'schema_version': 1, 'protocol_version': 'phase_f_rehearsal_v1', 'context': context,
                  'ci_evidence': ci, 'calibration_evidence': preflight['calibration_evidence'],
                  'partition': PHASE_F_REHEARSAL_PARTITION, 'scientific_use': 'disposable_engineering_only',
                  'block_roundtrip_passed': True, 'exact_scientific_replay_passed': True,
                  'scientific_result_digest': completed[0]['scientific_result_digest'],
                  'constructed_archive_probe': probe, 'reserved_partition_opened': False,
                  'calibration_partition_regenerated': False}
        with (root / 'rehearsal.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
        return record
    except Exception as error:
        if not (root / 'incident.json').exists():
            with (root / 'incident.json').open('x', encoding='utf-8') as stream:
                json.dump({'error_type': type(error).__name__, 'message': str(error),
                           'scientific_interpretation_stopped': True}, stream, sort_keys=True, indent=2)
                stream.write('\n')
        raise
    finally:
        (root / 'rehearsal.execution.lock').unlink(missing_ok=True)


def run_phase_f_reserved(*, repository_root, source_revision, output_directory,
                         calibration_archive, calibration_source_root,
                         generation_gate_path, gate_repository_root, progress=None) -> dict:
    # Authenticate the committed gate before expensive archive validation, and
    # before creating any reserved device, observation or output checkpoint.
    context = phase_f_context(repository_root, source_revision)
    gate = load_committed_generation_gate(generation_gate_path, gate_repository_root=gate_repository_root, context=context)
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    e._claim_execution_lock(root / 'execution.lock', source_revision=source_revision)
    try:
        preflight = phase_f_preflight(repository_root=repository_root, source_revision=source_revision,
            output_directory=root, partition_name=PROSPECTIVE_RESERVED_PARTITION,
            calibration_archive=calibration_archive, calibration_source_root=calibration_source_root)
        if preflight['context'] != context:
            raise ValueError('source changed during reserved preflight')
        return _execute_partition(root=root, repository_root=repository_root, context=context,
                                  name=PROSPECTIVE_RESERVED_PARTITION, gate=gate, progress=progress)
    finally:
        (root / 'execution.lock').unlink(missing_ok=True)
