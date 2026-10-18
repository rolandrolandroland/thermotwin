"""Complete calibrated procedure, source identity and reserved analysis freeze."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .operating_decision_prospective_calibration_freeze import (
    CALIBRATION_FREEZE_DIGEST, load_calibration_freeze, frozen_corrections,
)
from .operating_decision_prospective_phase_e import PHASE_E_SOURCE_PATHS, phase_e_protocol_payload
from .operating_decision_prospective_pilot import _digest, _validate_revision, _validate_sha256


PHASE_F_VERSION = 'operating_decision_prospective_phase_f_v1'
PHASE_F_DOMAIN = 'thermotwin.prospective_phase_f'
PHASE_F_REHEARSAL_PARTITION = 'p0_disposable_phase_f_roundtrip_v1'
RULE_PATH = Path(__file__).resolve().parents[1] / 'OPERATING_DECISION_PROSPECTIVE_PHASE_F_RULE.json'
# Fixed after constructing the complete rule from the accepted calibration.
RULE_DIGEST = 'd8ae3d9ea3748d4de776a672cf19798b145b589d681211ba88b8789186dd6341'
MATH_ENVIRONMENT = dict.fromkeys((
    'BLIS_NUM_THREADS', 'MKL_CBWR', 'MKL_DYNAMIC', 'MKL_NUM_THREADS',
    'NPY_DISABLE_CPU_FEATURES', 'NUMEXPR_MAX_THREADS', 'NUMEXPR_NUM_THREADS',
    'OMP_DYNAMIC', 'OMP_NUM_THREADS', 'OMP_PLACES', 'OMP_PROC_BIND', 'OMP_SCHEDULE',
    'OMP_THREAD_LIMIT', 'OPENBLAS_CORETYPE', 'OPENBLAS_NUM_THREADS', 'PYTHONHASHSEED',
    'PYTHONOPTIMIZE', 'VECLIB_MAXIMUM_THREADS',
))
PHASE_F_SOURCE_PATHS = tuple(sorted(set(PHASE_E_SOURCE_PATHS).union({
    'thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_E_CALIBRATION.json',
    'thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_RULE.json',
    'thermotwin/studies/operating_decision_prospective_calibration_freeze.py',
    'thermotwin/studies/operating_decision_prospective_phase_f.py',
    'thermotwin/studies/operating_decision_prospective_phase_f_decisions.py',
    'thermotwin/studies/operating_decision_prospective_phase_f_worker.py',
    'thermotwin/studies/operating_decision_prospective_phase_f_execution.py',
    'thermotwin/reports/operating_decision_prospective_phase_f.py',
})))


def complete_rule_payload() -> dict:
    accepted = load_calibration_freeze()
    parent = phase_e_protocol_payload(
        source_revision=accepted['numerical_source_revision'],
        source_manifest_digest=accepted['source_manifest_digest'],
    )
    return {
        'schema_version': 1, 'protocol_version': PHASE_F_VERSION,
        'calibration_artifact_digest': CALIBRATION_FREEZE_DIGEST,
        'calibration_scientific_result_digest': accepted['evidence']['scientific_result_digest'],
        'phase_e_protocol_digest': accepted['protocol_digest'],
        'original_numerical_source_revision': accepted['numerical_source_revision'],
        'original_numerical_source_manifest_digest': accepted['source_manifest_digest'],
        'math_environment': deepcopy(MATH_ENVIRONMENT),
        'design': {k: deepcopy(parent[k]) for k in (
            'campaign', 'calibration_partition', 'calibration_block_count',
            'reserved_partition', 'reserved_block_count', 'phase_d_final_freeze_artifact_digest',
            'selector_rule', 'draw_count', 'max_unstable_draws_per_source_action',
            'worker_count', 'sensor_scenario', 'cost_scenario', 'runtime_identity',
            'procedures', 'calibration', 'analysis', 'failure_policy',
            'paired_precision_artifact_digest', 'precision_and_compute',
        )},
        'final_corrections_kelvin': frozen_corrections(),
        'calibrated_decision': {
            'input_fields': ['raw_interval', 'development_offset_kelvin',
                             'verification_succeeded', 'pipeline_failure'],
            'interval': '[L - development_offset(action) - q(procedure), U + development_offset(action) + q(procedure)]',
            'arithmetic': 'exact binary fractions with outward endpoint rounding',
            'approve': 'verified finite lower >= clearance',
            'reject': 'verified finite upper < -clearance',
            'otherwise': 'insufficient_evidence',
            'primary_clearance_kelvin': 0.,
            'all_four_clearance_decisions_durably_saved_before_any_target_reveal': True,
            'selector_correction_is_procedure_specific_not_selected_action_specific': True,
            'corrections_change_selection_stop_fits_offsets_or_verification': False,
            'existing_before_reveal_files_on_resume_must_match_exactly': True,
        },
        'disposable_rehearsal': {
            'partition': PHASE_F_REHEARSAL_PARTITION, 'block_count': 1,
            'independent_computations': 2, 'scientific_use': 'engineering_only',
            'same_semantic_keys_only_for_explicit_exact_replay': True,
            'calibration_and_reserved_observations_generated': False,
        },
        'reserved_generation_gate': {
            'committed_complete_rule_and_source': True,
            'passing_exact_source_ci_required': True,
            'full_original_calibration_revalidation_required': True,
            'disposable_roundtrip_and_exact_scientific_replay_required': True,
            'constructed_100_block_archive_roundtrip_required': True,
            'independent_saved_evidence_reload_required': True,
            'process_tree_monitor_preflight_required': True,
            'maximum_process_tree_bytes': 4 * 1024**3,
            'separate_committed_generation_gate_required': True,
            'reserved_generation_authorized_by_this_rule_artifact': False,
        },
    }


def load_complete_rule(path: Path | str = RULE_PATH) -> dict:
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, Mapping):
        raise ValueError('complete calibrated rule must be a mapping')
    material = dict(value)
    digest = material.pop('artifact_digest', None)
    computed = hashlib.sha256(json.dumps(
        material, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode()).hexdigest()
    if digest != RULE_DIGEST or digest != computed or material != complete_rule_payload():
        raise ValueError('complete calibrated rule differs from the sealed procedure')
    return deepcopy(dict(value))


def phase_f_protocol_payload(*, source_revision: str, source_manifest_digest: str) -> dict:
    _validate_revision(source_revision)
    _validate_sha256('Phase F source manifest', source_manifest_digest)
    return {'protocol_version': PHASE_F_VERSION, 'source_revision': source_revision,
            'source_manifest_digest': source_manifest_digest,
            'complete_calibrated_rule': load_complete_rule()}


def phase_f_protocol_digest(*, source_revision: str, source_manifest_digest: str) -> str:
    return _digest(f'{PHASE_F_DOMAIN}.protocol_v1', phase_f_protocol_payload(
        source_revision=source_revision, source_manifest_digest=source_manifest_digest))
