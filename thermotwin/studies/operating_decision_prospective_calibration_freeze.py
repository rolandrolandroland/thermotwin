"""Authenticate completed calibration without changing the frozen selector."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .operating_decision_prospective_phase_d_final_freeze import load_prospective_phase_d_final_freeze
from .operating_decision_prospective_phase_e import phase_e_protocol_digest
from .operating_decision_prospective_phase_e_calibration import PHASE_E_PROCEDURES


CALIBRATION_FREEZE_DIGEST = '124d97f19c1cecb8cfefa6ee5a2b78f4f78570af8337b649f04bbc61435f4f2d'
CALIBRATION_FREEZE_PATH = Path(__file__).resolve().parents[1] / 'OPERATING_DECISION_PROSPECTIVE_PHASE_E_CALIBRATION.json'


def validate_calibration_freeze(payload: Mapping[str, object]) -> dict:
    if not isinstance(payload, Mapping):
        raise ValueError('calibration freeze must be a mapping')
    record = deepcopy(dict(payload))
    digest = record.pop('artifact_digest', None)
    encoded = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if digest != CALIBRATION_FREEZE_DIGEST or hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError('calibration freeze does not match the accepted immutable record')
    predecessor = load_prospective_phase_d_final_freeze()
    if (record['phase_d_final_freeze_artifact_digest'] != predecessor['artifact_digest']
        or record['protocol_digest'] != phase_e_protocol_digest(
            source_revision=record['numerical_source_revision'],
            source_manifest_digest=record['source_manifest_digest'])
        or set(record['calibration']['procedures']) != set(PHASE_E_PROCEDURES)
        or record['calibration']['all_required_corrections_finite'] is not True
        or record['calibration']['order_statistic_rank'] != 96
        or record['calibration']['block_count'] != 100
        or record['calibration']['confidence_is_joint_over_procedures'] is not False
        or record['authorization']['reserved_generation_authorized'] is not False):
        raise ValueError('calibration freeze lineage or semantics are invalid')
    record['artifact_digest'] = digest
    return record


def load_calibration_freeze(path: Path | str = CALIBRATION_FREEZE_PATH) -> dict:
    return validate_calibration_freeze(json.loads(Path(path).read_bytes()))


def frozen_corrections() -> dict:
    return {name: record['correction_kelvin'] for name, record in
            load_calibration_freeze()['calibration']['procedures'].items()}
