"""Persist calibrated decisions before invoking any target-reveal function."""

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Mapping

from .operating_decision import POLICY_NAMES
from .operating_decision_prospective_calibration_freeze import (
    CALIBRATION_FREEZE_DIGEST, frozen_corrections,
)
from .operating_decision_prospective_phase_d6_internal_check import (
    _sealed_payload, phase_d6_frozen_selector_rule,
)
from .operating_decision_prospective_phase_e_calibration import (
    PHASE_E_PROCEDURES, SECONDARY_CLEARANCES_KELVIN, SELECTOR_PROCEDURE,
    apply_calibrated_decision,
)
from .operating_decision_realism import STAGE3_TRUTH_CONDITIONS


DECISION_VERSION = 'prospective_phase_f_saved_calibrated_decisions_v1'
DECISION_DOMAIN = 'thermotwin.prospective_phase_f.before_reveal_v1'
BLINDED_FIELDS = {'raw_interval', 'development_offset_kelvin',
                  'verification_succeeded', 'pipeline_failure'}


def blinded_inputs_from_saved(saved_by_policy: Mapping, selected_policy: str | None) -> tuple:
    """Extract exactly the four allowed final-decision inputs from raw saves."""
    if not isinstance(saved_by_policy, Mapping) or set(saved_by_policy) - set(POLICY_NAMES):
        raise ValueError('unexpected fixed policy in blinded saves')
    if selected_policy is not None and selected_policy not in POLICY_NAMES:
        raise ValueError('unknown selected policy')
    offsets = phase_d6_frozen_selector_rule().development_offsets
    inputs, actions = {}, {}
    for procedure in PHASE_E_PROCEDURES:
        action = selected_policy if procedure == SELECTOR_PROCEDURE else procedure
        raw = saved_by_policy.get(action)
        verified = bool(isinstance(raw, Mapping) and any(
            v.get('passed') is True for v in raw.get('verifications', [])
            if isinstance(v, Mapping)))
        inputs[procedure] = {
            'raw_interval': deepcopy(raw.get('margin_envelope')) if isinstance(raw, Mapping) else None,
            'development_offset_kelvin': offsets.for_policy(action) if action is not None else 0.,
            'verification_succeeded': verified,
            'pipeline_failure': action is None or not isinstance(raw, Mapping),
        }
        actions[procedure] = action
    return inputs, actions


def calibrated_saves(inputs: Mapping, actions: Mapping) -> dict:
    if (not isinstance(inputs, Mapping) or not isinstance(actions, Mapping)
        or set(inputs) != set(PHASE_E_PROCEDURES) or set(actions) != set(PHASE_E_PROCEDURES)):
        raise ValueError('retain the selector and all four fixed procedures')
    corrections = frozen_corrections()
    procedures = {}
    for procedure in PHASE_E_PROCEDURES:
        if not isinstance(inputs[procedure], Mapping) or set(inputs[procedure]) != BLINDED_FIELDS:
            raise ValueError('final decision accepts only the four blinded fields')
        action = actions[procedure]
        if (procedure != SELECTOR_PROCEDURE and action != procedure
            or procedure == SELECTOR_PROCEDURE and action not in (None, *POLICY_NAMES)):
            raise ValueError('saved selected action is invalid')
        procedures[procedure] = {
            'input': deepcopy(dict(inputs[procedure])), 'selected_action': action,
            'decisions': [apply_calibrated_decision(
                inputs[procedure], corrections[procedure], decision_clearance_kelvin=clearance,
            ) for clearance in SECONDARY_CLEARANCES_KELVIN],
        }
    return {'calibration_artifact_digest': CALIBRATION_FREEZE_DIGEST,
            'clearances_kelvin': list(SECONDARY_CLEARANCES_KELVIN), 'procedures': procedures}


def seal_saved_decisions(saves: Mapping, *, context: Mapping, block: int, family: str) -> tuple:
    if type(block) is not int or block < 0 or family not in STAGE3_TRUTH_CONDITIONS:
        raise ValueError('invalid before-reveal case identity')
    if not isinstance(saves, Mapping) or set(saves) != {
        'calibration_artifact_digest', 'clearances_kelvin', 'procedures',
    }:
        raise ValueError('saved decision payload contains undeclared fields')
    procedures = saves['procedures']
    if (not isinstance(procedures, Mapping) or set(procedures) != set(PHASE_E_PROCEDURES)
        or any(not isinstance(v, Mapping) or set(v) != {'input', 'selected_action', 'decisions'}
               for v in procedures.values())):
        raise ValueError('saved decisions require exactly the five procedure records')
    inputs = {p: procedures[p]['input'] for p in PHASE_E_PROCEDURES}
    actions = {p: procedures[p]['selected_action'] for p in PHASE_E_PROCEDURES}
    if saves != calibrated_saves(inputs, actions):
        raise ValueError('calibrated decisions do not reproduce from blinded inputs')
    payload = {'schema_version': 1, 'protocol_version': DECISION_VERSION,
               'context': deepcopy(dict(context)), 'block': block, 'truth_condition': family,
               'phase': 'before_target_reveal', 'saves': deepcopy(dict(saves))}
    return _sealed_payload(payload, domain=DECISION_DOMAIN)


def load_saved_decisions(path: Path | str, *, context: Mapping, block: int, family: str) -> dict:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping) or 'saves' not in value:
        raise ValueError('missing blinded decision saves')
    expected, canonical = seal_saved_decisions(value['saves'], context=context, block=block, family=family)
    if raw != canonical or value != expected:
        raise ValueError('saved calibrated decisions are altered or bound to another context')
    return expected


def persist_saved_decisions(path: Path | str, saves: Mapping, *, context: Mapping,
                            block: int, family: str) -> dict:
    """Existing partial-block saves may only be reused if byte-identical."""
    path = Path(path)
    payload, raw = seal_saved_decisions(saves, context=context, block=block, family=family)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('resumed blinded decisions differ; preserve incident and stop')
        return load_saved_decisions(path, context=context, block=block, family=family)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    if partial.exists():
        raise FileExistsError('unresolved partial calibrated-decision save requires review')
    with partial.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    load_saved_decisions(partial, context=context, block=block, family=family)
    os.replace(partial, path)
    return payload
