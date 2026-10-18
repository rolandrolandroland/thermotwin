from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from thermotwin.studies.operating_decision import POLICY_NAMES, INSUFFICIENT_EVIDENCE
from thermotwin.studies.operating_decision_prospective_calibration_freeze import (
    load_calibration_freeze, validate_calibration_freeze, frozen_corrections,
)
from thermotwin.studies.operating_decision_prospective_phase_e_calibration import (
    PHASE_E_PROCEDURES, SELECTOR_PROCEDURE,
)
from thermotwin.studies.operating_decision_prospective_phase_f_decisions import (
    blinded_inputs_from_saved, calibrated_saves, persist_saved_decisions,
    load_saved_decisions, seal_saved_decisions,
)
from thermotwin.studies.operating_decision_realism import STAGE3_TRUTH_CONDITIONS


def saves_fixture():
    raw = {p: {'margin_envelope': {'lower': .2, 'upper': .4},
               'verifications': [{'passed': True}]} for p in POLICY_NAMES}
    inputs, actions = blinded_inputs_from_saved(raw, 'fixed_voltage')
    return calibrated_saves(inputs, actions)


class CalibrationFreezeTests(unittest.TestCase):
    def test_accepted_corrections_and_feasibility(self):
        record = load_calibration_freeze()
        self.assertEqual(record['calibration']['order_statistic_rank'], 96)
        self.assertEqual(record['calibration']['block_count'], 100)
        self.assertFalse(record['authorization']['reserved_generation_authorized'])
        self.assertEqual(frozen_corrections(), {
            SELECTOR_PROCEDURE: .025436425099518184,
            'stop_now': .04965713743107725, 'fixed_thermal': .010413810957399489,
            'fixed_voltage': .04106348498740038,
            'fixed_face_temperature': .025436425099518184,
        })

    def test_even_coherently_rehashed_calibration_change_is_rejected(self):
        record = load_calibration_freeze()
        record['calibration']['procedures'][SELECTOR_PROCEDURE]['correction_kelvin'] = 0.
        with self.assertRaises(ValueError):
            validate_calibration_freeze(record)
        record.pop('artifact_digest')
        record['artifact_digest'] = hashlib.sha256(json.dumps(
            record, sort_keys=True, separators=(',', ':'), allow_nan=False,
        ).encode()).hexdigest()
        with self.assertRaises(ValueError):
            validate_calibration_freeze(record)


class BeforeRevealDecisionTests(unittest.TestCase):
    def test_selector_has_own_q_and_selected_action_offset(self):
        saved = saves_fixture()['procedures']
        self.assertEqual(saved[SELECTOR_PROCEDURE]['selected_action'], 'fixed_voltage')
        self.assertEqual(saved[SELECTOR_PROCEDURE]['input']['development_offset_kelvin'], .074)
        self.assertEqual(saved[SELECTOR_PROCEDURE]['decisions'][0]['correction_kelvin'],
                         frozen_corrections()[SELECTOR_PROCEDURE])
        self.assertNotEqual(saved[SELECTOR_PROCEDURE]['decisions'][0]['final_interval'],
                            saved['fixed_voltage']['decisions'][0]['final_interval'])
        self.assertEqual([d['decision_clearance_kelvin'] for d in saved[SELECTOR_PROCEDURE]['decisions']],
                         [0., .05, .1, .2])

    def test_truth_or_family_cannot_enter_decision_api(self):
        saves = saves_fixture()
        inputs = {p: saves['procedures'][p]['input'] for p in PHASE_E_PROCEDURES}
        actions = {p: saves['procedures'][p]['selected_action'] for p in PHASE_E_PROCEDURES}
        for extra in ('true_margin', 'truth_family', 'final_response', 'device_token'):
            bad = deepcopy(inputs)
            bad[SELECTOR_PROCEDURE][extra] = 0.
            with self.assertRaisesRegex(ValueError, 'four blinded'):
                calibrated_saves(bad, actions)

    def test_failed_fixed_procedure_does_not_remove_other_decisions(self):
        raw = {p: {'margin_envelope': {'lower': .2, 'upper': .4},
                   'verifications': [{'passed': True}], 'excluded_candidates': ['five_state']}
               for p in POLICY_NAMES if p != 'fixed_voltage'}
        inputs, actions = blinded_inputs_from_saved(raw, None)
        saved = calibrated_saves(inputs, actions)['procedures']
        self.assertEqual(set(saved), set(PHASE_E_PROCEDURES))
        for p in (SELECTOR_PROCEDURE, 'fixed_voltage'):
            self.assertTrue(all(d['decision'] == INSUFFICIENT_EVIDENCE for d in saved[p]['decisions']))
        self.assertIsNotNone(saved['fixed_thermal']['decisions'][0]['final_interval'])

    def test_durable_save_reloads_identically_and_refuses_changed_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'before.json'
            kwargs = dict(context={'source_revision': 'a'*40}, block=0, family=STAGE3_TRUTH_CONDITIONS[0])
            saved = saves_fixture()
            first = persist_saved_decisions(path, saved, **kwargs)
            self.assertEqual(first, load_saved_decisions(path, **kwargs))
            self.assertEqual(first, persist_saved_decisions(path, saved, **kwargs))
            raw = path.read_bytes()
            changed = deepcopy(saved)
            changed['procedures'][SELECTOR_PROCEDURE]['decisions'][0]['decision'] = 'reject'
            with self.assertRaises(ValueError):
                persist_saved_decisions(path, changed, **kwargs)
            self.assertEqual(path.read_bytes(), raw)
            with self.assertRaises(ValueError):
                load_saved_decisions(path, **{**kwargs, 'block': 1})
            path.write_bytes(raw + b' ')
            with self.assertRaises(ValueError):
                load_saved_decisions(path, **kwargs)

    def test_unresolved_partial_and_extra_procedure_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'before.json'
            path.with_suffix('.json.partial').write_text('interrupted')
            kwargs = dict(context={}, block=0, family=STAGE3_TRUTH_CONDITIONS[0])
            with self.assertRaises(FileExistsError):
                persist_saved_decisions(path, saves_fixture(), **kwargs)
            extra = saves_fixture()
            extra['procedures']['undeclared'] = {}
            with self.assertRaises(ValueError):
                seal_saved_decisions(extra, **kwargs)


if __name__ == '__main__':
    unittest.main()
