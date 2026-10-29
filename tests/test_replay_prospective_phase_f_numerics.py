import copy
import importlib
from pathlib import Path
import tempfile
import unittest

from tools import replay_prospective_phase_f_numerics as replay
from tools import verify_prospective_phase_f_result as audit


def fixture():
    rows, cases = [], []
    for block in range(3):
        for family in audit.FAMILIES:
            decision, bounds = (('approve', (.1, .5)), ('reject', (-.5, -.1)),
                                ('insufficient_evidence', (-.01, .5)))[block]
            rows.append({'block': block, 'truth_family': family, 'procedure': audit.SELECTOR,
                         'selected_action': 'stop_now', 'decision': decision,
                         'final_interval': {'lower': bounds[0], 'upper': bounds[1]}, 'pipeline_failure': False})
            draws = []
            if block == 0 and family == audit.FAMILIES[0]:
                for action in audit.POLICIES[1:]:
                    draws.append({'policy_name': action, 'generator_model': 'four_state', 'draw_index': 0,
                                  'failed': False, 'initially_admissible_became_inadmissible': [],
                                  'initially_excluded_became_admissible': [],
                                  'candidate_outcomes': [{'status': 'admissible'}]})
                for index, status in enumerate(('fit_reached_bound', 'optimizer_not_converged'), 1):
                    draws.append({'policy_name': 'fixed_thermal', 'generator_model': 'four_state', 'draw_index': index,
                                  'failed': False, 'initially_admissible_became_inadmissible': ['five_state'],
                                  'initially_excluded_became_admissible': [],
                                  'candidate_outcomes': [{'status': status}, {'status': 'admissible'}]})
                draws.append({'policy_name': 'fixed_face_temperature', 'generator_model': 'four_state', 'draw_index': 1,
                              'failed': True, 'initially_admissible_became_inadmissible': [],
                              'initially_excluded_became_admissible': [], 'candidate_outcomes': []})
            cases.append({'block': block, 'truth_condition': family,
                          'n16_complete_uncertainty_result': {'draw_outcomes': draws}})
    return rows, cases


class ReplayPlanTests(unittest.TestCase):
    def test_all_family_action_cells_covered_and_absent_selected_cells_labeled(self):
        rows, cases = fixture()
        sample = replay.choose_targets(rows, cases)
        pairs = {(t['family'], p) for t in sample['tasks'] for p in t['policies']}
        self.assertEqual(pairs, {(f, p) for f in audit.FAMILIES for p in audit.POLICIES})
        self.assertEqual(len(sample['absent_selected_family_action_cells']), 9)
        self.assertEqual(sample['primary_pipeline_failure_count'], 0)
        self.assertEqual(sample['available_whole_predictive_failure_count'], 1)

    def test_boundary_ties_and_draw_targets_are_deterministic_under_input_reordering(self):
        rows, cases = fixture()
        expected = replay.choose_targets(rows, cases)
        self.assertEqual(expected, replay.choose_targets(list(reversed(rows)), list(reversed(cases))))
        targets = [r for t in expected['tasks'] for r in t['draws']]
        self.assertEqual(len(targets), 6)
        reasons = [r for d in targets for r in d['reasons']]
        self.assertIn('whole_predictive_draw_failure', reasons)
        self.assertIn('usable_candidate_exclusion_optimizer_not_converged', reasons)
        for label, block in (('approve', 0), ('reject', 1), ('insufficient_evidence', 2)):
            selected = [t for t in expected['tasks'] if any('nearest_saved_zero_endpoint_' + label in v for v in t['policies'].values())]
            self.assertEqual([(t['block'], t['family']) for t in selected], [(block, audit.FAMILIES[0])])

    def test_missing_failure_stratum_fails_instead_of_substituting_a_successful_draw(self):
        rows, cases = fixture()
        for case in cases:
            case['n16_complete_uncertainty_result']['draw_outcomes'] = [d for d in case['n16_complete_uncertainty_result']['draw_outcomes'] if not d['failed']]
        with self.assertRaisesRegex(ValueError, 'missing predictive replay stratum'):
            replay.choose_targets(rows, cases)


class ReplayIntegrityTests(unittest.TestCase):
    def test_scientific_digest_ignores_performance_but_detects_numerical_changes(self):
        value = {'task_id': 'case-00', 'passed': True, 'common_fit_set': {'objective': .5},
                 'started_at_utc': 'a', 'completed_at_utc': 'b', 'performance': {'cpu_seconds': 1.}}
        changed = {**value, 'started_at_utc': 'c', 'completed_at_utc': 'd', 'performance': {'cpu_seconds': 2.}}
        digest = lambda v: audit.digest(replay.CASE_DOMAIN, replay.worker_scientific_payload(v))
        self.assertEqual(digest(value), digest(changed))
        self.assertNotEqual(audit.canonical(value), audit.canonical(changed))
        self.assertNotEqual(digest(value), digest({**value, 'common_fit_set': {'objective': .6}}))

    def test_comparison_ignores_only_elapsed_decision_time_and_requires_exact_numbers(self):
        saved = {'decision_computation_seconds': 4., 'margin_envelope': {'lower': .1, 'upper': .3}, 'converged': True}
        timed = {**saved, 'decision_computation_seconds': 8.}
        replay.exact(replay.numerical_save(timed), replay.numerical_save(saved), 'clock')
        for changed in ({**saved, 'converged': False}, {**saved, 'converged': 1},
                        {**saved, 'margin_envelope': {'lower': .10000000000000002, 'upper': .3}}):
            with self.assertRaises(replay.ReplayDifference) as caught:
                replay.exact(replay.numerical_save(changed), replay.numerical_save(saved), 'science')
            self.assertEqual(caught.exception.details['expected'], replay.numerical_save(saved))

    def test_case_transport_preserves_observations_currents_and_probe_loading(self):
        modules = {name: importlib.import_module('thermotwin.' + name) for name in (
            'core.controls', 'studies.operating_decision', 'studies.operating_decision_realism',
            'studies.sensor_model_discrimination', 'studies.operating_decision_prospective_pilot')}
        c, o, r, s, p = (modules[name] for name in ('core.controls', 'studies.operating_decision',
                                                   'studies.operating_decision_realism',
                                                   'studies.sensor_model_discrimination', 'studies.operating_decision_prospective_pilot'))
        for policy in (o.default_fixed_policies()[0], o.default_fixed_policies()[-1]):
            def run(regime):
                return r.RealisticOperatingRun(regime, s.ObservableRun(regime.name, regime.current,
                    tuple(s.ObservableValue(channel, t, 299. + t) for t in (0., 1.) for channel in regime.channels)),
                    r.RunInstrumentation(o.COLD_FACE in regime.channels))
            regimes = (o.initial_acquisition_regime(), *policy.additional_regimes)
            channels = tuple(sorted({channel for regime in regimes for channel in regime.channels}))
            verification = o.OperatingRegime('verification', o.VERIFICATION, c.PiecewiseConstantCurrent.constant(.2), channels)
            final = o.OperatingRegime('untouched_final_operating_schedule', o.FINAL_EVALUATION, o.final_operating_current(), ())
            case = r.RealisticBlindedOperatingCase(r.Stage3CaseId('audit-test', policy.name, 0, r.CORRECTED_FINAL_SCHEDULE_ID),
                policy, tuple(run(v) for v in regimes), run(verification), final)
            payload = p._case_payload(case)
            replay.exact(p._case_payload(replay.decode_case(payload, modules)), payload, 'round trip')
            corrupted = copy.deepcopy(payload)
            corrupted['acquisition_runs'][0]['instrumentation']['temporary_face_sensor'] = True
            with self.assertRaises(ValueError):
                replay.decode_case(corrupted, modules)

    def test_saved_replay_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'accepted.json'
            replay.write_new(path, {'accepted': True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                replay.write_new(path, {'accepted': False})
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
