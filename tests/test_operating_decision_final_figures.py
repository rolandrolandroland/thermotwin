import copy
import unittest

from tools import operating_decision_final_figure_data as figures
from tools import verify_prospective_phase_f_result as audit


def cell_fixture():
    rows = []
    for block in range(20):
        for family in audit.FAMILIES:
            rows.append({'block': block, 'truth_family': family, 'selected_policy': 'stop_now',
                         'selection_failure': False, 'definitive_decision': block < 10,
                         'pipeline_failure': False, 'verification_failure': False,
                         'no_useful_measurement': block >= 10,
                         'ineligible_actions': ['fixed_face_temperature'] if block == 0 and family == audit.FAMILIES[0] else []})
    actions = [{'action': p, 'selected_count': 60 if p == 'stop_now' else 0,
                'definitive_count': 30 if p == 'stop_now' else 0} for p in (*audit.POLICIES, 'selection_failure')]
    return {'sensor_scenario': 'nominal', 'cost_scenario': 'balanced_face_equal', 'primary': True,
            'case_records': rows, 'summary': {'actions': actions, 'definitive_decision_count': 30,
            'no_useful_measurement_count': 30, 'selection_failure_count': 0, 'verification_failure_count': 0,
            'pipeline_failure_count': 0, 'ineligible_action_counts': {'fixed_thermal': 0, 'fixed_voltage': 0, 'fixed_face_temperature': 1}}}


class FigureDataIntegrityTests(unittest.TestCase):
    def test_map_keeps_abstentions_and_action_ineligibility_in_all_sixty_cases(self):
        cell = figures.map_cell(cell_fixture())
        self.assertEqual(cell['case_count'], 60)
        self.assertEqual(cell['actions'][0], {'action': 'stop_now', 'selected': 60, 'definitive': 30, 'abstentions': 30})
        self.assertEqual(cell['ineligible_actions']['fixed_face_temperature'], 1)
        self.assertEqual(cell['selection_failure'], 0)

    def test_map_rejects_dropped_duplicate_cases_and_rewritten_summaries(self):
        for mode in ('missing', 'duplicate', 'changed_summary'):
            cell = cell_fixture()
            if mode == 'missing': cell['case_records'].pop()
            elif mode == 'duplicate': cell['case_records'][-1] = cell['case_records'][0]
            else: cell['summary']['definitive_decision_count'] = 31
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                figures.map_cell(cell)

    def test_map_does_not_turn_a_whole_case_failure_into_a_successful_stop(self):
        cell = cell_fixture()
        row = cell['case_records'][-1]
        row.update(selection_failure=True, selected_policy=None)
        cell['summary']['selection_failure_count'] = 1
        cell['summary']['actions'][0]['selected_count'] = 59
        cell['summary']['actions'][-1]['selected_count'] = 1
        result = figures.map_cell(cell)
        self.assertEqual(sum(v['selected'] for v in result['actions']), 60)
        self.assertEqual(result['actions'][-1]['abstentions'], 1)

    def test_illustrative_rule_is_first_eligible_case_not_the_best_truth_or_largest_gain(self):
        rows, snapshots = [], {}
        for block, family, action, decision, width, model_count in (
            (0, audit.FAMILIES[0], 'stop_now', 'insufficient_evidence', .2, 2),
            (1, audit.FAMILIES[2], 'fixed_thermal', 'insufficient_evidence', .3, 2),
            (1, audit.FAMILIES[0], 'fixed_thermal', 'approve', .2, 2),
            (2, audit.FAMILIES[0], 'fixed_voltage', 'insufficient_evidence', .1, 2),
            (0, audit.FAMILIES[1], 'fixed_thermal', 'insufficient_evidence', .1, 1)):
            rows.append({'block': block, 'truth_family': family, 'procedure': audit.SELECTOR,
                         'selected_action': action, 'decision': decision, 'raw_interval': {'lower': -.1, 'upper': width - .1},
                         'development_offset_kelvin': 0., 'true_margin': 100 if block == 2 else -100})
            snapshots[block, family] = {'admissible_candidate_models': ['four_state', 'five_state'][:model_count],
                                        'provisional_margin_envelope': {'lower': -.4, 'upper': .4}}
        expected = (1, audit.FAMILIES[2])
        self.assertEqual(figures.illustrative_identity(rows, snapshots), expected)
        self.assertEqual(figures.illustrative_identity(list(reversed(rows)), snapshots), expected)
        for row in rows: row['true_margin'] *= -1
        self.assertEqual(figures.illustrative_identity(rows, snapshots), expected)

    def test_development_padding_counts_in_the_illustrative_narrowing_condition(self):
        row = {'block': 0, 'truth_family': audit.FAMILIES[0], 'procedure': audit.SELECTOR,
               'selected_action': 'fixed_voltage', 'decision': 'insufficient_evidence',
               'raw_interval': {'lower': -.1, 'upper': .1}, 'development_offset_kelvin': .4}
        snap = {(0, audit.FAMILIES[0]): {'admissible_candidate_models': ['four_state', 'five_state'],
                                       'provisional_margin_envelope': {'lower': -.4, 'upper': .4}}}
        with self.assertRaises(ValueError): figures.illustrative_identity([row], snap)

    def test_profiled_sensor_offset_matches_gaussian_shrinkage_and_does_not_invent_bias(self):
        self.assertEqual(figures.profiled_offset([1., 2.], [1., 2.], .02, 2.5), 0.)
        self.assertAlmostEqual(figures.profiled_offset([2., 2.], [1., 1.], 1., 1.), 2 / 3)
        with self.assertRaises(ValueError): figures.profiled_offset([1.], [], .02, 2.5)


if __name__ == '__main__':
    unittest.main()
