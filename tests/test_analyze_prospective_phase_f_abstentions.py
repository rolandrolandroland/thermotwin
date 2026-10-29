import copy
import unittest

from tools import analyze_prospective_phase_f_abstentions as diagnosis
from tools import verify_prospective_phase_f_result as audit


class AbstentionLayerTests(unittest.TestCase):
    def row(self, lower, upper, offset=0., correction=0., **extra):
        inp = {'raw_interval': {'lower': lower, 'upper': upper},
               'development_offset_kelvin': offset,
               'verification_succeeded': True, 'pipeline_failure': False, **extra}
        return {**inp, **audit.final_decision(inp, correction, 0.)}

    def test_mutually_exclusive_first_crossing_layers_and_asymmetric_zero_boundary(self):
        examples = [
            (self.row(-.2, .2, .1, .1), 'raw_interval_crosses_zero'),
            (self.row(.05, .2, .1, .1), 'development_padding'),
            (self.row(.1, .3, .05, .1), 'final_calibration'),
            (self.row(.3, .5, .05, .1), 'definitive'),
            (self.row(-.2, 0.), 'raw_interval_crosses_zero'),
            (self.row(0., .2), 'definitive'),
            (self.row(-.5, -.3, .05, .1), 'definitive'),
            (self.row(-.3, -.05, .1), 'development_padding'),
        ]
        for row, expected in examples:
            with self.subTest(expected=expected, interval=row['raw_interval']):
                self.assertEqual(diagnosis.layers(row)['first_abstaining_layer'], expected)

    def test_failed_cases_are_retained_and_saved_decision_changes_are_rejected(self):
        for field in ({'verification_succeeded': False}, {'pipeline_failure': True}, {'raw_interval': None}):
            self.assertEqual(diagnosis.layers(self.row(.2, .5, **field))['first_abstaining_layer'],
                             'invalid_or_unverified_interval')
        wrong = self.row(.2, .5)
        wrong['decision'] = diagnosis.ABSTAIN
        with self.assertRaisesRegex(ValueError, 'differs'):
            diagnosis.layers(wrong)

    def test_local_interval_ambiguity_is_distinct_from_between_model_disagreement(self):
        geometry = diagnosis.envelope_geometry([{'lower': -.3, 'upper': -.2}, {'lower': .2, 'upper': .3}])
        self.assertEqual(geometry['ambiguity'], 'opposite_definitive_model_intervals_only')
        self.assertAlmostEqual(geometry['excess_envelope_span_kelvin'], .5)
        geometry = diagnosis.envelope_geometry([{'lower': -.3, 'upper': .2}, {'lower': .1, 'upper': .4}])
        self.assertEqual(geometry['ambiguity'], 'at_least_one_individual_interval_crosses_zero')
        with self.assertRaises(ValueError):
            diagnosis.envelope_geometry([])


class PredictiveArithmeticTests(unittest.TestCase):
    def test_failures_and_candidate_loss_do_not_disappear_from_scored_widths(self):
        draw = {'failed': False, 'raw_after_width': .1, 'initially_admissible_became_inadmissible': []}
        self.assertAlmostEqual(diagnosis.padded_draw_width(draw, 1., .2), .5)
        self.assertEqual(diagnosis.padded_draw_width({**draw, 'failed': True}, 1., .2), 1.)
        self.assertEqual(diagnosis.padded_draw_width(
            {**draw, 'initially_admissible_became_inadmissible': ['five_state']}, 1., .2), 1.)
        self.assertAlmostEqual(diagnosis.padded_draw_width(
            {**draw, 'raw_after_width': 2., 'initially_admissible_became_inadmissible': ['five_state']}, 1., .2), 2.4)

    def fixture(self):
        draws = []
        for model, raw in (('four_state', .1), ('five_state', .4)):
            for index in range(16):
                draws.append({'policy_name': 'fixed_thermal', 'generator_model': model, 'draw_index': index,
                              'raw_after_width': raw, 'failed': model == 'four_state' and index == 0,
                              'initially_admissible_became_inadmissible': ['four_state'] if model == 'five_state' and index == 0 else [],
                              'initially_excluded_became_admissible': []})
        after = (15 * .8 + 1.) / 16
        evaluation = {'policy_name': 'fixed_thermal', 'eligible': True, 'declared_cost': 2.,
                      'uncertainty_before': 1., 'expected_uncertainty_after': after,
                      'expected_uncertainty_reduction': 1 - after, 'utility_per_cost': (1 - after) / 2}
        return {'n16_complete_uncertainty_result': {
            'acquisition_evidence': {'snapshot': {'admissible_candidate_models': ['four_state', 'five_state'],
                                                 'provisional_margin_envelope': {'lower': -.5, 'upper': .5}}},
            'draw_outcomes': draws}, 'frozen_selection_before_reveal': {'action_evaluations': [evaluation]}}

    def test_mean_within_source_worst_case_across_sources_and_every_draw_retained(self):
        record = diagnosis.forecast_record(self.fixture(), 'fixed_thermal', {'stop_now': 0., 'fixed_thermal': .2})
        self.assertAlmostEqual(record['predicted_padded_width_kelvin'], .8125)
        self.assertEqual(record['draw_count'], 32)
        self.assertEqual(record['failed_draw_count'], 1)
        self.assertEqual(record['candidate_loss_draw_count'], 1)

    def test_missing_duplicate_or_rewritten_predictive_evidence_is_rejected(self):
        for change in ('missing', 'duplicate', 'rewritten'):
            case = copy.deepcopy(self.fixture())
            if change == 'missing':
                case['n16_complete_uncertainty_result']['draw_outcomes'].pop()
            elif change == 'duplicate':
                case['n16_complete_uncertainty_result']['draw_outcomes'][0]['draw_index'] = 1
            else:
                case['frozen_selection_before_reveal']['action_evaluations'][0]['expected_uncertainty_after'] = 0.
            with self.subTest(change=change), self.assertRaises(ValueError):
                diagnosis.forecast_record(case, 'fixed_thermal', {'stop_now': 0., 'fixed_thermal': .2})

    def test_negative_realized_gain_and_narrowed_but_abstaining_cases_are_preserved(self):
        rows = [dict(predicted_reduction_kelvin=.2, realized_verified_reduction_kelvin=value,
                     realized_minus_predicted_width_kelvin=.2 - value,
                     selected=True, selector_decision_if_selected=diagnosis.ABSTAIN,
                     draw_count=16, failed_draw_count=0, candidate_loss_draw_count=0, candidate_recovery_draw_count=0)
                for value in (-.1, .3)]
        summary = diagnosis.forecast_summary(rows)
        self.assertEqual(summary['predicted_positive_but_no_realized_narrowing_count'], 1)
        self.assertEqual(summary['selected_narrowed_but_abstained'], 1)
        self.assertEqual(summary['selected_abstentions'], 2)
        self.assertAlmostEqual(summary['realized_verified_reduction_kelvin']['mean'], .1)


if __name__ == '__main__':
    unittest.main()
