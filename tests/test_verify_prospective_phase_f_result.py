from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools import verify_prospective_phase_f_result as audit


class IndependentStatisticalCalculationTests(unittest.TestCase):
    def test_binomial_limits_match_bernoulli_and_zero_error_closed_forms(self):
        self.assertEqual(audit.binomial_bounds(0, 0)['upper'], None)
        self.assertAlmostEqual(audit.binomial_bounds(0, 1)['upper'], 1 - audit.ALPHA, places=14)
        self.assertAlmostEqual(audit.binomial_bounds(1, 1)['lower'], audit.ALPHA, places=16)
        self.assertAlmostEqual(audit.binomial_bounds(0, 100)['upper'],
                               1 - audit.ALPHA**.01, places=14)
        self.assertAlmostEqual(audit.binomial_bounds(100, 100)['lower'],
                               audit.ALPHA**.01, places=14)

    def test_two_sided_binomial_limits_have_complement_symmetry(self):
        a, b = audit.binomial_bounds(19, 100), audit.binomial_bounds(81, 100)
        self.assertAlmostEqual(a['lower'] + b['upper'], 1., places=14)
        self.assertAlmostEqual(a['upper'] + b['lower'], 1., places=14)
        self.assertGreater(audit.binomial_bounds(0, 19)['upper'], .10)

    def test_student_t_integral_matches_cauchy_distribution(self):
        for value in (0., .5, 1., 4.):
            self.assertAlmostEqual(audit.student_t_tail(value, 1),
                                   .5 - math.atan(value) / math.pi, places=14)
        exact_critical = math.tan(math.pi * (.5 - audit.ALPHA))
        self.assertAlmostEqual(audit.student_t_critical(1), exact_critical, places=7)
        self.assertAlmostEqual(audit.student_t_tail(audit.student_t_critical(99), 99),
                               audit.ALPHA, places=14)

    def test_independent_comparison_rejects_changed_counts_and_pass_fail_flags(self):
        for value in ({'count': 2, 'passed': True}, {'count': 1, 'passed': False},
                      {'count': 1., 'passed': True}):
            with self.assertRaises(ValueError):
                audit.compare({'count': 1, 'passed': True}, value)
        with self.assertRaises(ValueError):
            audit.compare({'bound': .10}, {'bound': .101})


class IndependentRecordChecks(unittest.TestCase):
    def input(self, lower, upper, **overrides):
        return {'raw_interval': {'lower': lower, 'upper': upper},
                'development_offset_kelvin': 0., 'verification_succeeded': True,
                'pipeline_failure': False, **overrides}

    def test_decision_boundaries_and_exact_expansion(self):
        self.assertEqual(audit.final_decision(self.input(0., .5), 0., 0.)['decision'], 'approve')
        self.assertEqual(audit.final_decision(self.input(-.5, 0.), 0., 0.)['decision'], 'insufficient_evidence')
        self.assertEqual(audit.final_decision(self.input(-.5, -.25), 0., 0.)['decision'], 'reject')
        result = audit.final_decision(self.input(-.25, .25, development_offset_kelvin=.125), .125, 0.)
        self.assertEqual(result['final_interval'], {'lower': -.5, 'upper': .5})
        exact = Fraction(.1) - Fraction(.2) - Fraction(.3)
        self.assertLessEqual(Fraction(audit.outward(exact, True)), exact)
        self.assertGreaterEqual(Fraction(audit.outward(exact, False)), exact)

    def test_failures_are_retained_as_abstentions_and_extra_truth_is_rejected(self):
        for inp in (self.input(-.5, -.25, verification_succeeded=False),
                    self.input(-.5, -.25, pipeline_failure=True),
                    self.input(-.5, math.nan), self.input(-.5, -.25, raw_interval=None)):
            result = audit.final_decision(inp, 0., 0.)
            self.assertEqual(result['decision'], 'insufficient_evidence')
            self.assertIsNone(result['final_interval'])
        with self.assertRaises(ValueError):
            audit.final_decision({**self.input(0., .5), 'true_margin': .25}, 0., 0.)

    def test_case_matrix_rejects_missing_and_duplicate_cells(self):
        with self.assertRaises(ValueError):
            audit.matrix([])
        row = {'block': 0, 'truth_family': audit.FAMILIES[0], 'procedure': audit.SELECTOR}
        with self.assertRaises(ValueError):
            audit.matrix([row, row])

    def test_content_seal_and_final_json_bytes_are_both_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'
            body = {'schema_version': 1, 'decision': 'approve'}
            value = {**body, 'archive_content_digest': audit.digest('fixture', body), 'archive_size_bytes': 0}
            for _ in range(5):
                value['archive_size_bytes'] = len(audit.canonical(value) + b'\n')
            raw = audit.canonical(value) + b'\n'
            path.write_bytes(raw)
            self.assertEqual(audit.load_sealed(path, 'fixture'), value)
            path.write_bytes(raw + b' ')
            with self.assertRaises(ValueError):
                audit.load_sealed(path, 'fixture')
            changed = json.loads(raw)
            changed['decision'] = 'reject'
            path.write_bytes(audit.canonical(changed) + b'\n')
            with self.assertRaises(ValueError):
                audit.load_sealed(path, 'fixture')


if __name__ == '__main__':
    unittest.main()
