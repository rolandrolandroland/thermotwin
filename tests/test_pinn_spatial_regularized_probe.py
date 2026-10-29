"""Semantic tests of the prospective derivative gates; no scientific evidence."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1]/'tools/pinn_spatial/regularized-gradient-probe.py'
SPEC = importlib.util.spec_from_file_location('regularized_gradient_probe_tests', PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def row():
    samples = []
    for point in [value for value in probe.protocol() if value['case'] == 't02']:
        displacement = point['radius']*point['sign']
        # Known smooth profiled total; physical and penalty need not share
        # that derivative, so the FD must use the declared total objective.
        total = 3.-2.*displacement+7.*displacement**2
        penalty = .5+displacement
        samples.append(dict(point, status='solved', total_objective=total,
            physical_objective=total-penalty, penalty=penalty,
            projection={'rank': 2, 'columns': 2,
                'hard_singular_value_truncation': False, 'head_solve_calls': 1,
                'stationarity_relative': 1.e-12, 'algebraic_objective': total}))
    return {'case': 't02', 'samples': samples,
        'direction': {'ad_directional_derivative': -2., 'gradient_clipped': False},
        'original_state_restored_exactly': True}


class RegularizedProbeGateTests(unittest.TestCase):
    def test_exact_profiled_total_derivative_and_all_declared_points_pass(self):
        result = probe.case_checks(row())
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['point_checks']))
        self.assertEqual(len(result['point_checks']), 7)
        for value in result['directional_differences']:
            self.assertAlmostEqual(value['centered_fd'], -2., places=8)

    def test_unregularized_derivative_is_not_the_declared_total_derivative(self):
        value = row()
        value['direction']['ad_directional_derivative'] = -3.
        result = probe.case_checks(value)
        self.assertTrue(all(result['point_checks']))
        self.assertFalse(result['finest_fd_agrees_with_ad'])
        self.assertFalse(result['passed'])

    def test_best_coarse_radius_cannot_rescue_wrong_predeclared_finest(self):
        value = row()
        for sample in value['samples']:
            if sample['radius'] == probe.RADII[-1]:
                change = sample['sign']*sample['radius']
                sample['total_objective'] += change
                sample['physical_objective'] += change
                sample['projection']['algebraic_objective'] += change
        result = probe.case_checks(value)
        self.assertAlmostEqual(result['directional_differences'][0]['centered_fd'], -2., places=8)
        self.assertFalse(result['finest_fd_agrees_with_ad'])
        self.assertFalse(result['passed'])

    def test_two_finest_consistency_is_independent_of_finest_ad_match(self):
        value = row()
        for sample in value['samples']:
            if sample['radius'] == probe.RADII[-2]:
                change = sample['sign']*sample['radius']
                sample['total_objective'] += change
                sample['physical_objective'] += change
                sample['projection']['algebraic_objective'] += change
        result = probe.case_checks(value)
        self.assertTrue(result['finest_fd_agrees_with_ad'])
        self.assertFalse(result['two_finest_fd_agree'])
        self.assertFalse(result['passed'])

    def test_stationarity_rank_truncation_and_objective_mismatch_block(self):
        changes = [dict(stationarity_relative=2.e-8), dict(rank=1),
                   dict(hard_singular_value_truncation=True),
                   dict(algebraic_objective=50.), dict(columns=None, rank=None),
                   dict(stationarity_relative=-1.)]
        for change in changes:
            with self.subTest(change=change):
                value = row()
                value['samples'][0]['projection'].update(change)
                self.assertFalse(probe.case_checks(value)['passed'])

    def test_failed_unrun_nonfinite_and_bad_restoration_cannot_pass(self):
        for status in ('failed','not_run'):
            value = row(); value['samples'][3] = dict(value['samples'][3], status=status)
            self.assertFalse(probe.case_checks(value)['passed'])
        value = row(); value['samples'][0]['penalty'] = math.inf
        self.assertFalse(probe.case_checks(value)['passed'])
        value = row(); value['original_state_restored_exactly'] = False
        self.assertFalse(probe.case_checks(value)['passed'])

    def test_fourteen_opportunities_and_tolerances_are_not_relative_only(self):
        declared = probe.protocol()
        self.assertEqual(len(declared), 14)
        self.assertEqual(len({(p['case'],p['radius'],p['sign']) for p in declared}), 14)
        self.assertTrue(probe.close(0., 5.e-5, 1.e-4, .005))
        self.assertFalse(probe.close(0., 5.e-3, 1.e-4, .005))
        self.assertFalse(probe.close(math.nan, 0., 1.e-4, .005))


if __name__ == '__main__':
    unittest.main()
