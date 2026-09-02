import unittest

try:
    import torch

    from thermotwin.studies.imperfect_inverse_pinn import (
        ImperfectInversePINNConfig,
        build_imperfect_inverse_pinn_problem,
        imperfect_inverse_pinn_cases,
        imperfect_inverse_pinn_seeds,
        run_imperfect_inverse_pinn_study,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class ImperfectInversePINNStudyTests(unittest.TestCase):
    def test_cases_include_recovery_and_intentional_mismatch(self):
        cases = imperfect_inverse_pinn_cases()

        self.assertEqual(len(cases), 4)
        self.assertTrue(all(case.expected_recovery for case in cases[:3]))
        self.assertFalse(cases[-1].expected_recovery)
        self.assertEqual(cases[2].turnoff_half_width, 2.0)
        self.assertGreater(cases[-1].cold_face_bias, 0.0)

    def test_missingness_removes_aligned_cold_pair_readings(self):
        complete = build_imperfect_inverse_pinn_problem(
            imperfect_inverse_pinn_cases()[0], 10
        )
        missing = build_imperfect_inverse_pinn_problem(
            imperfect_inverse_pinn_cases()[1], 10
        )

        self.assertLess(len(missing.observations.time), len(complete.observations.time))
        self.assertEqual(
            len(missing.observations.cold_face),
            len(missing.observations.cold_exchanger),
        )
        self.assertNotIn(20.0, missing.observations.time)

    def test_seeds_do_not_collide_across_cases_or_trials(self):
        values = {
            value
            for case in range(4)
            for trial in range(3)
            for value in imperfect_inverse_pinn_seeds(100, case, trial)
        }
        self.assertEqual(len(values), 24)

    def test_tiny_campaign_retains_every_case(self):
        result = run_imperfect_inverse_pinn_study(
            ImperfectInversePINNConfig(
                trial_count_per_case=1,
                initial_resistances=(0.5,),
                training_epochs=1,
                hidden_width=4,
                hidden_layers=1,
                collocation_points=8,
            )
        )

        self.assertEqual(len(result.trials), 4)
        self.assertEqual(len(result.summaries), 4)
        self.assertTrue(all(summary.trial_count == 1 for summary in result.summaries))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            ImperfectInversePINNConfig(trial_count_per_case=0)
        with self.assertRaises(ValueError):
            ImperfectInversePINNConfig(
                trial_count_per_case=2,
                initial_resistances=(0.5,),
            )


if __name__ == "__main__":
    unittest.main()
