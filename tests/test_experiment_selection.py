import unittest

from thermotwin.experiment_selection import (
    ExperimentSelectionConfig,
    run_next_experiment_selection,
)


class ExperimentSelectionTests(unittest.TestCase):
    def test_configuration_rejects_invalid_prior_shape(self):
        with self.assertRaises(ValueError):
            ExperimentSelectionConfig(prior_standard_deviations=(0.1, 0.2))

    def test_selection_respects_budget_and_beats_naive_information(self):
        config = ExperimentSelectionConfig(
            current_amplitudes=(0.4, 0.8),
            pulse_durations=(5.0, 20.0),
            monte_carlo_trials=80,
        )

        result = run_next_experiment_selection(config)

        self.assertLessEqual(
            result.selected.electrical_energy,
            config.maximum_electrical_energy,
        )
        self.assertTrue(result.selected.feasible)
        self.assertGreater(
            result.selected.information_gain_nats,
            result.naive.information_gain_nats,
        )
        self.assertGreater(result.validation.rmse_reduction_percent, 0.0)
        self.assertGreater(result.validation.selected_nominal_95_coverage, 0.85)
        self.assertLess(result.validation.selected_nominal_95_coverage, 1.0)

    def test_energy_budget_can_make_high_energy_candidate_infeasible(self):
        result = run_next_experiment_selection(
            ExperimentSelectionConfig(
                current_amplitudes=(0.4, 1.2),
                pulse_durations=(5.0, 30.0),
                maximum_electrical_energy=20.0,
                monte_carlo_trials=20,
            )
        )

        high_energy = next(
            candidate
            for candidate in result.candidates
            if candidate.current_amplitude == 1.2
            and candidate.pulse_duration == 30.0
        )
        self.assertFalse(high_energy.feasible)


if __name__ == "__main__":
    unittest.main()
