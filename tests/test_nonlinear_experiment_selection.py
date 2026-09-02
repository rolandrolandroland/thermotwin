from dataclasses import replace
import unittest

from thermotwin.inference.joint_thermal_parameters import JointThermalFitConfig
from thermotwin.studies.nonlinear_experiment_selection import (
    NonlinearExperimentSelectionConfig,
    nonlinear_experiment_definitions,
    nonlinear_experiment_trial_seeds,
    run_nonlinear_experiment_selection_study,
)


class NonlinearExperimentSelectionTests(unittest.TestCase):
    def test_definitions_keep_selected_naive_and_resource_control_distinct(self):
        definitions = nonlinear_experiment_definitions(
            NonlinearExperimentSelectionConfig(trial_count=1)
        )

        self.assertEqual(tuple(item.role for item in definitions), (
            "selected", "naive", "resource_control"
        ))
        self.assertEqual(definitions[0].candidate.name, "0.8A_20s")
        self.assertEqual(definitions[1].candidate.name, "0.4A_5s")
        self.assertEqual(definitions[2].candidate.name, "0.6A_30s")
        self.assertGreater(definitions[2].candidate.electrical_energy, 20.0)

    def test_seed_blocks_are_reproducible_and_nonoverlapping(self):
        self.assertEqual(nonlinear_experiment_trial_seeds(100, 0), (100, 101))
        self.assertEqual(nonlinear_experiment_trial_seeds(100, 1), (102, 103))
        with self.assertRaises(ValueError):
            nonlinear_experiment_trial_seeds(-1, 0)

    def test_small_nonlinear_campaign_has_paired_trials_and_rank_gate(self):
        config = NonlinearExperimentSelectionConfig(
            trial_count=1,
            fit=JointThermalFitConfig(gauss_newton_iterations=3),
        )

        result = run_nonlinear_experiment_selection_study(
            config, include_profiles=False
        )

        self.assertEqual(len(result.trials), 3)
        self.assertEqual(len({trial.truth_seed for trial in result.trials}), 1)
        self.assertEqual(len({trial.noise_seed for trial in result.trials}), 1)
        self.assertEqual(result.selected_identifiability.supported_rank, 3)
        self.assertEqual(result.zero_current_identifiability.supported_rank, 0)
        self.assertEqual(len(result.summaries), 3)

    def test_invalid_campaign_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            NonlinearExperimentSelectionConfig(trial_count=0)
        with self.assertRaises(ValueError):
            NonlinearExperimentSelectionConfig(profile_log_offsets=())


if __name__ == "__main__":
    unittest.main()
