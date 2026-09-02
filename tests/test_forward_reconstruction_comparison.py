import unittest

try:
    import torch

    from thermotwin.pinn.forward_piecewise import unipolar_pulse_contact_experiment
    from thermotwin.studies.forward_reconstruction_comparison import (
        ForwardReconstructionComparisonConfig,
        ForwardReconstructionCriteria,
        build_forward_reconstruction_observations,
        forward_reconstruction_seeds,
        run_forward_reconstruction_comparison,
        train_matched_forward_reconstruction_models,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class ForwardReconstructionComparisonTests(unittest.TestCase):
    def test_observation_design_is_sparse_noisy_and_missing_at_turnoff(self):
        config = ForwardReconstructionComparisonConfig(trial_count=1, epochs=1)
        ideal, incomplete = build_forward_reconstruction_observations(
            unipolar_pulse_contact_experiment(),
            config,
            observation_seed=10,
        )

        self.assertEqual(len(ideal.observations), 62)
        self.assertEqual(len(incomplete.observations), 56)
        self.assertFalse(any(17.0 <= item.time <= 23.0 for item in incomplete.observations))
        self.assertEqual({item.location for item in incomplete.observations}, {
            item.location for item in ideal.observations
        })

    def test_pair_starts_from_bit_identical_parameters(self):
        config = ForwardReconstructionComparisonConfig(
            trial_count=1,
            epochs=1,
            hidden_width=4,
            hidden_layers=1,
            collocation_points=12,
        )
        experiment = unipolar_pulse_contact_experiment()
        _, observations = build_forward_reconstruction_observations(
            experiment, config, observation_seed=10
        )

        _, _, difference = train_matched_forward_reconstruction_models(
            experiment, observations, config, neural_seed=11
        )

        self.assertEqual(difference, 0.0)

    def test_tiny_campaign_retains_pair_and_exact_continuity(self):
        result = run_forward_reconstruction_comparison(
            ForwardReconstructionComparisonConfig(
                trial_count=1,
                epochs=1,
                hidden_width=4,
                hidden_layers=1,
                collocation_points=12,
                energy_sampling_interval=2.0,
            )
        )

        self.assertEqual(len(result.trials), 1)
        self.assertEqual(len(result.summaries), 2)
        self.assertEqual(result.trials[0].initialization_maximum_absolute_difference, 0.0)
        self.assertEqual(
            result.trials[0].physics_informed.maximum_boundary_temperature_jump,
            0.0,
        )
        self.assertEqual(
            result.trials[0].data_only.maximum_boundary_temperature_jump,
            0.0,
        )

    def test_seed_blocks_and_invalid_configuration(self):
        self.assertEqual(forward_reconstruction_seeds(100, 0), (100, 101))
        self.assertEqual(forward_reconstruction_seeds(100, 1), (102, 103))
        with self.assertRaises(ValueError):
            forward_reconstruction_seeds(-1, 0)
        with self.assertRaises(ValueError):
            ForwardReconstructionComparisonConfig(trial_count=0)
        with self.assertRaises(ValueError):
            ForwardReconstructionComparisonConfig(missing_start_time=25, missing_end_time=20)
        with self.assertRaises(ValueError):
            ForwardReconstructionCriteria(maximum_hidden_face_rmse=0.0)


if __name__ == "__main__":
    unittest.main()
