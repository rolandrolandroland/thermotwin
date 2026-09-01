import math
import unittest

from thermotwin.pinn.distributed_inverse import InverseDistributedHistory
from thermotwin.studies.distributed_inverse_robustness import (
    DistributedInverseRobustnessSeeds,
)
from thermotwin.studies.distributed_pinn_training_audit import (
    DistributedPINNTrainingAuditConfig,
    DistributedPINNTrainingAuditTrial,
    distributed_curve_shape_metrics,
    distributed_training_checkpoint,
    summarize_distributed_pinn_training_audit,
)


def _history(property_values, *, physics_loss=0.12, observation_loss=1.5):
    count = len(property_values)
    return InverseDistributedHistory(
        total_loss=tuple(physics_loss + observation_loss for _ in range(count)),
        physics_loss=tuple(physics_loss for _ in range(count)),
        observation_loss=tuple(observation_loss for _ in range(count)),
        smoothness_loss=tuple(0.01 for _ in range(count)),
        shrinkage_loss=tuple(0.0 for _ in range(count)),
        property_values=tuple(property_values),
    )


class DistributedPINNTrainingAuditTests(unittest.TestCase):
    def test_default_config_is_small_frozen_cpu_budget(self):
        config = DistributedPINNTrainingAuditConfig()
        self.assertEqual(config.trial_count, 3)
        self.assertEqual(config.checkpoint_epochs, (600, 1_200, 2_400))
        self.assertEqual(config.physics_weight, 10.0)
        self.assertEqual(config.maximum_normalized_observation_loss, 2.0)
        self.assertEqual(config.maximum_physics_residual_ratio, 0.25)

    def test_config_rejects_invalid_values(self):
        for values in (
            {"trial_count": 0},
            {"first_seed": -1},
            {"checkpoint_epochs": ()},
            {"checkpoint_epochs": (600, 600)},
            {"checkpoint_epochs": (1_200, 600)},
            {"maximum_physics_residual_ratio": 0.0},
            {"physics_weight": 0.0},
            {"minimum_shape_ratio": 2.0, "maximum_shape_ratio": 1.0},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    DistributedPINNTrainingAuditConfig(**values)

    def test_shape_metrics_distinguish_exact_curve_from_flat_mean(self):
        exact = distributed_curve_shape_metrics((1.04, 1.07, 1.03))
        flat = distributed_curve_shape_metrics((1.0466666667,) * 3)
        self.assertAlmostEqual(exact.amplitude_ratio, 1.0)
        self.assertAlmostEqual(exact.center_contrast_ratio, 1.0)
        self.assertTrue(exact.improves_over_best_constant)
        self.assertAlmostEqual(flat.amplitude_ratio, 0.0)
        self.assertAlmostEqual(flat.center_contrast_ratio, 0.0)
        self.assertFalse(flat.improves_over_best_constant)

    def test_checkpoint_separates_operational_and_truth_known_gates(self):
        config = DistributedPINNTrainingAuditConfig(
            trial_count=1, checkpoint_epochs=(1,)
        )
        checkpoint = distributed_training_checkpoint(
            _history(((1.0466666667,) * 3,)),
            epoch=1,
            baseline_values=(1.0, 1.0, 1.0),
            reference_rate_rms=1.0,
            residual_rate_scale=1.0,
            config=config,
        )
        self.assertAlmostEqual(checkpoint.physics_residual_rms, 0.2)
        self.assertTrue(checkpoint.operationally_acceptable)
        self.assertFalse(checkpoint.shape_recovered)
        self.assertIn("amplitude_ratio", checkpoint.shape_failure_reasons)

    def test_checkpoint_reports_failed_physics_without_truth_metric_leakage(self):
        config = DistributedPINNTrainingAuditConfig(
            trial_count=1, checkpoint_epochs=(1,)
        )
        checkpoint = distributed_training_checkpoint(
            _history(((1.04, 1.07, 1.03),), physics_loss=0.75),
            epoch=1,
            baseline_values=(1.0, 1.0, 1.0),
            reference_rate_rms=1.0,
            residual_rate_scale=1.0,
            config=config,
        )
        self.assertFalse(checkpoint.operationally_acceptable)
        self.assertEqual(checkpoint.operational_failure_reasons, ("physics_residual",))
        self.assertTrue(checkpoint.shape_recovered)

    def test_summary_keeps_each_checkpoint_and_trial(self):
        config = DistributedPINNTrainingAuditConfig(
            trial_count=1, checkpoint_epochs=(1, 2)
        )
        history = _history(
            ((1.0466666667,) * 3, (1.04, 1.07, 1.03)),
        )
        checkpoints = tuple(
            distributed_training_checkpoint(
                history,
                epoch=epoch,
                baseline_values=(1.0, 1.0, 1.0),
                reference_rate_rms=1.0,
                residual_rate_scale=1.0,
                config=config,
            )
            for epoch in (1, 2)
        )
        trial = DistributedPINNTrainingAuditTrial(
            trial_index=0,
            seeds=DistributedInverseRobustnessSeeds(1, (2, 3, 4, 5)),
            checkpoints=checkpoints,
            first_operational_epoch=1,
        )
        summaries = summarize_distributed_pinn_training_audit((trial,))
        self.assertEqual(tuple(item.epoch for item in summaries), (1, 2))
        self.assertEqual(summaries[0].shape_recovered_count, 0)
        self.assertEqual(summaries[1].shape_recovered_count, 1)
        self.assertTrue(math.isfinite(summaries[1].mean_coefficient_rmse))


if __name__ == "__main__":
    unittest.main()
