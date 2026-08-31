from dataclasses import replace
import math
import unittest

from thermotwin.observations.distributed import DistributedObservationChannels
from thermotwin.simulation.distributed import distributed_reference_experiment
from thermotwin.studies.distributed_inverse_robustness import (
    DistributedInverseRobustnessConfig,
    DistributedInverseRobustnessSeeds,
    DistributedInverseRobustnessTrial,
    DistributedRecoveryCriteria,
    distributed_inverse_robustness_seeds,
    noisy_distributed_inverse_observations,
    recovery_failure_reasons,
    summarize_distributed_inverse_robustness,
)


def _trial(
    index: int,
    *,
    conventional=(1.04, 1.07, 1.03),
    pinn=(1.05, 1.06, 1.04),
    conventional_success=True,
    pinn_success=True,
):
    return DistributedInverseRobustnessTrial(
        trial_index=index,
        seeds=DistributedInverseRobustnessSeeds(
            neural=100 + 3 * index,
            observations=(101 + 3 * index, 102 + 3 * index),
        ),
        conventional_multipliers=conventional,
        conventional_initial_normalized_loss=100.0,
        conventional_final_normalized_loss=1.0,
        conventional_loss_reduction_fraction=0.99,
        conventional_maximum_absolute_multiplier_error=max(
            abs(value - truth)
            for value, truth in zip(conventional, (1.04, 1.07, 1.03))
        ),
        conventional_reached_search_bound=False,
        conventional_success=conventional_success,
        conventional_failure_reasons=() if conventional_success else ("final_loss",),
        inverse_pinn_multipliers=pinn,
        inverse_pinn_initial_normalized_loss=100.0,
        inverse_pinn_final_normalized_loss=1.0,
        inverse_pinn_loss_reduction_fraction=0.99,
        inverse_pinn_maximum_absolute_multiplier_error=(
            max(
                abs(value - truth)
                for value, truth in zip(pinn, (1.04, 1.07, 1.03))
            )
            if pinn is not None
            else math.inf
        ),
        inverse_pinn_success=pinn_success,
        inverse_pinn_failure_reasons=() if pinn_success else ("multiplier_error",),
    )


class DistributedInverseRobustnessTests(unittest.TestCase):
    def test_default_config_predeclares_trials_noise_and_success_gate(self):
        config = DistributedInverseRobustnessConfig()
        self.assertEqual(config.trial_count, 5)
        self.assertEqual(config.first_seed, 27_001)
        self.assertEqual(config.inverse_pinn_epochs, 600)
        self.assertEqual(config.temperature_standard_deviation, 0.01)
        self.assertEqual(config.voltage_standard_deviation, 1.0e-5)
        self.assertEqual(config.criteria.maximum_absolute_multiplier_error, 0.10)
        self.assertEqual(config.criteria.minimum_loss_reduction_fraction, 0.90)
        self.assertEqual(config.criteria.maximum_final_normalized_loss, 5.0)

    def test_configs_reject_invalid_values(self):
        for values in (
            {"trial_count": 0},
            {"trial_count": True},
            {"first_seed": -1},
            {"inverse_pinn_epochs": 0},
            {"observation_interval": 0.0},
            {"temperature_standard_deviation": float("nan")},
            {"voltage_standard_deviation": -1.0},
            {"criteria": "invalid"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    DistributedInverseRobustnessConfig(**values)
        for values in (
            {"maximum_absolute_multiplier_error": 0.0},
            {"minimum_loss_reduction_fraction": 1.0},
            {"maximum_final_normalized_loss": float("inf")},
        ):
            with self.subTest(criteria=values):
                with self.assertRaises(ValueError):
                    DistributedRecoveryCriteria(**values)

    def test_seed_blocks_are_reproducible_and_nonoverlapping(self):
        self.assertEqual(
            distributed_inverse_robustness_seeds(100, 0, 4),
            DistributedInverseRobustnessSeeds(100, (101, 102, 103, 104)),
        )
        self.assertEqual(
            distributed_inverse_robustness_seeds(100, 1, 4),
            DistributedInverseRobustnessSeeds(105, (106, 107, 108, 109)),
        )
        all_seeds = tuple(
            seed
            for trial_index in range(10)
            for seed in (
                distributed_inverse_robustness_seeds(100, trial_index, 4).neural,
                *distributed_inverse_robustness_seeds(
                    100, trial_index, 4
                ).observations,
            )
        )
        self.assertEqual(len(all_seeds), len(set(all_seeds)))

    def test_noisy_observations_repeat_and_use_independent_regime_seeds(self):
        first = distributed_reference_experiment(
            temperature_dependent=True,
            current=0.4,
            duration=0.02,
            cell_count=4,
            time_step=0.001,
        )
        second = replace(first, current=-0.4)
        arguments = dict(
            observation_interval=0.01,
            temperature_standard_deviation=0.01,
            voltage_standard_deviation=1.0e-5,
        )
        noisy = noisy_distributed_inverse_observations(
            (first, second), seeds=(10, 11), **arguments
        )
        repeated = noisy_distributed_inverse_observations(
            (first, second), seeds=(10, 11), **arguments
        )
        changed = noisy_distributed_inverse_observations(
            (first, second), seeds=(12, 13), **arguments
        )
        self.assertEqual(noisy, repeated)
        self.assertNotEqual(noisy, changed)
        channels = DistributedObservationChannels().names()
        self.assertEqual(
            tuple(item.channel for item in noisy[0].observations[:3]), channels
        )

    def test_predeclared_gate_reports_every_failed_reason(self):
        reasons = recovery_failure_reasons(
            maximum_multiplier_error=0.11,
            initial_normalized_loss=10.0,
            final_normalized_loss=6.0,
            criteria=DistributedRecoveryCriteria(),
        )
        self.assertEqual(
            reasons,
            ("multiplier_error", "final_loss", "loss_reduction"),
        )

    def test_summary_retains_completed_threshold_failures(self):
        failed = _trial(
            1,
            pinn=(0.8, 0.8, 0.8),
            pinn_success=False,
        )
        summary = summarize_distributed_inverse_robustness((_trial(0), failed))
        self.assertEqual(summary.trial_count, 2)
        self.assertEqual(summary.inverse_pinn_completed_count, 2)
        self.assertEqual(summary.inverse_pinn_success_count, 1)
        self.assertEqual(summary.conventional_search_bound_hits, 0)
        self.assertGreater(summary.inverse_pinn_multiplier_rmse, 0.1)
        self.assertAlmostEqual(summary.inverse_pinn_mean_multipliers[0], 0.925)


if __name__ == "__main__":
    unittest.main()
