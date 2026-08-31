import math
import unittest

from thermotwin.inference.distributed_regularization import (
    second_difference_roughness,
)
from thermotwin.studies.distributed_independent_validation import (
    ESTIMATOR_NAMES,
    DistributedIndependentEstimatorResult,
    DistributedIndependentHoldout,
    DistributedIndependentValidationConfig,
    DistributedIndependentValidationCriteria,
    DistributedIndependentValidationSeeds,
    DistributedIndependentValidationTrial,
    DistributedMismatchPredictionMetrics,
    distributed_independent_validation_experiments,
    independent_validation_seeds,
    smooth_resistivity_truth,
    summarize_distributed_independent_validation,
)


def _metrics(value=0.001):
    return DistributedMismatchPredictionMetrics(
        cold_face_rmse=value,
        hot_face_rmse=value,
        internal_temperature_rmse=value,
        voltage_rmse=value * 1.0e-2,
        maximum_absolute_temperature_error=2.0 * value,
        maximum_prediction_energy_balance_residual=1.0e-15,
    )


def _estimator(name, *, success=True):
    return DistributedIndependentEstimatorResult(
        name=name,
        smoothness_weight=0.0,
        multipliers=(1.04, 1.07, 1.03),
        final_normalized_observation_loss=1.0,
        log_multiplier_roughness=0.01,
        in_support_property_relative_rmse=0.02,
        in_support_property_maximum_relative_error=0.03,
        extended_property_relative_rmse=0.04,
        predictions=(("gated", _metrics()), ("diagnostic", _metrics(0.1))),
        success=success,
        failure_reasons=() if success else ("in_support_property_error",),
    )


class DistributedIndependentValidationTests(unittest.TestCase):
    def test_default_configuration_freezes_independent_truth_and_matched_penalty(self):
        config = DistributedIndependentValidationConfig()
        self.assertEqual(config.trial_count, 3)
        self.assertEqual(config.first_seed, 47_001)
        self.assertEqual(config.inverse_pinn_epochs, 600)
        self.assertEqual(config.truth_node_count, 25)
        self.assertEqual(config.truth_time_step, 2.5e-4)
        self.assertEqual(config.matched_smoothness_weight, 25.0)

    def test_configuration_rejects_invalid_values(self):
        for values in (
            {"trial_count": 0},
            {"first_seed": -1},
            {"inverse_pinn_epochs": 0},
            {"truth_node_count": 4},
            {"truth_time_step": 0.0},
            {"observation_interval": float("nan")},
            {"matched_smoothness_weight": 0.0},
            {"criteria": "invalid"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    DistributedIndependentValidationConfig(**values)
        with self.assertRaises(ValueError):
            DistributedIndependentValidationCriteria(maximum_voltage_rmse=0.0)

    def test_truth_matches_old_knot_values_but_not_between_knot_linear_shape(self):
        truth = smooth_resistivity_truth()
        expected = (1.092e-5, 1.07e-5, 0.9888e-5)
        for temperature, value in zip((285.0, 300.0, 315.0), expected):
            self.assertAlmostEqual(truth.value(temperature), value)
        midpoint_of_endpoints = 0.5 * (truth.value(285.0) + truth.value(300.0))
        self.assertNotAlmostEqual(truth.value(292.5), midpoint_of_endpoints)

    def test_seed_blocks_are_unique_while_pinn_variants_share_initialization(self):
        first = independent_validation_seeds(100, 0, 3)
        second = independent_validation_seeds(100, 1, 3)
        self.assertEqual(first, DistributedIndependentValidationSeeds(100, (101, 102, 103)))
        self.assertEqual(second, DistributedIndependentValidationSeeds(104, (105, 106, 107)))
        self.assertFalse(set((first.neural, *first.observations)) & set((second.neural, *second.observations)))

    def test_shared_roughness_is_zero_for_linear_log_coefficients(self):
        self.assertAlmostEqual(second_difference_roughness((0.0, 0.1, 0.2)), 0.0)
        self.assertAlmostEqual(second_difference_roughness((0.0, 0.0, 0.2)), 0.04)

    def test_experiment_suite_has_two_gated_holdouts_and_one_extrapolation_diagnostic(self):
        prepared = distributed_independent_validation_experiments(
            DistributedIndependentValidationConfig(
                trial_count=1,
                inverse_pinn_epochs=1,
                truth_node_count=9,
                truth_time_step=5.0e-4,
            )
        )
        names, baselines, truths, holdouts, _, _ = prepared
        self.assertEqual(len(names), len(baselines))
        self.assertEqual(len(baselines), len(truths))
        self.assertEqual(sum(item.counts_toward_gate for item in holdouts), 2)
        self.assertIn("outside_support", holdouts[-1].name)

    def test_summary_keeps_failed_estimators_and_excludes_diagnostic_holdout_from_means(self):
        trial = DistributedIndependentValidationTrial(
            trial_index=0,
            seeds=DistributedIndependentValidationSeeds(100, (101, 102, 103)),
            estimators=tuple(
                _estimator(name, success=name != "pinn_unregularized")
                for name in ESTIMATOR_NAMES
            ),
        )
        holdouts = (
            DistributedIndependentHoldout("gated", None, None, None, True),
            DistributedIndependentHoldout("diagnostic", None, None, None, False),
        )
        summaries = summarize_distributed_independent_validation((trial,), holdouts)
        pinn = next(item for item in summaries if item.name == "pinn_unregularized")
        self.assertEqual(pinn.completed_count, 1)
        self.assertEqual(pinn.success_count, 0)
        self.assertAlmostEqual(pinn.mean_in_support_internal_temperature_rmse, 0.001)
        self.assertTrue(math.isfinite(pinn.mean_in_support_voltage_rmse))


if __name__ == "__main__":
    unittest.main()
