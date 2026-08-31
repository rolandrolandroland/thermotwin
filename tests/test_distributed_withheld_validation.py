import math
import unittest

from thermotwin.simulation.distributed import (
    distributed_reference_experiment,
    run_distributed_leg_experiment,
)
from thermotwin.studies.distributed_inverse_robustness import (
    DistributedInverseRobustnessSeeds,
)
from thermotwin.studies.distributed_withheld_validation import (
    DistributedWithheldPredictionCriteria,
    DistributedWithheldPredictionMetrics,
    DistributedWithheldValidationConfig,
    DistributedWithheldValidationTrial,
    distributed_inverse_experiment_names,
    evaluate_distributed_withheld_prediction,
    summarize_distributed_withheld_validation,
    withheld_prediction_failure_reasons,
)


def _metrics(value: float = 0.01):
    return DistributedWithheldPredictionMetrics(
        cold_face_rmse=value,
        hot_face_rmse=value,
        internal_temperature_rmse=value,
        voltage_rmse=value * 1.0e-3,
        maximum_absolute_temperature_error=2.0 * value,
        maximum_energy_balance_residual=1.0e-14,
    )


def _trial(index: int, *, conventional_success=True, pinn_success=True):
    conventional = _metrics(0.01 if conventional_success else 0.04)
    pinn = _metrics(0.008 if pinn_success else 0.05)
    return DistributedWithheldValidationTrial(
        trial_index=index,
        seeds=DistributedInverseRobustnessSeeds(
            100 + 4 * index,
            (101 + 4 * index, 102 + 4 * index, 103 + 4 * index),
        ),
        conventional_multipliers=(1.04, 1.07, 1.03),
        conventional_maximum_absolute_multiplier_error=0.0,
        conventional_reached_search_bound=False,
        conventional_prediction=conventional,
        conventional_prediction_success=conventional_success,
        conventional_prediction_failure_reasons=(
            () if conventional_success else ("cold_face_rmse",)
        ),
        inverse_pinn_multipliers=(1.05, 1.06, 1.04),
        inverse_pinn_maximum_absolute_multiplier_error=0.01,
        inverse_pinn_prediction=pinn,
        inverse_pinn_prediction_success=pinn_success,
        inverse_pinn_prediction_failure_reasons=(
            () if pinn_success else ("cold_face_rmse",)
        ),
    )


class DistributedWithheldValidationTests(unittest.TestCase):
    def test_default_config_freezes_heldout_regime_noise_and_gate(self):
        config = DistributedWithheldValidationConfig()
        self.assertEqual(config.trial_count, 5)
        self.assertEqual(config.first_seed, 37_001)
        self.assertEqual(config.inverse_pinn_epochs, 600)
        self.assertEqual(config.withheld_experiment_index, 3)
        self.assertEqual(
            distributed_inverse_experiment_names()[3],
            "positive_0.4A_20K_lift",
        )
        self.assertEqual(config.criteria.maximum_cold_face_rmse, 0.03)
        self.assertEqual(config.criteria.maximum_voltage_rmse, 3.0e-5)
        self.assertEqual(config.criteria.maximum_absolute_temperature_error, 0.08)

    def test_configs_reject_invalid_values(self):
        for values in (
            {"trial_count": 0},
            {"first_seed": -1},
            {"inverse_pinn_epochs": 0},
            {"withheld_experiment_index": -1},
            {"observation_interval": 0.0},
            {"temperature_standard_deviation": float("inf")},
            {"voltage_standard_deviation": -1.0},
            {"criteria": "invalid"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    DistributedWithheldValidationConfig(**values)
        with self.assertRaises(ValueError):
            DistributedWithheldPredictionCriteria(maximum_voltage_rmse=0.0)

    def test_identical_trajectory_has_zero_transfer_error_and_closes_energy(self):
        experiment = distributed_reference_experiment(
            temperature_dependent=True,
            duration=0.02,
            cell_count=4,
            time_step=0.001,
        )
        result = run_distributed_leg_experiment(experiment)
        metrics = evaluate_distributed_withheld_prediction(result, result)
        self.assertEqual(metrics.cold_face_rmse, 0.0)
        self.assertEqual(metrics.hot_face_rmse, 0.0)
        self.assertEqual(metrics.internal_temperature_rmse, 0.0)
        self.assertEqual(metrics.voltage_rmse, 0.0)
        self.assertEqual(metrics.maximum_absolute_temperature_error, 0.0)
        self.assertLess(metrics.maximum_energy_balance_residual, 1.0e-12)

    def test_prediction_gate_reports_all_exceeded_and_nonfinite_metrics(self):
        reasons = withheld_prediction_failure_reasons(
            DistributedWithheldPredictionMetrics(
                cold_face_rmse=0.04,
                hot_face_rmse=math.nan,
                internal_temperature_rmse=0.04,
                voltage_rmse=4.0e-5,
                maximum_absolute_temperature_error=0.09,
                maximum_energy_balance_residual=2.0e-10,
            ),
            DistributedWithheldPredictionCriteria(),
        )
        self.assertEqual(
            reasons,
            (
                "cold_face_rmse",
                "nonfinite_hot_face_rmse",
                "internal_temperature_rmse",
                "voltage_rmse",
                "maximum_temperature_error",
                "energy_balance_residual",
            ),
        )

    def test_summary_retains_failed_completed_predictions(self):
        trials = (_trial(0), _trial(1, conventional_success=False, pinn_success=False))
        summary = summarize_distributed_withheld_validation(trials)
        self.assertEqual(summary.trial_count, 2)
        self.assertEqual(summary.conventional_success_count, 1)
        self.assertEqual(summary.inverse_pinn_success_count, 1)
        self.assertEqual(summary.inverse_pinn_completed_count, 2)
        self.assertEqual(summary.conventional_search_bound_hits, 0)
        self.assertAlmostEqual(
            summary.conventional_mean_prediction.cold_face_rmse, 0.025
        )
        self.assertAlmostEqual(summary.inverse_pinn_mean_prediction.cold_face_rmse, 0.029)


if __name__ == "__main__":
    unittest.main()
