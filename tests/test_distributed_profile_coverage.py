import math
import unittest

from thermotwin.inference.distributed_profile_likelihood import (
    DistributedProfileInterval,
    DistributedProfileLikelihoodConfig,
)
from thermotwin.studies.distributed_profile_coverage import (
    DistributedProfileCoverageConfig,
    DistributedProfileCoverageEstimatorResult,
    DistributedProfileCoverageSeeds,
    DistributedProfileCoverageTrial,
    distributed_profile_coverage_seeds,
    summarize_distributed_profile_coverage,
)


def _interval(level, lower, upper, *, bound=False):
    return DistributedProfileInterval(
        confidence_level=level,
        threshold=1.0 if level < 0.9 else 3.841459,
        lower_log_multiplier=lower,
        upper_log_multiplier=upper,
        lower_hits_bound=bound,
        upper_hits_bound=False,
    )


def _estimator(name, *, intervals=True):
    interval_68 = (_interval(0.6827, -0.1, 0.1),) * 3 if intervals else ()
    interval_95 = (_interval(0.95, -0.2, 0.2),) * 3 if intervals else ()
    return DistributedProfileCoverageEstimatorResult(
        name=name,
        multipliers=(1.0, 1.0, 1.0),
        property_relative_rmse=0.02,
        property_maximum_relative_error=0.03,
        normalized_observation_loss=1.0,
        intervals_68=interval_68,
        intervals_95=interval_95,
        coefficient_coverage_68=(True, False, True) if intervals else (),
        coefficient_coverage_95=(True, True, True) if intervals else (),
        holdout_internal_temperature_rmse=0.001,
        holdout_voltage_rmse=1.0e-5,
    )


class DistributedProfileCoverageTests(unittest.TestCase):
    def test_default_configuration_freezes_coverage_and_pinn_budgets(self):
        config = DistributedProfileCoverageConfig()
        self.assertEqual(config.trial_count, 20)
        self.assertEqual(config.pinn_trial_count, 10)
        self.assertEqual(config.inverse_pinn_epochs, 400)
        self.assertEqual(config.representative_profile_points, 5)
        self.assertGreater(config.shrinkage_weight, 0.0)

    def test_seed_blocks_are_disjoint(self):
        first = distributed_profile_coverage_seeds(100, 0, 3)
        second = distributed_profile_coverage_seeds(100, 1, 3)
        self.assertEqual(first, DistributedProfileCoverageSeeds(100, (101, 102, 103)))
        self.assertEqual(second, DistributedProfileCoverageSeeds(104, (105, 106, 107)))
        self.assertFalse(
            set((first.neural, *first.observations))
            & set((second.neural, *second.observations))
        )

    def test_configuration_rejects_invalid_budgets_and_profiles(self):
        for values in (
            {"trial_count": 0},
            {"pinn_trial_count": 21},
            {"first_seed": -1},
            {"representative_profile_points": 2},
            {"shrinkage_weight": 0.0},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    DistributedProfileCoverageConfig(**values)
        with self.assertRaises(ValueError):
            DistributedProfileLikelihoodConfig(profile_points=2)

    def test_summary_separates_interval_estimators_from_pinn_point_estimates(self):
        trial = DistributedProfileCoverageTrial(
            trial_index=0,
            seeds=DistributedProfileCoverageSeeds(100, (101, 102, 103)),
            estimators=(
                _estimator("conventional_unregularized"),
                _estimator("pinn_unregularized", intervals=False),
            ),
        )
        summaries = summarize_distributed_profile_coverage((trial,))
        conventional = next(
            item for item in summaries if item.name == "conventional_unregularized"
        )
        pinn = next(item for item in summaries if item.name == "pinn_unregularized")
        self.assertEqual(conventional.interval_trial_count, 1)
        self.assertAlmostEqual(conventional.coefficient_coverage_68, 2.0 / 3.0)
        self.assertEqual(conventional.simultaneous_coverage_95, 1.0)
        self.assertEqual(pinn.interval_trial_count, 0)
        self.assertTrue(math.isnan(pinn.coefficient_coverage_95))


if __name__ == "__main__":
    unittest.main()
