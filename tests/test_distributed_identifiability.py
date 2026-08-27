import math
import unittest

from thermotwin.inference.distributed_identifiability import (
    DistributedIdentifiabilityConfig,
    DistributedPropertyCoefficient,
    analyze_distributed_identifiability,
)
from thermotwin.observations.distributed import DistributedObservationChannels
from thermotwin.simulation.distributed import (
    distributed_identifiability_experiments,
)


class DistributedIdentifiabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = distributed_identifiability_experiments()

    def test_one_parameter_has_rank_one(self):
        result = analyze_distributed_identifiability(
            self.experiments[:2],
            (DistributedPropertyCoefficient("electrical_resistivity", 1),),
            DistributedIdentifiabilityConfig(observation_interval=0.2),
        )
        self.assertEqual(result.effective_rank, 1)
        self.assertEqual(len(result.singular_values), 1)
        self.assertGreater(result.singular_values[0], 0.0)
        self.assertTrue(math.isfinite(result.condition_number))

    def test_three_seebeck_knots_are_analyzed_without_truth_grid_locking(self):
        parameters = tuple(
            DistributedPropertyCoefficient("seebeck_coefficient", index)
            for index in range(3)
        )
        result = analyze_distributed_identifiability(
            self.experiments,
            parameters,
            DistributedIdentifiabilityConfig(observation_interval=0.2),
        )
        self.assertEqual(result.observation_count, 60)
        self.assertEqual(len(result.jacobian), result.observation_count)
        self.assertEqual(len(result.information_matrix), 3)
        self.assertGreaterEqual(result.effective_rank, 1)
        self.assertGreater(result.temperature_range[1], result.temperature_range[0])

    def test_heat_rate_channels_expand_the_observation_vector(self):
        base = analyze_distributed_identifiability(
            self.experiments[:1],
            (DistributedPropertyCoefficient("thermal_conductivity", 1),),
            DistributedIdentifiabilityConfig(observation_interval=0.4),
        )
        expanded = analyze_distributed_identifiability(
            self.experiments[:1],
            (DistributedPropertyCoefficient("thermal_conductivity", 1),),
            DistributedIdentifiabilityConfig(
                observation_interval=0.4,
                channels=DistributedObservationChannels(
                    cold_side_heat=True,
                    hot_side_heat=True,
                ),
            ),
        )
        self.assertGreater(expanded.observation_count, base.observation_count)

    def test_out_of_range_property_coefficient_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            analyze_distributed_identifiability(
                self.experiments[:1],
                (DistributedPropertyCoefficient("thermal_conductivity", 9),),
            )


if __name__ == "__main__":
    unittest.main()
