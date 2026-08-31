import math
import unittest

from thermotwin.inference.distributed_identifiability import (
    DistributedIdentifiabilityAssessment,
    DistributedIdentifiabilityConfig,
    DistributedIdentifiabilityGateConfig,
    DistributedIdentifiabilityResult,
    DistributedPropertyCoefficient,
    analyze_distributed_identifiability,
    assess_distributed_identifiability,
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

    def test_practical_gate_distinguishes_zero_weak_and_supported_spectra(self):
        def result(singular_values):
            return DistributedIdentifiabilityResult(
                parameter_names=("a", "b", "c"),
                observation_count=3,
                jacobian=((0.0, 0.0, 0.0),) * 3,
                information_matrix=((0.0, 0.0, 0.0),) * 3,
                singular_values=singular_values,
                effective_rank=0,
                condition_number=math.inf,
                column_norms=(0.0, 0.0, 0.0),
                temperature_range=(290.0, 310.0),
            )

        zero = assess_distributed_identifiability(result((0.0, 0.0, 0.0)))
        weak = assess_distributed_identifiability(result((100.0, 4.0, 0.5)))
        supported = assess_distributed_identifiability(result((100.0, 10.0, 4.0)))
        self.assertEqual(zero.status, "structurally_non_identifiable")
        self.assertEqual(weak.status, "practically_non_identifiable")
        self.assertEqual(weak.supported_rank, 2)
        self.assertEqual(supported.status, "supported")
        self.assertEqual(supported.supported_rank, 3)
        self.assertIsInstance(supported, DistributedIdentifiabilityAssessment)

    def test_practical_gate_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            DistributedIdentifiabilityGateConfig(maximum_log_displacement=0.0)


if __name__ == "__main__":
    unittest.main()
