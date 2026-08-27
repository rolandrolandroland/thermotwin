import math
import unittest

from thermotwin.inference.distributed_experiment_selection import (
    DistributedExperimentSelectionConfig,
    linearized_distributed_uncertainty,
    select_distributed_experiment,
)
from thermotwin.inference.distributed_identifiability import (
    DistributedIdentifiabilityConfig,
    DistributedPropertyCoefficient,
)


class DistributedExperimentSelectionTests(unittest.TestCase):
    def test_small_candidate_set_selects_a_feasible_experiment(self):
        parameter = DistributedPropertyCoefficient(
            "electrical_resistivity", 1
        )
        result = select_distributed_experiment(
            (parameter,),
            DistributedExperimentSelectionConfig(
                current_amplitudes=(0.3, 0.6),
                pulse_durations=(0.1,),
                reservoir_temperature_lifts=(0.0,),
                duration=0.3,
                pulse_start_time=0.05,
                cell_count=4,
                time_step=0.001,
                identifiability=DistributedIdentifiabilityConfig(
                    observation_interval=0.1
                ),
            ),
        )
        self.assertEqual(len(result.candidates), 2)
        self.assertTrue(result.selected.feasible)
        self.assertIn(result.selected, result.candidates)
        self.assertGreater(result.selected.information_gain_nats, 0.0)

    def test_linearized_uncertainty_has_unit_diagonal_correlations(self):
        parameters = (
            DistributedPropertyCoefficient("thermal_conductivity", 0),
            DistributedPropertyCoefficient("thermal_conductivity", 1),
        )
        uncertainty = linearized_distributed_uncertainty(
            parameters,
            ((100.0, 10.0), (10.0, 50.0)),
            prior_log_standard_deviation=0.2,
        )
        self.assertEqual(len(uncertainty.log_standard_errors), 2)
        self.assertAlmostEqual(uncertainty.correlation_matrix[0][0], 1.0)
        self.assertAlmostEqual(uncertainty.correlation_matrix[1][1], 1.0)
        self.assertTrue(
            all(
                lower < 1.0 < upper
                for lower, upper in uncertainty.multiplier_95_intervals
            )
        )
        self.assertTrue(
            all(math.isfinite(value) for value in uncertainty.log_standard_errors)
        )


if __name__ == "__main__":
    unittest.main()
