import math
import unittest

from thermotwin.studies.distributed_observation_identifiability import (
    DistributedObservationCaseResult,
    DistributedObservationFitResult,
    DistributedObservationIdentifiabilityConfig,
    distributed_observation_case_definitions,
    estimator_coefficient_spread,
    run_distributed_observation_identifiability_study,
)


class DistributedObservationIdentifiabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_distributed_observation_identifiability_study(
            DistributedObservationIdentifiabilityConfig(
                truth_node_count=9,
                truth_time_step=5.0e-4,
                inverse_pinn_epochs=1,
            ),
            fit_models=False,
        )

    def test_case_definitions_separate_current_and_sensor_information(self):
        cases = distributed_observation_case_definitions()
        self.assertEqual(
            tuple(item.name for item in cases),
            (
                "full_bidirectional",
                "zero_current_only",
                "positive_temperature_only",
                "positive_temperature_voltage",
            ),
        )
        self.assertEqual(cases[0].experiment_indices, (0, 1, 2))
        self.assertFalse(cases[2].channels.voltage)
        self.assertTrue(cases[3].channels.voltage)

    def test_frozen_gate_accepts_only_the_full_bidirectional_case(self):
        statuses = {
            case.definition.name: case.assessment.status
            for case in self.result.cases
        }
        self.assertEqual(statuses["full_bidirectional"], "supported")
        self.assertEqual(
            statuses["zero_current_only"], "structurally_non_identifiable"
        )
        self.assertEqual(
            statuses["positive_temperature_only"],
            "practically_non_identifiable",
        )
        self.assertEqual(
            statuses["positive_temperature_voltage"],
            "practically_non_identifiable",
        )

    def test_zero_current_has_exactly_zero_resistivity_sensitivity(self):
        zero = next(
            case
            for case in self.result.cases
            if case.definition.name == "zero_current_only"
        )
        self.assertEqual(zero.identifiability.singular_values, (0.0, 0.0, 0.0))
        self.assertEqual(zero.assessment.supported_rank, 0)
        self.assertIn("withheld", zero.fit_policy)
        self.assertEqual(zero.fits, ())

    def test_positive_voltage_supports_average_but_not_all_curve_directions(self):
        voltage = next(
            case
            for case in self.result.cases
            if case.definition.name == "positive_temperature_voltage"
        )
        self.assertEqual(voltage.assessment.supported_rank, 2)
        self.assertGreater(voltage.identifiability.singular_values[0], 100.0)
        self.assertLess(voltage.identifiability.singular_values[-1], 1.0)

    def test_configuration_requires_multiple_in_bounds_initial_curves(self):
        for values in (
            {"initial_log_multiplier_sets": ((0.0, 0.0, 0.0),)},
            {"initial_log_multiplier_sets": ((0.0, 0.0), (0.0, 0.0))},
            {"first_noise_seed": -1},
            {"truth_node_count": 4},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    DistributedObservationIdentifiabilityConfig(**values)

    def test_coefficient_spread_uses_only_successful_curves(self):
        fit = lambda start, values: DistributedObservationFitResult(
            estimator="conventional",
            start_index=start,
            initial_multipliers=(1.0, 1.0, 1.0),
            fitted_multipliers=values,
            normalized_observation_loss=1.0,
            property_relative_rmse=0.1,
            property_maximum_relative_error=0.2,
            holdout_internal_temperature_rmse=0.01,
            holdout_voltage_rmse=1.0e-5,
            permitted_by_prefit_gate=False,
        )
        template = self.result.cases[0]
        case = DistributedObservationCaseResult(
            definition=template.definition,
            identifiability=template.identifiability,
            assessment=template.assessment,
            fit_policy=template.fit_policy,
            fits=(fit(0, (0.8, 1.0, 1.2)), fit(1, (1.1, 0.9, 1.0))),
        )
        self.assertAlmostEqual(
            estimator_coefficient_spread(case, "conventional"), 0.3
        )
        self.assertTrue(math.isinf(estimator_coefficient_spread(case, "pinn")))


if __name__ == "__main__":
    unittest.main()
