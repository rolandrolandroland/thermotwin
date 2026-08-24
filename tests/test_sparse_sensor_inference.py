import unittest

from thermotwin.sparse_sensor_inference import (
    ACCESSIBLE_SENSOR_NAMES,
    SparseSensorInferenceConfig,
    build_sparse_sensor_problem,
    fit_sparse_sensor_parameters,
    run_sparse_sensor_inference_experiment,
)


class SparseSensorInferenceTests(unittest.TestCase):
    def test_problem_exposes_only_exchanger_sensors_and_has_missing_records(self):
        problem = build_sparse_sensor_problem()

        self.assertEqual(
            tuple(sensor.name for sensor in problem.observations.sensors),
            ACCESSIBLE_SENSOR_NAMES,
        )
        cold_count = len(
            problem.observations.observations_for(ACCESSIBLE_SENSOR_NAMES[0])
        )
        hot_count = len(
            problem.observations.observations_for(ACCESSIBLE_SENSOR_NAMES[1])
        )
        self.assertLess(cold_count, hot_count)
        self.assertEqual(hot_count, 81)

    def test_joint_fit_recovers_parameters_and_transfers(self):
        config = SparseSensorInferenceConfig(
            grid_points_per_axis=7,
            refinement_count=3,
        )
        result = run_sparse_sensor_inference_experiment(config)

        self.assertLess(
            abs(result.fit.inferred_cold_contact_resistance - 0.25),
            0.02,
        )
        self.assertLess(
            abs(result.fit.inferred_sensor_time_constant - 1.5),
            0.25,
        )
        self.assertLess(result.fit.observation_rmse, 0.03)
        self.assertLess(
            result.withheld_validation.accessible_sensor_rmse,
            0.02,
        )
        self.assertTrue(
            all(interval.contains_truth for interval in result.uncertainty.intervals)
        )
        self.assertGreater(
            abs(result.uncertainty.correlation[0][1]),
            0.01,
        )
        grid_evaluation_count = (
            config.grid_points_per_axis**2 * config.refinement_count
        )
        grid_best = min(
            item.mean_squared_error
            for item in result.fit.evaluations[:grid_evaluation_count]
        )
        self.assertLess(
            result.fit.observation_rmse**2,
            grid_best,
        )
        self.assertNotEqual(
            result.fit.inferred_cold_contact_resistance,
            config.true_cold_contact_resistance,
        )
        self.assertGreater(result.hidden_state_validation.cold_face_rmse, 0.0)

    def test_local_polish_recovers_truth_that_is_not_a_grid_node(self):
        config = SparseSensorInferenceConfig(
            true_cold_contact_resistance=0.273,
            true_sensor_time_constant=1.73,
            grid_points_per_axis=7,
            refinement_count=2,
            local_polish_iterations=16,
        )
        fit = fit_sparse_sensor_parameters(build_sparse_sensor_problem(config))

        self.assertLess(
            abs(fit.inferred_cold_contact_resistance - 0.273),
            0.02,
        )
        self.assertLess(
            abs(fit.inferred_sensor_time_constant - 1.73),
            0.30,
        )

    def test_configuration_rejects_zero_noise_for_interval_model(self):
        with self.assertRaises(ValueError):
            SparseSensorInferenceConfig(noise_standard_deviation=0.0)
        with self.assertRaises(ValueError):
            SparseSensorInferenceConfig(local_polish_iterations=0)


if __name__ == "__main__":
    unittest.main()
