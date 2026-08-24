import unittest

from thermotwin import (
    FourNodeContactTemperatureTrajectory,
    IdealTemperatureSensor,
    IdealVirtualTestStand,
    ObservationDataset,
    PiecewiseConstantCurrent,
    TemperatureSensorLocation,
    TemperatureObservation,
    constant_current_contact_reference_experiment,
    ideal_four_sensor_test_stand,
    observe_contact_trajectory,
    regular_measurement_times,
    run_four_node_contact_experiment,
    run_ideal_contact_reference_test_stand,
)


class VirtualTestStandTests(unittest.TestCase):
    def setUp(self):
        self.linear_trajectory = FourNodeContactTemperatureTrajectory(
            time=(0.0, 2.0),
            cold_face=(300.0, 290.0),
            hot_face=(300.0, 310.0),
            cold_exchanger=(300.0, 296.0),
            hot_exchanger=(300.0, 306.0),
        )

    def test_sensor_name_is_clean_and_location_is_typed(self):
        sensor = IdealTemperatureSensor(" cold ", "cold_face")

        self.assertEqual(sensor.name, "cold")
        self.assertIs(sensor.location, TemperatureSensorLocation.COLD_FACE)
        with self.assertRaises(ValueError):
            IdealTemperatureSensor("  ", "cold_face")
        with self.assertRaises(ValueError):
            IdealTemperatureSensor("sensor", "not_a_node")

    def test_test_stand_requires_sensors_unique_names_and_valid_interval(self):
        sensor = IdealTemperatureSensor(
            "sensor",
            TemperatureSensorLocation.COLD_FACE,
        )
        for sensors, interval in (
            ((), 1.0),
            ((sensor, sensor), 1.0),
            ((sensor,), 0.0),
            ((sensor,), -1.0),
            ((sensor,), float("inf")),
            ((sensor,), float("nan")),
        ):
            with self.subTest(sensors=sensors, interval=interval):
                with self.assertRaises(ValueError):
                    IdealVirtualTestStand(sensors, interval)

    def test_multiple_named_sensors_may_share_one_location(self):
        stand = IdealVirtualTestStand(
            sensors=(
                IdealTemperatureSensor("cold_a", "cold_face"),
                IdealTemperatureSensor("cold_b", "cold_face"),
            ),
            sampling_interval=1.0,
        )
        dataset = observe_contact_trajectory(
            self.linear_trajectory,
            current=1.0,
            test_stand=stand,
        )

        self.assertEqual(len(dataset.observations), 6)
        for first, second in zip(
            dataset.observations[::2],
            dataset.observations[1::2],
        ):
            self.assertEqual(first.temperature, second.temperature)

    def test_regular_times_include_zero_and_exact_final_time(self):
        self.assertEqual(
            regular_measurement_times(2.5, 1.0),
            (0.0, 1.0, 2.0, 2.5),
        )
        self.assertEqual(regular_measurement_times(0.0, 1.0), (0.0,))
        self.assertEqual(len(regular_measurement_times(60.0, 1.0)), 61)

    def test_regular_times_reject_invalid_inputs(self):
        invalid_cases = (
            (-1.0, 1.0),
            (float("nan"), 1.0),
            (1.0, 0.0),
            (1.0, -1.0),
            (1.0, float("inf")),
        )
        for duration, interval in invalid_cases:
            with self.subTest(duration=duration, interval=interval):
                with self.assertRaises(ValueError):
                    regular_measurement_times(duration, interval)

    def test_linear_interpolation_and_location_mapping(self):
        dataset = observe_contact_trajectory(
            self.linear_trajectory,
            current=1.0,
            test_stand=ideal_four_sensor_test_stand(
                sampling_interval=1.0
            ),
        )

        middle = dataset.observations[4:8]
        self.assertEqual(
            tuple(observation.location for observation in middle),
            tuple(sensor.location for sensor in dataset.sensors),
        )
        self.assertEqual(
            tuple(observation.temperature for observation in middle),
            (295.0, 305.0, 298.0, 303.0),
        )

    def test_current_is_right_continuous_at_measurement_switch(self):
        current = PiecewiseConstantCurrent.step(
            transition_time=1.0,
            before_current=0.0,
            after_current=2.0,
        )
        dataset = observe_contact_trajectory(
            self.linear_trajectory,
            current=current,
            test_stand=IdealVirtualTestStand(
                sensors=(IdealTemperatureSensor("cold", "cold_face"),),
                sampling_interval=1.0,
            ),
        )

        self.assertEqual(
            tuple(observation.current for observation in dataset.observations),
            (0.0, 2.0, 2.0),
        )

    def test_reference_dataset_has_expected_schema_counts_and_initial_values(
        self,
    ):
        dataset = run_ideal_contact_reference_test_stand()

        self.assertEqual(len(dataset.measurement_times), 61)
        self.assertEqual(len(dataset.observations), 244)
        self.assertEqual(dataset.sampling_interval, 1.0)
        self.assertEqual(dataset.time_unit, "s")
        self.assertEqual(dataset.temperature_unit, "K")
        self.assertEqual(dataset.current_unit, "A")
        self.assertEqual(
            tuple(observation.sensor_name for observation in dataset.observations[:4]),
            (
                "cold_face_sensor",
                "hot_face_sensor",
                "cold_exchanger_sensor",
                "hot_exchanger_sensor",
            ),
        )
        self.assertTrue(
            all(
                observation.temperature == 300.0
                for observation in dataset.observations[:4]
            )
        )
        self.assertTrue(
            all(
                observation.current == 1.0
                for observation in dataset.observations
            )
        )

    def test_reference_final_observations_match_hidden_truth(self):
        experiment = constant_current_contact_reference_experiment()
        truth = run_four_node_contact_experiment(experiment).trajectory
        dataset = run_ideal_contact_reference_test_stand()
        final_observations = dataset.observations[-4:]

        self.assertEqual(
            tuple(observation.temperature for observation in final_observations),
            (
                truth.cold_face[-1],
                truth.hot_face[-1],
                truth.cold_exchanger[-1],
                truth.hot_exchanger[-1],
            ),
        )

    def test_dataset_filters_known_sensor_and_rejects_unknown_name(self):
        dataset = run_ideal_contact_reference_test_stand()
        cold_face = dataset.observations_for("cold_face_sensor")

        self.assertEqual(len(cold_face), 61)
        self.assertTrue(
            all(
                observation.location is TemperatureSensorLocation.COLD_FACE
                for observation in cold_face
            )
        )
        self.assertEqual(
            tuple(observation.time for observation in cold_face),
            dataset.measurement_times,
        )
        with self.assertRaises(ValueError):
            dataset.observations_for("unknown")

    def test_dataset_does_not_expose_dense_truth_trajectory(self):
        dataset = run_ideal_contact_reference_test_stand()

        self.assertFalse(hasattr(dataset, "trajectory"))
        self.assertFalse(hasattr(dataset, "truth"))
        self.assertFalse(hasattr(dataset, "cold_face"))

    def test_dataset_rejects_unknown_mislabeled_duplicate_or_unsorted_records(
        self,
    ):
        sensor = IdealTemperatureSensor("cold", "cold_face")
        valid = TemperatureObservation(
            time=0.0,
            sensor_name="cold",
            location="cold_face",
            temperature=300.0,
            current=1.0,
        )
        invalid_observation_sets = (
            (
                TemperatureObservation(
                    0.0,
                    "unknown",
                    "cold_face",
                    300.0,
                    1.0,
                ),
            ),
            (
                TemperatureObservation(
                    0.0,
                    "cold",
                    "hot_face",
                    300.0,
                    1.0,
                ),
            ),
            (valid, valid),
            (
                TemperatureObservation(
                    1.0,
                    "cold",
                    "cold_face",
                    299.0,
                    1.0,
                ),
                valid,
            ),
        )

        for observations in invalid_observation_sets:
            with self.subTest(observations=observations):
                with self.assertRaises(ValueError):
                    ObservationDataset(
                        observations=observations,
                        sensors=(sensor,),
                        sampling_interval=1.0,
                    )

    def test_malformed_trajectories_are_rejected(self):
        malformed = (
            FourNodeContactTemperatureTrajectory((), (), (), (), ()),
            FourNodeContactTemperatureTrajectory(
                (0.0, 1.0),
                (300.0,),
                (300.0, 301.0),
                (300.0, 300.0),
                (300.0, 300.0),
            ),
            FourNodeContactTemperatureTrajectory(
                (0.0, 0.0),
                (300.0, 300.0),
                (300.0, 300.0),
                (300.0, 300.0),
                (300.0, 300.0),
            ),
            FourNodeContactTemperatureTrajectory(
                (1.0, 2.0),
                (300.0, 300.0),
                (300.0, 300.0),
                (300.0, 300.0),
                (300.0, 300.0),
            ),
            FourNodeContactTemperatureTrajectory(
                (0.0, 1.0),
                (300.0, float("nan")),
                (300.0, 300.0),
                (300.0, 300.0),
                (300.0, 300.0),
            ),
        )
        stand = ideal_four_sensor_test_stand()

        for trajectory in malformed:
            with self.subTest(trajectory=trajectory):
                with self.assertRaises(ValueError):
                    observe_contact_trajectory(
                        trajectory,
                        current=1.0,
                        test_stand=stand,
                    )


if __name__ == "__main__":
    unittest.main()
