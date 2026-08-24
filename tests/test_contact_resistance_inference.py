from dataclasses import replace
import unittest

from thermotwin import PiecewiseConstantCurrent
from thermotwin.contact_resistance_inference import (
    ContactResistanceDatasetSplit,
    ContactResistanceRegime,
    ContactResistanceRegimeDataset,
    ContactResistanceSearchConfig,
    contact_resistance_experiment,
    contact_resistance_training_loss,
    fit_cold_contact_resistance,
    format_contact_resistance_inference_report,
    reference_contact_resistance_dataset_split,
    reference_contact_resistance_regimes,
    run_contact_resistance_inference_experiment,
    simulate_contact_resistance_observations,
)


class ContactResistanceInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_contact_resistance_inference_experiment()
        cls.datasets = cls.result.datasets

    def test_reference_regimes_freeze_whole_experiment_splits(self):
        training, validation, test = reference_contact_resistance_regimes()

        self.assertEqual(
            (training.name, training.split),
            ("unipolar_training_pulse", "train"),
        )
        self.assertEqual(
            training.current.transition_times,
            (5.0, 20.0),
        )
        self.assertEqual(training.current.values, (0.0, 1.0, 0.0))
        self.assertEqual(
            (validation.name, validation.split),
            ("lower_amplitude_validation_pulse", "validation"),
        )
        self.assertEqual(
            validation.current.transition_times,
            (10.0, 30.0),
        )
        self.assertEqual(validation.current.values, (0.0, 0.6, 0.0))
        self.assertEqual(
            (test.name, test.split),
            ("bipolar_test_pulse", "test"),
        )
        self.assertEqual(
            test.current.transition_times,
            (5.0, 20.0, 35.0, 50.0),
        )
        self.assertEqual(test.current.values, (0.0, 1.0, 0.0, -1.0, 0.0))

    def test_regime_rejects_invalid_name_split_or_current(self):
        valid_current = PiecewiseConstantCurrent.constant(0.0)
        invalid_cases = (
            {"name": "", "split": "train", "current": valid_current},
            {"name": "run", "split": "other", "current": valid_current},
            {"name": "run", "split": None, "current": valid_current},
            {"name": "run", "split": "train", "current": 1.0},
        )
        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ContactResistanceRegime(**values)

    def test_candidate_experiment_changes_only_schedule_and_cold_contact(self):
        regime = reference_contact_resistance_regimes()[0]
        experiment = contact_resistance_experiment(
            regime,
            cold_contact_resistance=0.5,
        )

        self.assertIs(experiment.current, regime.current)
        self.assertEqual(
            experiment.thermal_parameters.cold_contact_resistance,
            0.5,
        )
        self.assertEqual(
            experiment.thermal_parameters.hot_contact_resistance,
            0.25,
        )
        self.assertEqual(
            experiment.thermoelectric_parameters.thermal_conductance,
            0.5,
        )
        self.assertEqual(experiment.duration, 60.0)
        self.assertEqual(experiment.time_step, 0.1)
        self.assertEqual(experiment.cold_reservoir_temperature, 300.0)
        self.assertEqual(experiment.hot_reservoir_temperature, 300.0)

    def test_candidate_resistance_must_be_finite_and_positive(self):
        regime = reference_contact_resistance_regimes()[0]

        for value in (0.0, -1.0, float("inf"), float("nan"), "invalid"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    contact_resistance_experiment(
                        regime,
                        cold_contact_resistance=value,
                    )

    def test_reference_split_has_complete_ideal_regime_datasets(self):
        self.assertEqual(len(self.datasets.train), 1)
        self.assertEqual(len(self.datasets.validation), 1)
        self.assertEqual(len(self.datasets.test), 1)

        for dataset in (
            self.datasets.train
            + self.datasets.validation
            + self.datasets.test
        ):
            self.assertEqual(len(dataset.observations.measurement_times), 61)
            self.assertEqual(len(dataset.observations.observations), 244)
            self.assertEqual(dataset.observations.sampling_interval, 1.0)
            self.assertFalse(hasattr(dataset, "true_contact_resistance"))
            self.assertFalse(hasattr(dataset.observations, "truth"))

    def test_dataset_split_rejects_empty_mislabeled_or_duplicate_regimes(self):
        train = self.datasets.train[0]
        validation = self.datasets.validation[0]
        test = self.datasets.test[0]

        with self.assertRaises(ValueError):
            ContactResistanceRegimeDataset("invalid", train.observations)
        with self.assertRaises(ValueError):
            ContactResistanceRegimeDataset(train.regime, "invalid")

        mislabeled = replace(validation, regime=replace(validation.regime, split="train"))
        duplicate_test = replace(
            test,
            regime=replace(test.regime, name=train.regime.name),
        )

        invalid_splits = (
            {"train": (), "validation": (validation,), "test": (test,)},
            {"train": None, "validation": (validation,), "test": (test,)},
            {"train": ("invalid",), "validation": (validation,), "test": (test,)},
            {"train": (train,), "validation": (mislabeled,), "test": (test,)},
            {
                "train": (train,),
                "validation": (validation,),
                "test": (duplicate_test,),
            },
        )
        for values in invalid_splits:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ContactResistanceDatasetSplit(**values)

    def test_training_switch_is_right_continuous_and_temperature_is_continuous(self):
        observations = self.datasets.train[0].observations
        cold_face = {
            observation.time: observation
            for observation in observations.observations_for(
                "cold_face_sensor"
            )
        }
        cold_exchanger = {
            observation.time: observation
            for observation in observations.observations_for(
                "cold_exchanger_sensor"
            )
        }

        self.assertEqual(cold_face[5.0].current, 1.0)
        self.assertEqual(cold_face[5.0].temperature, 300.0)
        self.assertEqual(cold_exchanger[5.0].temperature, 300.0)
        self.assertLess(cold_face[6.0].temperature, 300.0)
        self.assertLess(cold_exchanger[6.0].temperature, 300.0)
        self.assertEqual(cold_face[20.0].current, 0.0)
        self.assertGreater(
            cold_exchanger[20.0].temperature,
            cold_face[20.0].temperature,
        )

    def test_frozen_training_transient_has_expected_peak_contact_gap(self):
        observations = self.datasets.train[0].observations
        cold_face = observations.observations_for("cold_face_sensor")
        cold_exchanger = observations.observations_for(
            "cold_exchanger_sensor"
        )
        gaps = tuple(
            (
                face.time,
                exchanger.temperature - face.temperature,
            )
            for face, exchanger in zip(cold_face, cold_exchanger)
        )
        peak_time, peak_gap = max(gaps, key=lambda item: item[1])

        self.assertEqual(peak_time, 20.0)
        self.assertAlmostEqual(peak_gap, 1.541668759, places=9)
        self.assertAlmostEqual(cold_face[20].temperature, 297.448415902)
        self.assertAlmostEqual(cold_exchanger[20].temperature, 298.990084661)

    def test_larger_resistance_increases_driven_contact_gap(self):
        regime = self.datasets.train[0].regime
        gaps = []
        for resistance in (0.10, 0.25, 0.50):
            predicted = simulate_contact_resistance_observations(
                regime,
                cold_contact_resistance=resistance,
            )
            cold_face = predicted.observations_for("cold_face_sensor")[20]
            cold_exchanger = predicted.observations_for(
                "cold_exchanger_sensor"
            )[20]
            gaps.append(cold_exchanger.temperature - cold_face.temperature)

        self.assertLess(gaps[0], gaps[1])
        self.assertLess(gaps[1], gaps[2])

    def test_training_loss_has_exact_synthetic_minimum_at_truth(self):
        low_loss = contact_resistance_training_loss(0.10, self.datasets.train)
        true_loss = contact_resistance_training_loss(0.25, self.datasets.train)
        high_loss = contact_resistance_training_loss(0.50, self.datasets.train)

        self.assertEqual(true_loss, 0.0)
        self.assertGreater(low_loss, true_loss)
        self.assertGreater(high_loss, true_loss)
        self.assertAlmostEqual(low_loss, 0.03757467722442)
        self.assertAlmostEqual(high_loss, 0.05104280388841)

    def test_hot_sensor_values_do_not_enter_the_training_objective(self):
        original = self.datasets.train[0]
        changed_observations = tuple(
            replace(observation, temperature=observation.temperature + 10.0)
            if observation.sensor_name.startswith("hot_")
            else observation
            for observation in original.observations.observations
        )
        changed_dataset = replace(
            original,
            observations=replace(
                original.observations,
                observations=changed_observations,
            ),
        )

        self.assertEqual(
            contact_resistance_training_loss(0.5, (original,)),
            contact_resistance_training_loss(0.5, (changed_dataset,)),
        )

    def test_search_config_rejects_invalid_bounds_and_stopping_values(self):
        invalid_configs = (
            {"lower_bound": 0.0},
            {"lower_bound": float("nan")},
            {"lower_bound": "invalid"},
            {"lower_bound": 1.0, "upper_bound": 0.5},
            {"resistance_tolerance": 0.0},
            {"max_iterations": 0},
            {"max_iterations": 1.5},
            {"max_iterations": True},
        )
        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ContactResistanceSearchConfig(**values)

    def test_fit_rejects_nontraining_regimes(self):
        with self.assertRaisesRegex(ValueError, "only training"):
            fit_cold_contact_resistance(self.datasets.validation)

    def test_fit_recovers_truth_and_records_bounded_search_history(self):
        fit = self.result.fit

        self.assertAlmostEqual(
            fit.inferred_cold_contact_resistance,
            0.25,
            places=7,
        )
        self.assertLess(fit.training_mean_squared_error, 1e-15)
        self.assertEqual(fit.iterations, 39)
        self.assertEqual(len(fit.evaluations), 42)
        self.assertTrue(
            all(
                0.05 <= item.cold_contact_resistance <= 1.0
                for item in fit.evaluations
            )
        )

    def test_unseen_regimes_validate_without_time_point_leakage(self):
        summary = self.result.summary

        self.assertLess(summary.relative_parameter_error_percent, 1e-5)
        self.assertEqual(summary.training_metrics[0].split, "train")
        self.assertEqual(summary.validation_metrics[0].split, "validation")
        self.assertEqual(summary.test_metrics[0].split, "test")
        for metrics in (
            summary.training_metrics
            + summary.validation_metrics
            + summary.test_metrics
        ):
            self.assertLess(metrics.fitted_pair_rmse, 1e-7)
            self.assertLess(metrics.all_sensor_rmse, 1e-7)

    def test_text_report_contains_parameter_and_regime_results(self):
        report = format_contact_resistance_inference_report(self.result)

        self.assertIn("true resistance: 0.250000000 K/W", report)
        self.assertIn("inferred resistance: 0.250000002 K/W", report)
        self.assertIn("train unipolar_training_pulse", report)
        self.assertIn("validation lower_amplitude_validation_pulse", report)
        self.assertIn("test bipolar_test_pulse", report)


if __name__ == "__main__":
    unittest.main()
