from dataclasses import replace
import math
import unittest

from thermotwin.contact_resistance_inference import (
    ContactResistanceDatasetSplit,
    ContactResistanceRegimeDataset,
    ContactResistanceSearchConfig,
    reference_contact_resistance_dataset_split,
)
from thermotwin.contact_resistance_noise_study import (
    ContactResistanceNoiseSeeds,
    ContactResistanceNoiseStudyConfig,
    contact_resistance_noise_seeds,
    format_contact_resistance_noise_study_report,
    noisy_contact_resistance_dataset_split,
    run_contact_resistance_noise_study,
    run_contact_resistance_noise_trial,
    summarize_contact_resistance_noise_trials,
)


class ContactResistanceNoiseStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ideal = reference_contact_resistance_dataset_split()
        cls.small_config = ContactResistanceNoiseStudyConfig(trial_count=5)
        cls.small_study = run_contact_resistance_noise_study(cls.small_config)

    def test_default_config_freezes_noise_trials_seeds_and_search(self):
        config = ContactResistanceNoiseStudyConfig()

        self.assertEqual(config.noise_standard_deviation, 0.05)
        self.assertEqual(config.trial_count, 100)
        self.assertEqual(config.first_seed, 2026)
        self.assertEqual(config.search.lower_bound, 0.05)
        self.assertEqual(config.search.upper_bound, 1.0)
        self.assertEqual(config.search.resistance_tolerance, 1e-6)
        self.assertEqual(config.search.max_iterations, 64)

    def test_config_rejects_invalid_values(self):
        invalid_configs = (
            {"noise_standard_deviation": -1.0},
            {"noise_standard_deviation": float("inf")},
            {"noise_standard_deviation": float("nan")},
            {"noise_standard_deviation": "invalid"},
            {"noise_standard_deviation": True},
            {"trial_count": 0},
            {"trial_count": 1.5},
            {"trial_count": True},
            {"first_seed": 1.5},
            {"first_seed": True},
            {"first_seed": -1},
            {"search": "invalid"},
        )
        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ContactResistanceNoiseStudyConfig(**values)

    def test_trial_seed_mapping_is_reproducible_and_nonoverlapping(self):
        self.assertEqual(
            contact_resistance_noise_seeds(2026, 0),
            ContactResistanceNoiseSeeds(2026, 2027, 2028),
        )
        self.assertEqual(
            contact_resistance_noise_seeds(2026, 1),
            ContactResistanceNoiseSeeds(2029, 2030, 2031),
        )
        seeds = tuple(
            seed
            for trial_index in range(10)
            for seed in contact_resistance_noise_seeds(
                2026,
                trial_index,
            )
        )
        self.assertEqual(len(seeds), len(set(seeds)))

        for first_seed, trial_index in (
            (1.5, 0),
            (True, 0),
            (-1, 0),
            (1, -1),
            (1, 1.5),
        ):
            with self.subTest(
                first_seed=first_seed,
                trial_index=trial_index,
            ):
                with self.assertRaises(ValueError):
                    contact_resistance_noise_seeds(first_seed, trial_index)

    def test_multiple_regimes_get_independent_noise_across_split_boundaries(self):
        first_training = self.ideal.train[0]
        second_training = ContactResistanceRegimeDataset(
            regime=replace(
                first_training.regime,
                name="second_training_regime",
            ),
            observations=first_training.observations,
        )
        multiple_regimes = ContactResistanceDatasetSplit(
            train=(first_training, second_training),
            validation=self.ideal.validation,
            test=self.ideal.test,
        )
        noisy = noisy_contact_resistance_dataset_split(
            multiple_regimes,
            noise_standard_deviation=0.05,
            seeds=contact_resistance_noise_seeds(2026, 0),
        )

        second_training_noise = tuple(
            observed.temperature - ideal.temperature
            for observed, ideal in zip(
                noisy.train[1].observations.observations,
                second_training.observations.observations,
            )
        )
        validation_noise = tuple(
            observed.temperature - ideal.temperature
            for observed, ideal in zip(
                noisy.validation[0].observations.observations,
                self.ideal.validation[0].observations.observations,
            )
        )

        self.assertNotEqual(second_training_noise, validation_noise)

    def test_zero_noise_split_is_exact_ideal_limiting_case(self):
        noisy = noisy_contact_resistance_dataset_split(
            self.ideal,
            noise_standard_deviation=0.0,
            seeds=ContactResistanceNoiseSeeds(1, 2, 3),
        )

        self.assertEqual(noisy, self.ideal)
        self.assertIsNot(noisy, self.ideal)

    def test_same_seeds_repeat_and_different_seeds_change_noise(self):
        first = noisy_contact_resistance_dataset_split(
            self.ideal,
            noise_standard_deviation=0.05,
            seeds=ContactResistanceNoiseSeeds(10, 11, 12),
        )
        repeated = noisy_contact_resistance_dataset_split(
            self.ideal,
            noise_standard_deviation=0.05,
            seeds=ContactResistanceNoiseSeeds(10, 11, 12),
        )
        changed = noisy_contact_resistance_dataset_split(
            self.ideal,
            noise_standard_deviation=0.05,
            seeds=ContactResistanceNoiseSeeds(13, 14, 15),
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)

    def test_noise_preserves_regime_schema_and_changes_only_temperature(self):
        noisy = noisy_contact_resistance_dataset_split(
            self.ideal,
            noise_standard_deviation=0.05,
            seeds=ContactResistanceNoiseSeeds(20, 21, 22),
        )

        for ideal_group, noisy_group in (
            (self.ideal.train, noisy.train),
            (self.ideal.validation, noisy.validation),
            (self.ideal.test, noisy.test),
        ):
            self.assertEqual(len(ideal_group), len(noisy_group))
            for ideal_dataset, noisy_dataset in zip(ideal_group, noisy_group):
                self.assertEqual(ideal_dataset.regime, noisy_dataset.regime)
                self.assertEqual(
                    ideal_dataset.observations.sensors,
                    noisy_dataset.observations.sensors,
                )
                self.assertEqual(
                    ideal_dataset.observations.measurement_times,
                    noisy_dataset.observations.measurement_times,
                )
                changed_temperatures = 0
                for ideal, observed in zip(
                    ideal_dataset.observations.observations,
                    noisy_dataset.observations.observations,
                ):
                    self.assertEqual(ideal.time, observed.time)
                    self.assertEqual(ideal.sensor_name, observed.sensor_name)
                    self.assertIs(ideal.location, observed.location)
                    self.assertEqual(ideal.current, observed.current)
                    changed_temperatures += (
                        ideal.temperature != observed.temperature
                    )
                self.assertGreater(changed_temperatures, 0)

    def test_zero_noise_trial_recovers_noise_free_limit(self):
        config = ContactResistanceNoiseStudyConfig(
            noise_standard_deviation=0.0,
            trial_count=1,
        )
        trial = run_contact_resistance_noise_trial(
            0,
            self.ideal,
            config,
        )

        self.assertLess(trial.absolute_parameter_error, 1e-6)
        self.assertFalse(trial.reached_search_bound)
        self.assertEqual(trial.search_evaluations, 32)
        self.assertAlmostEqual(
            trial.training_observation_rmse,
            trial.training_truth_rmse,
        )
        self.assertAlmostEqual(
            trial.validation_observation_rmse,
            trial.validation_truth_rmse,
        )
        self.assertAlmostEqual(
            trial.test_observation_rmse,
            trial.test_truth_rmse,
        )

    def test_first_noisy_trial_is_frozen_regression(self):
        trial = self.small_study.trials[0]

        self.assertEqual(trial.trial_index, 0)
        self.assertEqual(
            trial.seeds,
            ContactResistanceNoiseSeeds(2026, 2027, 2028),
        )
        self.assertAlmostEqual(
            trial.inferred_cold_contact_resistance,
            0.25272690702318823,
        )
        self.assertAlmostEqual(trial.training_observation_rmse, 0.0524166719301579)
        self.assertAlmostEqual(trial.validation_observation_rmse, 0.04837759956982414)
        self.assertAlmostEqual(trial.test_observation_rmse, 0.048076145974745084)
        self.assertLess(
            trial.training_truth_rmse,
            trial.training_observation_rmse,
        )
        self.assertFalse(trial.reached_search_bound)

    def test_small_study_is_reproducible(self):
        repeated = run_contact_resistance_noise_study(self.small_config)

        self.assertEqual(repeated, self.small_study)
        self.assertEqual(len(repeated.trials), 5)

    def test_small_summary_matches_empirical_trial_statistics(self):
        summary = self.small_study.summary
        estimates = tuple(
            trial.inferred_cold_contact_resistance
            for trial in self.small_study.trials
        )
        errors = tuple(estimate - 0.25 for estimate in estimates)

        self.assertAlmostEqual(
            summary.mean_inferred_resistance,
            sum(estimates) / len(estimates),
        )
        self.assertAlmostEqual(
            summary.mean_parameter_bias,
            sum(errors) / len(errors),
        )
        self.assertAlmostEqual(
            summary.parameter_rmse,
            math.sqrt(sum(error * error for error in errors) / len(errors)),
        )
        self.assertLessEqual(
            summary.minimum_inferred_resistance,
            summary.percentile_05_resistance,
        )
        self.assertLessEqual(
            summary.percentile_05_resistance,
            summary.median_inferred_resistance,
        )
        self.assertLessEqual(
            summary.median_inferred_resistance,
            summary.percentile_95_resistance,
        )
        self.assertLessEqual(
            summary.percentile_95_resistance,
            summary.maximum_inferred_resistance,
        )
        self.assertGreater(summary.parameter_standard_deviation, 0.0)
        self.assertEqual(summary.search_bound_hits, 0)

    def test_one_trial_summary_has_zero_empirical_standard_deviation(self):
        trial = self.small_study.trials[0]
        summary = summarize_contact_resistance_noise_trials(
            (trial,),
            noise_standard_deviation=0.05,
        )

        self.assertEqual(summary.trial_count, 1)
        self.assertEqual(summary.parameter_standard_deviation, 0.0)
        self.assertEqual(
            summary.minimum_inferred_resistance,
            summary.maximum_inferred_resistance,
        )

        with self.assertRaises(ValueError):
            summarize_contact_resistance_noise_trials(
                (),
                noise_standard_deviation=0.05,
            )

    def test_different_first_seed_changes_trial_distribution(self):
        changed = run_contact_resistance_noise_study(
            ContactResistanceNoiseStudyConfig(
                trial_count=2,
                first_seed=5000,
            )
        )

        self.assertNotEqual(
            changed.summary.mean_inferred_resistance,
            self.small_study.summary.mean_inferred_resistance,
        )

    def test_text_report_contains_uncertainty_and_error_metrics(self):
        report = format_contact_resistance_noise_study_report(
            self.small_study
        )

        self.assertIn("trials: 5", report)
        self.assertIn("true cold contact resistance: 0.250000000 K/W", report)
        self.assertIn("temperature noise standard deviation: 0.050000 K", report)
        self.assertIn("sample parameter standard deviation:", report)
        self.assertIn("empirical 5th--95th percentile interval:", report)
        self.assertIn("search bound hits: 0", report)
        self.assertIn("mean fitted-pair observation RMSE", report)
        self.assertIn("mean fitted-pair truth RMSE", report)


if __name__ == "__main__":
    unittest.main()
