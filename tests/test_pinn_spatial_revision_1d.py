"""Scientific checks for the additive 1-D design-revision protocol.

These tests use the nominal material and declared development designs.  They do
not read historical truth, generate named evaluation data, or train a model.
"""

from dataclasses import FrozenInstanceError, replace
import json
import math
import unittest

import numpy as np
import torch

from thermotwin.core.controls import current_at
from thermotwin.observations.distributed import (
    DistributedObservation,
    DistributedObservationSet,
)
from thermotwin.studies.pinn_spatial_revision_1d import (
    CONFIG,
    _jsonable,
    _seed,
    candidate_designs,
    contrast_information,
    design_experiments,
    noise_whitened_jacobian,
)


class Revision1DScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_observation_dataclass_serializes_without_losing_measurement_keys(self):
        dataset = DistributedObservationSet((
            DistributedObservation(0.0, "cold_face_temperature", 286.0),
            DistributedObservation(0.0, "hot_face_temperature", 290.0),
            DistributedObservation(0.04, "voltage", -0.00123),
        ))
        decoded = json.loads(json.dumps(_jsonable(dataset), allow_nan=False))
        reconstructed = DistributedObservationSet(tuple(
            DistributedObservation(**item) for item in decoded["observations"]))
        self.assertEqual(reconstructed, dataset)
        self.assertEqual(reconstructed.keys(), dataset.keys())
        self.assertEqual(reconstructed.values(), dataset.values())

    def test_noise_streams_separate_trials_and_experiments_with_intentional_pairing(self):
        namespace = "unit-test-revision-1d-unconsumed"
        seeds = {(trial, experiment): _seed(namespace, trial, experiment)
                 for trial in range(3) for experiment in range(6)}
        self.assertEqual(len(set(seeds.values())), 18)
        self.assertTrue(all(value >= 1_000_000 for value in seeds.values()))
        for (trial, experiment), seed in seeds.items():
            # Truth family is deliberately absent from the seed key, pairing
            # noise between constitutive controls for each trial/regime.
            self.assertEqual(seed, _seed(namespace, trial, experiment))
            self.assertNotEqual(seed, _seed(namespace + "-other", trial, experiment))

    def test_contrast_uncertainty_tracks_weak_shape_mode_not_common_level(self):
        # Columns span level, slope, and endpoint/center contrast.  At unit
        # multipliers only the third direction enters the center contrast.
        basis = np.column_stack((
            np.ones(3) / math.sqrt(3),
            np.array([-1.0, 0.0, 1.0]) / math.sqrt(2),
            np.array([-1.0, 2.0, -1.0]) / math.sqrt(6),
        ))
        information = contrast_information(np.diag([1000.0, 100.0, 10.0]) @ basis.T, np.ones(3))
        stronger_level = contrast_information(np.diag([1e6, 100.0, 10.0]) @ basis.T, np.ones(3))
        stronger_shape = contrast_information(np.diag([1000.0, 100.0, 100.0]) @ basis.T, np.ones(3))

        self.assertTrue(information["full_rank"])
        self.assertAlmostEqual(information["contrast_standard_error"], math.sqrt(1.5) / 10, places=12)
        self.assertAlmostEqual(stronger_level["contrast_standard_error"], information["contrast_standard_error"], places=10)
        self.assertAlmostEqual(stronger_shape["contrast_standard_error"], information["contrast_standard_error"] / 10, places=12)
        weak = np.asarray(information["weakest_direction"])
        self.assertAlmostEqual(abs(float(weak @ basis[:, 2])), 1.0, places=12)

    def test_contrast_covariance_uses_log_parameter_gradient(self):
        multipliers = np.array([0.7, 1.2, 1.4])
        precision = 20.0
        information = contrast_information(precision * np.eye(3), multipliers)
        expected = math.sqrt((0.5 * multipliers[0])**2 + multipliers[1]**2
                             + (0.5 * multipliers[2])**2) / precision
        self.assertAlmostEqual(information["contrast_standard_error"], expected, places=12)

    def test_rank_deficiency_cannot_report_precise_contrast(self):
        # A level measurement alone has no information about either shape mode.
        information = contrast_information(np.ones((4, 3)), np.ones(3))
        self.assertFalse(information["full_rank"])
        self.assertTrue(math.isinf(information["contrast_standard_error"]))

    def test_declared_grid_stays_in_material_domain_and_reverses_current(self):
        designs = candidate_designs()
        self.assertEqual(len(designs), 16)
        self.assertEqual(len({item["id"] for item in designs}), len(designs))
        with self.assertRaises(FrozenInstanceError):
            CONFIG.epochs = 1

        for design in designs:
            with self.subTest(design=design["id"]):
                self.assertIn(design["current_magnitude"], (0.4, 0.8))
                self.assertIn(design["observation_interval"], (0.04, 0.08))
                for low, high in design["windows"]:
                    self.assertGreaterEqual(low, 285.0)
                    self.assertLessEqual(high, 315.0)
                    self.assertLess(low, high)
                experiments = design_experiments(design)
                self.assertEqual(len(experiments), 2 * len(design["windows"]))
                for window in design["windows"]:
                    paired = [experiment for experiment in experiments if (
                        experiment.initial_cold_face_temperature,
                        experiment.initial_hot_face_temperature,
                    ) == tuple(window)]
                    self.assertEqual(len(paired), 2)
                    self.assertEqual(
                        {current_at(experiment.current, 0.0) for experiment in paired},
                        {-design["current_magnitude"], design["current_magnitude"]},
                    )
                    for experiment in paired:
                        temperatures = (
                            experiment.initial_cold_face_temperature,
                            experiment.initial_hot_face_temperature,
                            experiment.cold_reservoir_temperature,
                            experiment.hot_reservoir_temperature,
                            *experiment.initial_cells(),
                        )
                        self.assertTrue(all(285.0 <= value <= 315.0 for value in temperatures))

    def test_separated_thermal_bands_add_nominal_shape_information(self):
        # Match current and sampling density, so information gain comes from
        # the declared thermal design rather than noise-scale changes.
        designs = candidate_designs()
        central = next(item for item in designs if len(item["windows"]) == 1
                       and tuple(item["windows"][0]) == (295.0, 305.0)
                       and item["current_magnitude"] == 0.8
                       and item["observation_interval"] == 0.08)
        bands = next(item for item in designs if tuple(map(tuple, item["windows"])) == (
            (286.0, 290.0), (298.0, 302.0), (310.0, 314.0))
            and item["current_magnitude"] == 0.8
            and item["observation_interval"] == 0.08)

        central_jacobian, central_rows, central_values = noise_whitened_jacobian(
            design_experiments(central), np.zeros(3), central["observation_interval"])
        band_jacobian, band_rows, band_values = noise_whitened_jacobian(
            design_experiments(bands), np.zeros(3), bands["observation_interval"])
        self.assertEqual(central_jacobian.shape, (len(central_rows), 3))
        self.assertEqual(band_jacobian.shape, (len(band_rows), 3))
        self.assertEqual(len(central_values), len(central_rows))
        self.assertEqual(len(band_values), len(band_rows))
        self.assertTrue(np.isfinite(band_jacobian).all())
        central_info = contrast_information(central_jacobian, np.ones(3))
        band_info = contrast_information(band_jacobian, np.ones(3))
        self.assertTrue(central_info["full_rank"])
        self.assertTrue(band_info["full_rank"])
        # Normalize singular values by observation count: merely repeating the
        # central experiment three times cannot satisfy this stronger test.
        central_weak = min(central_info["singular_values"]) / math.sqrt(len(central_rows))
        band_weak = min(band_info["singular_values"]) / math.sqrt(len(band_rows))
        self.assertGreater(band_weak, 2 * central_weak)
        self.assertLess(band_info["contrast_standard_error"], central_info["contrast_standard_error"] / 2)

    def test_zero_current_has_no_nominal_resistivity_information(self):
        design = candidate_designs()[0]
        experiment = replace(design_experiments(design)[0], current=0.0)
        jacobian, rows, values = noise_whitened_jacobian(
            (experiment,), np.zeros(3), design["observation_interval"])
        self.assertGreater(len(rows), 0)
        self.assertEqual(len(rows), len(values))
        np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))
        self.assertFalse(contrast_information(jacobian, np.ones(3))["full_rank"])


if __name__ == "__main__":
    unittest.main()
