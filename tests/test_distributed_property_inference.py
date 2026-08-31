from dataclasses import replace
import math
import unittest

from thermotwin.inference.distributed_properties import (
    DistributedPropertyFitConfig,
    fit_distributed_property,
)
from thermotwin.observations.distributed import (
    DistributedObservationChannels,
    run_distributed_virtual_experiment,
)
from thermotwin.physics.distributed import PiecewiseLinearProperty
from thermotwin.simulation.distributed import distributed_reference_experiment


class DistributedPropertyInferenceTests(unittest.TestCase):
    def test_continuous_fit_reduces_loss_without_truth_grid_alignment(self):
        nominal = distributed_reference_experiment(
            temperature_dependent=True,
            current=0.6,
            duration=0.12,
            cell_count=4,
            time_step=0.001,
        )
        baseline = nominal.material.electrical_resistivity
        self.assertIsInstance(baseline, PiecewiseLinearProperty)
        truth_offsets = (0.0, math.log(1.07), 0.0)
        truth_property = baseline.with_values(
            tuple(
                value * math.exp(offset)
                for value, offset in zip(baseline.values, truth_offsets)
            )
        )
        truth = replace(
            nominal,
            material=replace(
                nominal.material,
                electrical_resistivity=truth_property,
            ),
        )
        channels = DistributedObservationChannels(
            cold_face_temperature=False,
            hot_face_temperature=False,
            voltage=True,
        )
        observations = run_distributed_virtual_experiment(
            truth,
            observation_interval=0.03,
            channels=channels,
        )
        fit = fit_distributed_property(
            (nominal,),
            (observations,),
            DistributedPropertyFitConfig(
                property_name="electrical_resistivity",
                observation_interval=0.03,
                channels=channels,
                initial_log_multipliers=(0.0, -0.08, 0.0),
                log_multiplier_bounds=(-0.15, 0.15),
                coordinate_passes=2,
                golden_section_iterations=8,
                voltage_standard_deviation=1e-5,
            ),
        )
        self.assertLess(
            fit.mean_normalized_squared_error,
            fit.evaluations[0].mean_normalized_squared_error,
        )
        self.assertAlmostEqual(
            fit.fitted_values[1] / baseline.values[1],
            1.07,
            delta=0.02,
        )
        self.assertNotEqual(fit.log_multipliers[1], truth_offsets[1])

    def test_mismatched_observation_count_is_rejected(self):
        with self.assertRaises(ValueError):
            DistributedPropertyFitConfig(
                property_name="thermal_conductivity",
                smoothness_weight=-1.0,
            )
        experiment = distributed_reference_experiment(
            temperature_dependent=True,
            duration=0.01,
            cell_count=4,
            time_step=0.001,
        )
        with self.assertRaises(ValueError):
            fit_distributed_property(
                (experiment,),
                (),
                DistributedPropertyFitConfig(
                    property_name="thermal_conductivity"
                ),
            )


if __name__ == "__main__":
    unittest.main()
