from dataclasses import replace
import math
import unittest

from thermotwin.observations.distributed import (
    DistributedObservationChannels,
    run_distributed_virtual_experiment,
)
from thermotwin.physics.distributed import PiecewiseLinearProperty
from thermotwin.simulation.distributed import distributed_reference_experiment

try:
    import torch

    from thermotwin.pinn.distributed_inverse import (
        InverseDistributedPropertyConfig,
        InverseDistributedPropertyPINN,
        train_inverse_distributed_property_pinn,
        train_multi_experiment_inverse_distributed_property_pinn,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class DistributedInversePINNTests(unittest.TestCase):
    def setUp(self):
        self.experiment = distributed_reference_experiment(
            temperature_dependent=True,
            current=0.5,
            duration=0.06,
            cell_count=4,
            time_step=0.001,
        )

    def test_property_parameterization_preserves_sign_and_positivity(self):
        model = InverseDistributedPropertyPINN(
            self.experiment,
            property_name="electrical_resistivity",
            baseline_material=self.experiment.material,
            initial_log_multipliers=(0.2, -0.1, 0.05),
            hidden_width=8,
            hidden_layers=1,
            temperature_scale=5.0,
        )
        self.assertTrue(torch.all(model.property_values > 0.0))
        self.assertEqual(tuple(model.property_values.shape), (3,))

    def test_all_three_property_curves_can_be_released_independently(self):
        for property_name in (
            "seebeck_coefficient",
            "electrical_resistivity",
            "thermal_conductivity",
        ):
            with self.subTest(property_name=property_name):
                model = InverseDistributedPropertyPINN(
                    self.experiment,
                    property_name=property_name,
                    baseline_material=self.experiment.material,
                    initial_log_multipliers=(0.0, 0.0, 0.0),
                    hidden_width=8,
                    hidden_layers=1,
                    temperature_scale=5.0,
                )
                self.assertEqual(
                    tuple(model.property_overrides()),
                    (property_name,),
                )
                self.assertEqual(tuple(model.property_values.shape), (3,))

    def test_short_inverse_training_reduces_joint_loss(self):
        baseline_property = self.experiment.material.electrical_resistivity
        self.assertIsInstance(baseline_property, PiecewiseLinearProperty)
        truth_property = baseline_property.with_values(
            tuple(value * 1.06 for value in baseline_property.values)
        )
        truth_experiment = replace(
            self.experiment,
            material=replace(
                self.experiment.material,
                electrical_resistivity=truth_property,
            ),
        )
        channels = DistributedObservationChannels(
            cold_face_temperature=True,
            hot_face_temperature=True,
            voltage=True,
        )
        observations = run_distributed_virtual_experiment(
            truth_experiment,
            observation_interval=0.03,
            channels=channels,
        )
        training = train_inverse_distributed_property_pinn(
            truth_experiment,
            observations,
            InverseDistributedPropertyConfig(
                property_name="electrical_resistivity",
                hidden_width=10,
                hidden_layers=2,
                interior_space_points=4,
                time_points=6,
                voltage_space_points=8,
                epochs=80,
                network_learning_rate=3e-3,
                property_learning_rate=2e-3,
                initial_log_multipliers=(math.log(0.9),) * 3,
                residual_rate_scale=1.0,
                seed=12,
                device="cpu",
            ),
            baseline_material=self.experiment.material,
        )
        self.assertLess(
            training.history.total_loss[-1],
            training.history.total_loss[0],
        )
        self.assertTrue(
            all(value > 0.0 for value in training.history.property_values[-1])
        )
        self.assertNotEqual(
            training.history.property_values[-1],
            training.history.property_values[0],
        )

    def test_two_experiments_share_one_property_curve(self):
        second = replace(self.experiment, current=-0.5)
        channels = DistributedObservationChannels(
            cold_face_temperature=True,
            hot_face_temperature=True,
            voltage=True,
        )
        observations = tuple(
            run_distributed_virtual_experiment(
                experiment,
                observation_interval=0.03,
                channels=channels,
            )
            for experiment in (self.experiment, second)
        )
        training = train_multi_experiment_inverse_distributed_property_pinn(
            (self.experiment, second),
            observations,
            InverseDistributedPropertyConfig(
                property_name="thermal_conductivity",
                hidden_width=8,
                hidden_layers=1,
                interior_space_points=3,
                time_points=4,
                voltage_space_points=5,
                epochs=15,
                initial_log_multipliers=(0.05, 0.05, 0.05),
                seed=3,
                device="cpu",
            ),
        )
        self.assertEqual(len(training.model.temperature_models), 2)
        self.assertEqual(len(training.history.property_values[-1]), 3)
        self.assertLess(
            training.history.total_loss[-1],
            training.history.total_loss[0],
        )


if __name__ == "__main__":
    unittest.main()
