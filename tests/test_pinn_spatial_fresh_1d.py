"""Scientific checks for the additive prospective 1-D PINN protocol."""

from dataclasses import replace
import math
import unittest

import torch

from thermotwin.pinn.distributed_inverse import (
    InverseDistributedPropertyConfig,
    _temperature_model_observation_predictions,
    train_multi_experiment_inverse_distributed_property_pinn,
)
from thermotwin.simulation.distributed import distributed_inverse_constant_experiments
from thermotwin.simulation.distributed_independent import (
    independent_distributed_voltage,
    observe_independent_distributed_result,
    run_independent_distributed_experiment,
)
from thermotwin.studies.distributed_independent_validation import smooth_resistivity_truth
from thermotwin.studies.pinn_spatial_fresh_1d import (
    Fresh1DValidationConfig,
    _frozen_model_metrics,
    _source_hashes,
    fresh_1d_curve_diagnostics,
    fresh_1d_operational_decision,
    fresh_1d_seeds,
)


class Fresh1DScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_independent_seed_namespace_separates_trials_and_roles(self):
        namespace = Fresh1DValidationConfig().namespace
        blocks = [fresh_1d_seeds(namespace, index) for index in range(3)]
        all_seeds = [seed for block in blocks for seed in (block["neural"], *block["observations"])]
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        self.assertTrue(all(value >= 1_000_000 for value in all_seeds))
        self.assertEqual(blocks[0], fresh_1d_seeds(namespace, 0))
        self.assertNotEqual(blocks[0], fresh_1d_seeds(namespace + "other", 0))
        self.assertTrue(all(len(digest) == 64 for digest in _source_hashes().values()))

    def test_truth_blind_gate_cannot_substitute_low_data_error_for_physics(self):
        config = Fresh1DValidationConfig()
        self.assertFalse(fresh_1d_operational_decision(0.0, 0.3, config)["acceptable"])
        self.assertFalse(fresh_1d_operational_decision(3.0, 0.0, config)["acceptable"])
        self.assertTrue(fresh_1d_operational_decision(2.0, 0.25, config)["acceptable"])
        for value in (math.inf, math.nan):
            self.assertFalse(fresh_1d_operational_decision(value, 0.0, config)["acceptable"])

    def test_knot_match_does_not_remove_cubic_representation_error(self):
        baseline = distributed_inverse_constant_experiments()[0].material.electrical_resistivity
        truth = smooth_resistivity_truth()
        exact_knots = baseline.with_values(tuple(truth.value(t) for t in baseline.temperatures))
        diagnostic = fresh_1d_curve_diagnostics(exact_knots, baseline, truth, Fresh1DValidationConfig())
        self.assertTrue(diagnostic["knot_shape_recovered"])
        self.assertGreater(diagnostic["actual_cubic_relative_rmse"], 1e-4)
        self.assertGreater(diagnostic["representation_projection_relative_rmse"], 1e-4)
        self.assertLess(diagnostic["representation_projection_relative_rmse"], diagnostic["actual_cubic_relative_rmse"])
        self.assertFalse(diagnostic["projection_is_data_effective_truth"])

    def test_flat_multiplier_can_approximate_level_without_recovering_shape(self):
        baseline = distributed_inverse_constant_experiments()[0].material.electrical_resistivity
        flat = baseline.with_values(tuple(value * 1.0466666666667 for value in baseline.values))
        diagnostic = fresh_1d_curve_diagnostics(flat, baseline, smooth_resistivity_truth(), Fresh1DValidationConfig())
        self.assertLess(diagnostic["actual_cubic_relative_rmse"], 0.03)
        self.assertFalse(diagnostic["knot_shape_recovered"])
        self.assertIn("knot_amplitude_ratio", diagnostic["knot_shape_failure_reasons"])

    def test_zero_current_voltage_cannot_identify_resistivity(self):
        experiment = distributed_inverse_constant_experiments()[0]
        state = (290.0, 295.0, 300.0, 305.0, 310.0)
        original = experiment.material.electrical_resistivity
        altered = replace(experiment, material=replace(experiment.material,
            electrical_resistivity=original.with_values(tuple(2 * value for value in original.values))))
        self.assertAlmostEqual(independent_distributed_voltage(experiment, state, current=0.0),
                               independent_distributed_voltage(altered, state, current=0.0))
        self.assertGreater(abs(independent_distributed_voltage(experiment, state, current=0.8)
                               - independent_distributed_voltage(altered, state, current=0.8)), 1e-4)

    def test_final_loss_is_recomputed_for_the_saved_optimizer_state(self):
        experiment = distributed_inverse_constant_experiments()[1]
        clean = observe_independent_distributed_result(experiment,
            run_independent_distributed_experiment(experiment, node_count=5, time_step=0.002),
            observation_interval=0.2)
        training_config = InverseDistributedPropertyConfig(
            property_name="electrical_resistivity", hidden_width=4, hidden_layers=1,
            interior_space_points=2, time_points=2, voltage_space_points=16,
            epochs=1, network_learning_rate=2e-3, property_learning_rate=2e-3,
            physics_weight=10.0, seed=1_123_456,
        )
        training = train_multi_experiment_inverse_distributed_property_pinn(
            (experiment,), (clean,), training_config)
        predicted = _temperature_model_observation_predictions(
            training.model.temperature_models[0], training.model.baseline_material,
            training.model.property_overrides(), experiment, clean, voltage_space_points=16)
        observed = torch.tensor(clean.values(), dtype=torch.float32)
        scales = torch.tensor([training_config.observation_scale(item.channel) for item in clean.observations])
        final_loss = float(((predicted - observed) / scales).detach().square().mean())
        metrics = _frozen_model_metrics(training, (experiment,), (clean,), Fresh1DValidationConfig(), 1.0)
        self.assertAlmostEqual(metrics["observation_loss"], final_loss, places=5)
        self.assertNotAlmostEqual(metrics["observation_loss"], training.history.observation_loss[-1], places=3)
        self.assertTrue(math.isfinite(metrics["denser_grid_physics_residual_ratio"]))
        self.assertTrue(all(not parameter.requires_grad for parameter in training.model.parameters()))


if __name__ == "__main__":
    unittest.main()
