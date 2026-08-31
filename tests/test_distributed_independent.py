from dataclasses import replace
import unittest

from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.observations.distributed import DistributedObservationChannels
from thermotwin.simulation.distributed import distributed_reference_experiment
from thermotwin.simulation.distributed_independent import (
    PolynomialTemperatureProperty,
    observe_independent_distributed_result,
    run_independent_distributed_experiment,
)


class DistributedIndependentTruthTests(unittest.TestCase):
    def test_polynomial_property_has_analytic_oriented_integral(self):
        prop = PolynomialTemperatureProperty(300.0, 10.0, (2.0, 0.5, 0.25))
        forward = prop.integral(295.0, 305.0)
        reverse = prop.integral(305.0, 295.0)
        self.assertAlmostEqual(forward, reverse * -1.0)
        self.assertAlmostEqual(prop.integral(300.0, 300.0), 0.0)

    def test_equal_temperature_zero_current_is_stationary(self):
        experiment = distributed_reference_experiment(
            temperature_dependent=True,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
            duration=0.01,
        )
        experiment = replace(
            experiment,
            initial_cold_face_temperature=300.0,
            initial_hot_face_temperature=300.0,
        )
        result = run_independent_distributed_experiment(
            experiment, node_count=9, time_step=0.0005
        )
        self.assertTrue(
            all(
                abs(temperature - 300.0) < 1.0e-12
                for state in result.trajectory.temperature
                for temperature in state
            )
        )
        self.assertTrue(all(abs(value) < 1.0e-12 for value in result.diagnostics.voltage))

    def test_transition_is_an_exact_truth_time_and_observations_are_right_continuous(self):
        experiment = distributed_reference_experiment(
            temperature_dependent=True,
            current=PiecewiseConstantCurrent.step(
                transition_time=0.007,
                before_current=0.0,
                after_current=0.5,
            ),
            duration=0.012,
        )
        result = run_independent_distributed_experiment(
            experiment, node_count=9, time_step=0.002
        )
        self.assertIn(0.007, result.trajectory.time)
        observations = observe_independent_distributed_result(
            experiment,
            result,
            observation_interval=0.007,
            channels=DistributedObservationChannels(
                cold_face_temperature=False,
                hot_face_temperature=False,
                voltage=True,
            ),
        )
        switched = next(item for item in observations.observations if item.time == 0.007)
        self.assertGreater(switched.value, 0.0)

    def test_truth_time_step_refinement_converges(self):
        experiment = distributed_reference_experiment(
            temperature_dependent=True,
            current=0.8,
            duration=0.02,
        )
        coarse = run_independent_distributed_experiment(
            experiment, node_count=11, time_step=0.0005
        )
        fine = run_independent_distributed_experiment(
            experiment, node_count=11, time_step=0.00025
        )
        for coarse_value, fine_value in zip(
            coarse.trajectory.temperature[-1], fine.trajectory.temperature[-1]
        ):
            self.assertAlmostEqual(coarse_value, fine_value, places=7)

    def test_truth_spatial_refinement_converges(self):
        experiment = distributed_reference_experiment(
            temperature_dependent=True,
            current=0.8,
            duration=0.01,
        )
        results = {
            nodes: run_independent_distributed_experiment(
                experiment, node_count=nodes, time_step=0.0001
            )
            for nodes in (9, 17, 25, 33)
        }
        reference = results[33]
        cold_errors = tuple(
            abs(
                results[nodes].trajectory.temperature[-1][0]
                - reference.trajectory.temperature[-1][0]
            )
            for nodes in (9, 17, 25)
        )
        voltage_errors = tuple(
            abs(results[nodes].diagnostics.voltage[-1] - reference.diagnostics.voltage[-1])
            for nodes in (9, 17, 25)
        )
        self.assertGreater(cold_errors[0], cold_errors[1])
        self.assertGreater(cold_errors[1], cold_errors[2])
        self.assertGreater(voltage_errors[0], voltage_errors[1])
        self.assertGreater(voltage_errors[1], voltage_errors[2])


if __name__ == "__main__":
    unittest.main()
