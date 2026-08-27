import unittest

from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.observations.distributed import (
    DistributedObservationChannels,
    add_distributed_gaussian_noise,
    observe_distributed_trajectory,
    regular_distributed_observation_times,
)
from thermotwin.simulation.distributed import (
    distributed_reference_experiment,
    run_distributed_leg_experiment,
)


class DistributedObservationTests(unittest.TestCase):
    def test_regular_times_include_exact_duration(self):
        self.assertEqual(
            regular_distributed_observation_times(1.0, 0.3),
            (0.0, 0.3, 0.6, 0.8999999999999999, 1.0),
        )

    def test_voltage_at_switch_uses_right_continuous_current(self):
        experiment = distributed_reference_experiment(
            current=PiecewiseConstantCurrent.step(
                transition_time=0.015,
                before_current=0.0,
                after_current=0.5,
            ),
            duration=0.03,
            cell_count=5,
            time_step=0.01,
        )
        result = run_distributed_leg_experiment(experiment)
        observations = observe_distributed_trajectory(
            experiment,
            result.trajectory,
            times=(0.0, 0.015, 0.03),
            channels=DistributedObservationChannels(
                cold_face_temperature=False,
                hot_face_temperature=False,
                voltage=True,
            ),
        )
        voltage = observations.values()
        self.assertGreater(voltage[1], voltage[0])

    def test_noise_is_reproducible_and_channel_scaled(self):
        experiment = distributed_reference_experiment(
            duration=0.01, cell_count=4, time_step=0.001
        )
        result = run_distributed_leg_experiment(experiment)
        observations = observe_distributed_trajectory(
            experiment,
            result.trajectory,
            times=(0.0, 0.01),
        )
        scales = {
            "cold_face_temperature": 0.01,
            "hot_face_temperature": 0.01,
            "voltage": 1e-5,
        }
        first = add_distributed_gaussian_noise(
            observations, standard_deviations=scales, seed=4
        )
        second = add_distributed_gaussian_noise(
            observations, standard_deviations=scales, seed=4
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, observations)


if __name__ == "__main__":
    unittest.main()
