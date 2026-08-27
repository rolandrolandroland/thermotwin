from dataclasses import replace
import unittest

from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.simulation.distributed import (
    distributed_reference_experiment,
    run_distributed_leg_experiment,
)


class DistributedSimulationTests(unittest.TestCase):
    def test_equilibrium_is_preserved_exactly(self):
        experiment = distributed_reference_experiment(
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
            duration=0.05,
            cell_count=5,
            time_step=0.001,
        )
        experiment = replace(
            experiment,
            initial_cold_face_temperature=300.0,
            initial_hot_face_temperature=300.0,
        )
        result = run_distributed_leg_experiment(experiment)
        self.assertTrue(all(value == 300.0 for value in result.trajectory.cold_face))
        self.assertTrue(all(value == 300.0 for value in result.trajectory.hot_face))
        self.assertTrue(
            all(value == 300.0 for row in result.trajectory.cells for value in row)
        )

    def test_current_transition_is_an_exact_output_time(self):
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
        self.assertIn(0.015, result.trajectory.time)
        index = result.trajectory.time.index(0.015)
        self.assertEqual(result.diagnostics.current[index], 0.5)

    def test_positive_current_initially_cools_cold_face_and_heats_hot_face(self):
        experiment = distributed_reference_experiment(
            current=0.5,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
            duration=0.01,
            cell_count=5,
            time_step=0.0005,
        )
        experiment = replace(
            experiment,
            initial_cold_face_temperature=300.0,
            initial_hot_face_temperature=300.0,
        )
        result = run_distributed_leg_experiment(experiment)
        self.assertLess(result.trajectory.cold_face[-1], 300.0)
        self.assertGreater(result.trajectory.hot_face[-1], 300.0)

    def test_halving_time_step_converges(self):
        coarse_experiment = distributed_reference_experiment(
            current=0.8,
            duration=0.1,
            cell_count=6,
            time_step=0.002,
        )
        fine_experiment = replace(coarse_experiment, time_step=0.001)
        coarse = run_distributed_leg_experiment(coarse_experiment).trajectory
        fine = run_distributed_leg_experiment(fine_experiment).trajectory
        self.assertLess(abs(coarse.cold_face[-1] - fine.cold_face[-1]), 1e-7)
        self.assertLess(abs(coarse.hot_face[-1] - fine.hot_face[-1]), 1e-7)

    def test_diagnostics_close_energy_at_every_state(self):
        result = run_distributed_leg_experiment(
            distributed_reference_experiment(
                current=-0.4,
                duration=0.03,
                cell_count=5,
                time_step=0.001,
            )
        )
        self.assertLess(
            max(
                abs(value)
                for value in result.diagnostics.instantaneous_energy_balance_residual
            ),
            1e-12,
        )


if __name__ == "__main__":
    unittest.main()
