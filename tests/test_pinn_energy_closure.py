from dataclasses import replace
import unittest

try:
    import torch

    from thermotwin.pinn.energy_closure import (
        ContactPINNEnergyClosureConfig,
        evaluate_piecewise_contact_energy_closure,
    )
    from thermotwin.pinn.forward_piecewise import (
        PiecewiseContactForwardPINN,
        unipolar_pulse_contact_experiment,
    )
except ModuleNotFoundError:
    torch = None


def _constant_model(experiment, transitions):
    model = PiecewiseContactForwardPINN(
        duration=experiment.duration,
        transition_times=transitions,
        initial_temperatures=(300.0, 300.0, 300.0, 300.0),
        hidden_width=4,
        hidden_layers=1,
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    return model


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class PINNEnergyClosureTests(unittest.TestCase):
    def test_zero_current_equilibrium_closes_exactly(self):
        reference = unipolar_pulse_contact_experiment()
        experiment = replace(reference, current=0.0)
        model = _constant_model(experiment, ())

        result = evaluate_piecewise_contact_energy_closure(
            model,
            experiment,
            ContactPINNEnergyClosureConfig(sampling_interval=1.0),
        )

        self.assertEqual(result.rate_closure_rms, 0.0)
        self.assertEqual(result.final_cumulative_closure_error, 0.0)
        self.assertEqual(result.normalized_rate_closure_rms, 0.0)

    def test_power_jump_is_integrated_per_segment_without_a_ramp(self):
        experiment = unipolar_pulse_contact_experiment()
        model = _constant_model(experiment, (5.0, 20.0))

        result = evaluate_piecewise_contact_energy_closure(
            model,
            experiment,
            ContactPINNEnergyClosureConfig(sampling_interval=2.0),
        )

        expected_electrical_energy = (
            experiment.thermoelectric_parameters.electrical_resistance * 15.0
        )
        self.assertAlmostEqual(
            result.cumulative_net_input[-1], expected_electrical_energy, places=5
        )
        self.assertAlmostEqual(
            result.final_cumulative_closure_error,
            -expected_electrical_energy,
            places=5,
        )
        at_first_switch = tuple(
            current
            for time, current in zip(result.time, result.current)
            if time == 5.0
        )
        at_second_switch = tuple(
            current
            for time, current in zip(result.time, result.current)
            if time == 20.0
        )
        self.assertEqual(at_first_switch, (0.0, 1.0))
        self.assertEqual(at_second_switch, (1.0, 0.0))

    def test_invalid_interval_and_model_are_rejected(self):
        with self.assertRaises(ValueError):
            ContactPINNEnergyClosureConfig(sampling_interval=0.0)
        with self.assertRaises(ValueError):
            evaluate_piecewise_contact_energy_closure(
                torch.nn.Linear(1, 4),
                unipolar_pulse_contact_experiment(),
            )

        experiment = unipolar_pulse_contact_experiment()
        model_with_wrong_switch = _constant_model(experiment, (5.0, 19.0))
        with self.assertRaisesRegex(ValueError, "current switches"):
            evaluate_piecewise_contact_energy_closure(
                model_with_wrong_switch,
                experiment,
            )


if __name__ == "__main__":
    unittest.main()
