import inspect
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.design.control_comparison import piecewise_electrical_energy
from thermotwin.physics.thermoelectric import ThermoelectricParameters
from thermotwin.simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
)
from thermotwin.studies import operating_decision_realism as realism
from thermotwin.studies.operating_decision import (
    ACQUISITION,
    FINAL_EVALUATION,
    OperatingBand,
    OperatingRegime,
    default_fixed_policies,
    operating_margin,
)
from thermotwin.studies.operating_decision_prospective_costs import (
    prospective_action_resources,
)
from thermotwin.studies.operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
    RunInstrumentation,
    _truth_prediction,
    realism_truth_for_trial,
)
from thermotwin.studies.operating_decision_resources import (
    RealizedRunEnergy,
    realized_regime_terminal_energy,
)
from thermotwin.studies.sensor_model_discrimination import COLD_EXCHANGER


def _config_with_step(time_step: float) -> OperatingDecisionRealismConfig:
    base = OperatingDecisionRealismConfig()
    sensor = replace(
        base.sensor,
        dense_time_step=time_step,
        fit=replace(base.sensor.fit, dense_time_step=time_step),
        selection=replace(base.sensor.selection, dense_time_step=time_step),
    )
    return replace(base, sensor=sensor)


def _final_prediction(
    truth_condition: str,
    config: OperatingDecisionRealismConfig,
):
    regime = OperatingRegime(
        "untouched_final_operating_schedule",
        FINAL_EVALUATION,
        config.final_current,
        (),
    )
    return _truth_prediction(
        truth_condition,
        realism_truth_for_trial(config, 0),
        regime,
        RunInstrumentation(False),
        config,
    )


class OperatingDecisionPhaseBAcceptanceTests(unittest.TestCase):
    def test_declared_campaign_parameters_are_physical(self):
        reference = constant_current_contact_reference_experiment()
        thermoelectric = reference.thermoelectric_parameters
        thermal = reference.thermal_parameters
        self.assertTrue(math.isfinite(thermoelectric.seebeck_coefficient))
        self.assertGreater(thermoelectric.seebeck_coefficient, 0.0)
        self.assertGreater(thermoelectric.electrical_resistance, 0.0)
        self.assertGreater(thermoelectric.thermal_conductance, 0.0)
        for value in vars(thermal).values():
            self.assertTrue(math.isfinite(value))
            self.assertGreater(value, 0.0)

        config = OperatingDecisionRealismConfig()
        for trial_index in range(3):
            truth = realism_truth_for_trial(config, trial_index)
            self.assertTrue(all(value > 0.0 for value in truth.physical_values))
            self.assertGreater(truth.interface_mass, 0.0)
            self.assertGreater(truth.series_resistance, 0.0)
            self.assertGreater(truth.face_sensor.thermal_capacitance, 0.0)
            self.assertGreater(truth.face_sensor.response_time_constant, 0.0)

    def test_final_margin_extremum_is_switch_aligned_and_grid_converged(self):
        steps = (0.5, 0.25, 0.125, 0.0625)
        transitions = OperatingDecisionRealismConfig().final_current.transition_times

        for truth_condition in STAGE3_TRUTH_CONDITIONS:
            with self.subTest(truth_condition=truth_condition):
                predictions = tuple(
                    _final_prediction(truth_condition, _config_with_step(step))
                    for step in steps
                )
                for prediction in predictions:
                    self.assertTrue(set(transitions).issubset(prediction.time))
                    peak_index = max(
                        range(len(prediction.cold_face)),
                        key=prediction.cold_face.__getitem__,
                    )
                    self.assertEqual(prediction.time[peak_index], 58.0)

                margins = tuple(
                    operating_margin(
                        prediction.cold_face,
                        OperatingDecisionRealismConfig().band,
                    )
                    for prediction in predictions
                )
                reference = margins[-1]
                self.assertLess(abs(margins[1] - reference), 1.0e-6)
                self.assertLess(
                    abs(margins[1] - reference),
                    abs(margins[0] - reference),
                )

                reference_peak = max(predictions[-1].cold_face)
                for offset, expected_inside in ((1.0e-6, True), (-1.0e-6, False)):
                    band = OperatingBand(280.0, reference_peak + offset)
                    signs = tuple(
                        operating_margin(prediction.cold_face, band) >= 0.0
                        for prediction in predictions[1:]
                    )
                    self.assertEqual(signs, (expected_inside,) * len(signs))
                zero_band = OperatingBand(280.0, reference_peak)
                self.assertEqual(
                    operating_margin(predictions[-1].cold_face, zero_band),
                    0.0,
                )

    def test_terminal_energy_sign_and_nominal_realized_boundary_are_explicit(self):
        parameters = ThermoelectricParameters(
            seebeck_coefficient=0.05,
            electrical_resistance=2.0,
            thermal_conductance=0.5,
        )
        time = (0.0, 10.0)
        cold = (300.0, 300.0)
        hot = (310.0, 310.0)
        consuming = piecewise_electrical_energy(
            time,
            cold,
            hot,
            parameters,
            PiecewiseConstantCurrent.constant(0.1),
            start_time=0.0,
            end_time=10.0,
        )
        generating = piecewise_electrical_energy(
            time,
            cold,
            hot,
            parameters,
            PiecewiseConstantCurrent.constant(-0.1),
            start_time=0.0,
            end_time=10.0,
        )
        self.assertAlmostEqual(consuming, 0.7)
        self.assertAlmostEqual(generating, -0.3)

        regime = OperatingRegime(
            "signed_energy_probe",
            ACQUISITION,
            PiecewiseConstantCurrent.constant(-0.1),
            (COLD_EXCHANGER,),
        )
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            RealizedRunEnergy(
                truth_condition=STAGE3_TRUTH_CONDITIONS[0],
                trial_index=0,
                regime=regime,
                instrumentation=RunInstrumentation(False),
                terminal_energy=generating,
            )

        selection_parameters = set(
            inspect.signature(prospective_action_resources).parameters
        )
        realized_parameters = set(
            inspect.signature(realized_regime_terminal_energy).parameters
        )
        self.assertEqual(selection_parameters, {"physical_config", "scenario"})
        self.assertTrue({"truth_condition", "trial_index"} <= realized_parameters)

    def test_campaign_runner_saves_before_reveal_and_scores_after_reveal(self):
        base = OperatingDecisionRealismConfig()
        config = replace(base, sensor=replace(base.sensor, trial_count=1))
        events = []

        def build(_truth, _trial, policy, _config):
            events.append(("build", policy.name))
            return policy.name

        def decide(case, *, config):
            events.append(("save", case))
            return f"saved:{case}"

        def reveal(_truth, _trial, case, _config):
            events.append(("reveal", case))
            return f"revealed:{case}"

        def score(saved, revealed, case, _energies, _config):
            self.assertEqual(saved, f"saved:{case}")
            self.assertEqual(revealed, f"revealed:{case}")
            events.append(("score", case))
            return f"scored:{case}"

        with (
            patch.object(realism, "_nominal_energy_by_run", return_value={}),
            patch.object(realism, "build_realistic_blinded_case", side_effect=build),
            patch.object(realism, "decide_realistic_blinded_case", side_effect=decide),
            patch.object(
                realism,
                "reveal_realistic_operating_outcome",
                side_effect=reveal,
            ),
            patch.object(realism, "_score_realistic_decision", side_effect=score),
        ):
            records = realism.run_operating_decision_realism_truth(
                STAGE3_TRUTH_CONDITIONS[0],
                config,
            )

        policies = tuple(policy.name for policy in default_fixed_policies())
        self.assertEqual(records, tuple(f"scored:{name}" for name in policies))
        self.assertEqual(
            tuple(events),
            tuple(
                event
                for name in policies
                for event in (
                    ("build", name),
                    ("save", name),
                    ("reveal", name),
                    ("score", name),
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
