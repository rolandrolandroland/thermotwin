import math
import unittest
from dataclasses import replace

from thermotwin.reports.operating_decision_realism import (
    format_operating_decision_realism_report,
)
from thermotwin.studies.operating_decision import (
    FIXED_FACE_TEMPERATURE,
    FIXED_VOLTAGE,
    default_fixed_policies,
)
from thermotwin.studies.operating_decision_realism import (
    FIT_BOUND_TOLERANCE,
    STAGE3_TRUTH_CONDITIONS,
    TEMPERATURE_DEPENDENT_CONTACT,
    OperatingDecisionRealismConfig,
    RealisticObservablePrediction,
    RealisticOperatingRun,
    _box_projected_gradient,
    _margin_at_offsets,
    _regularized_bias_residuals,
    build_realistic_blinded_case,
    decide_realistic_blinded_case,
    fit_realistic_acquisition_models,
    nominal_realistic_margins,
    realism_truth_for_trial,
    reveal_realistic_operating_outcome,
    run_operating_decision_realism,
    score_realistic_saved_decision,
)
from thermotwin.studies.sensor_model_discrimination import (
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
    MODEL_NAMES,
    VOLTAGE,
    ObservableRun,
    ObservableValue,
)


class OperatingDecisionRealismTests(unittest.TestCase):
    @staticmethod
    def small_config(*, fit_iterations=1):
        base = OperatingDecisionRealismConfig()
        return replace(
            base,
            sensor=replace(
                base.sensor,
                trial_count=1,
                fit_iterations=fit_iterations,
            ),
        )

    def test_box_projected_gradient_uses_kkt_signs_at_active_bounds(self):
        bounds = ((0.0, 1.0),) * 6
        projected = _box_projected_gradient(
            (0.0, 0.0, 1.0, 1.0, 0.5, 0.5),
            (2.0, -2.0, -2.0, 2.0, 3.0, -3.0),
            bounds,
        )
        self.assertEqual(projected, (0.0, -2.0, 0.0, 2.0, 3.0, -3.0))
        self.assertEqual(
            _box_projected_gradient(
                (0.5 * FIT_BOUND_TOLERANCE, 2.0 * FIT_BOUND_TOLERANCE),
                (1.0, 1.0),
                ((0.0, 1.0), (0.0, 1.0)),
            ),
            (0.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            _box_projected_gradient((0.5,), (math.nan,), ((0.0, 1.0),))

    def test_realism_assumptions_are_reproducible_and_in_calibration_ranges(self):
        config = self.small_config()
        first = realism_truth_for_trial(config, 0)
        self.assertEqual(first, realism_truth_for_trial(config, 0))
        self.assertNotEqual(first, realism_truth_for_trial(config, 1))
        self.assertTrue(
            config.series_resistance_bounds[0]
            <= first.series_resistance
            <= config.series_resistance_bounds[1]
        )
        self.assertTrue(
            config.face_sensor_capacitance_bounds[0]
            <= first.face_sensor.thermal_capacitance
            <= config.face_sensor_capacitance_bounds[1]
        )
        self.assertTrue(
            config.face_sensor_response_bounds[0]
            <= first.face_sensor.response_time_constant
            <= config.face_sensor_response_bounds[1]
        )
        self.assertTrue(
            config.contact_beta_bounds[0]
            <= first.contact_beta
            <= config.contact_beta_bounds[1]
        )

    def test_family_c_is_truth_only_and_nominal_boundary_remains_sensitive(self):
        config = self.small_config()
        self.assertNotIn(TEMPERATURE_DEPENDENT_CONTACT, MODEL_NAMES)
        margins = dict(nominal_realistic_margins(config))
        self.assertLess(margins[FOUR_STATE_MODEL], 0.0)
        self.assertGreater(margins[FIVE_STATE_MODEL], 0.0)
        self.assertGreater(margins[TEMPERATURE_DEPENDENT_CONTACT], 0.0)

    def test_probe_is_prospective_and_removed_from_the_paired_final_target(self):
        config = self.small_config()
        policy_by_name = {item.name: item for item in default_fixed_policies()}
        cases = {
            name: build_realistic_blinded_case(
                STAGE3_TRUTH_CONDITIONS[0],
                0,
                policy,
                config,
            )
            for name, policy in policy_by_name.items()
        }
        initial = tuple(case.acquisition_runs[0] for case in cases.values())
        self.assertTrue(all(item == initial[0] for item in initial[1:]))
        face = cases[FIXED_FACE_TEMPERATURE]
        self.assertFalse(
            face.acquisition_runs[0].instrumentation.temporary_face_sensor
        )
        self.assertTrue(
            face.acquisition_runs[1].instrumentation.temporary_face_sensor
        )
        self.assertTrue(face.verification_run.instrumentation.temporary_face_sensor)
        self.assertFalse(face.final_instrumentation.temporary_face_sensor)
        voltage = cases[FIXED_VOLTAGE]
        self.assertFalse(
            voltage.acquisition_runs[0].instrumentation.temporary_face_sensor
        )
        self.assertNotIn(
            VOLTAGE,
            {item.channel for item in voltage.acquisition_runs[0].observations.values},
        )
        self.assertIn(
            VOLTAGE,
            {item.channel for item in voltage.acquisition_runs[1].observations.values},
        )
        margins = tuple(
            reveal_realistic_operating_outcome(
                STAGE3_TRUTH_CONDITIONS[0],
                0,
                case,
                config,
            ).true_margin
            for case in cases.values()
        )
        self.assertTrue(
            all(math.isclose(value, margins[0], abs_tol=1.0e-12) for value in margins)
        )
        self.assertTrue(all(not hasattr(case, "truth_condition") for case in cases.values()))

    def test_calibrated_offset_cannot_hide_a_time_shaped_voltage_error(self):
        config = self.small_config()
        times = tuple(float(index) for index in range(8))
        prediction = RealisticObservablePrediction(
            values=tuple(ObservableValue(VOLTAGE, time, 2.0) for time in times),
            time=times,
            cold_face=(300.0,) * len(times),
            hot_face=(300.0,) * len(times),
        )

        def run(errors):
            return ObservableRun(
                "voltage",
                default_fixed_policies()[2].additional_regimes[0].current,
                tuple(
                    ObservableValue(VOLTAGE, time, 2.0 + error)
                    for time, error in zip(times, errors)
                ),
            )

        noise = dict(config.sensor.channel_noise)[VOLTAGE]
        constant = _regularized_bias_residuals(
            run((2.0 * noise,) * len(times)), prediction, config
        )
        shaped = _regularized_bias_residuals(
            run(tuple((2.0 if index % 2 else -2.0) * noise for index in range(8))),
            prediction,
            config,
        )
        self.assertLess(
            sum(value * value for value in constant),
            sum(value * value for value in shaped),
        )

    def test_face_fit_keeps_probe_nuisance_but_final_forecast_removes_it(self):
        config = self.small_config()
        case = build_realistic_blinded_case(
            STAGE3_TRUTH_CONDITIONS[0],
            0,
            default_fixed_policies()[-1],
            config,
        )
        fits = fit_realistic_acquisition_models(case, config)
        self.assertFalse(fits.failures)
        expected_dimensions = {FOUR_STATE_MODEL: 6, FIVE_STATE_MODEL: 7}
        for fit in fits.fits:
            self.assertEqual(
                len(fit.log_multipliers),
                expected_dimensions[fit.model_name],
            )
            self.assertIsNotNone(fit.face_sensor)
            changed = list(fit.log_multipliers)
            changed[fit.parameter_names.index("probe_capacitance")] += 0.05
            changed[fit.parameter_names.index("probe_response")] -= 0.05
            self.assertEqual(
                _margin_at_offsets(fit, fit.log_multipliers, case.final_regime, config),
                _margin_at_offsets(fit, changed, case.final_regime, config),
            )

    def test_verification_observations_cannot_change_acquisition_fits(self):
        config = self.small_config()
        case = build_realistic_blinded_case(
            STAGE3_TRUTH_CONDITIONS[0],
            0,
            default_fixed_policies()[0],
            config,
        )
        shifted_observations = case.verification_run.observations._replace(
            values=tuple(
                item._replace(value=item.value + 100.0)
                for item in case.verification_run.observations.values
            )
        )
        shifted = replace(
            case,
            verification_run=RealisticOperatingRun(
                case.verification_run.regime,
                shifted_observations,
                case.verification_run.instrumentation,
            ),
        )
        original_fits = fit_realistic_acquisition_models(case, config)
        shifted_fits = fit_realistic_acquisition_models(shifted, config)
        self.assertEqual(original_fits, shifted_fits)
        self.assertEqual(
            {fit.model_name: len(fit.log_multipliers) for fit in original_fits.fits},
            {FOUR_STATE_MODEL: 4, FIVE_STATE_MODEL: 5},
        )
        self.assertTrue(
            all("series_resistance" in fit.parameter_names for fit in original_fits.fits)
        )

    def test_one_saved_decision_can_be_scored_only_against_its_reveal(self):
        config = self.small_config()
        case = build_realistic_blinded_case(
            TEMPERATURE_DEPENDENT_CONTACT,
            0,
            default_fixed_policies()[0],
            config,
        )
        saved = decide_realistic_blinded_case(case, config=config)
        revealed = reveal_realistic_operating_outcome(
            TEMPERATURE_DEPENDENT_CONTACT,
            0,
            case,
            config,
        )
        scored = score_realistic_saved_decision(saved, revealed, case, config)
        self.assertEqual(scored.saved, saved)
        self.assertEqual(scored.true_margin, revealed.true_margin)
        with self.assertRaisesRegex(ValueError, "share identity"):
            score_realistic_saved_decision(
                saved,
                revealed._replace(
                    case_id=revealed.case_id._replace(device_token="wrong")
                ),
                case,
                config,
            )

    def test_small_runner_returns_all_paired_policy_truth_cells(self):
        config = self.small_config()
        result = run_operating_decision_realism(config)
        self.assertEqual(len(result.trials), 12)
        self.assertEqual(len(result.summaries), 12)
        self.assertEqual(
            {
                (summary.truth_condition, summary.policy_name)
                for summary in result.summaries
            },
            {
                (truth, policy.name)
                for truth in STAGE3_TRUTH_CONDITIONS
                for policy in default_fixed_policies()
            },
        )
        report = format_operating_decision_realism_report(result)
        self.assertIn("Stage 3 realism stress test", report)
        self.assertIn("Temperature-dependent-contact truth", report)
        self.assertIn("temporary face probe is removed", report)


if __name__ == "__main__":
    unittest.main()
