import unittest

from thermotwin.reports.operating_decision import (
    format_operating_decision_report,
)
from thermotwin.studies.operating_decision import (
    APPROVE,
    FINAL_EVALUATION,
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    INSUFFICIENT_EVIDENCE,
    REJECT,
    STOP_NOW,
    BlindedOperatingCase,
    MarginEnvelope,
    OperatingDecisionConfig,
    OperatingRegime,
    OperatingRun,
    RevealedOperatingOutcome,
    SavedOperatingDecision,
    build_blinded_operating_case,
    classify_margin_envelope,
    default_fixed_policies,
    final_operating_current,
    fit_acquisition_models,
    nominal_operating_margins,
    rate_estimate,
    run_operating_decision,
    score_saved_decision,
    verification_score,
)
from thermotwin.studies.sensor_model_discrimination import (
    COLD_EXCHANGER,
    COLD_FACE,
    HOT_EXCHANGER,
    VOLTAGE,
    ObservableRun,
    ObservableValue,
    SensorDiscriminationConfig,
)


class OperatingDecisionTests(unittest.TestCase):
    @staticmethod
    def small_config(fit_iterations=1):
        return OperatingDecisionConfig(
            sensor=SensorDiscriminationConfig(
                trial_count=1,
                first_seed=191_001,
                fit_iterations=fit_iterations,
            )
        )

    @classmethod
    def setUpClass(cls):
        cls.config = cls.small_config()
        cls.result = run_operating_decision(cls.config)

    def test_frozen_final_schedule_straddles_nominal_model_families(self):
        current = final_operating_current()
        self.assertEqual(current.transition_times, (5.0, 25.0, 38.0, 58.0))
        self.assertEqual(current.values, (0.0, 1.0, 0.0, -0.8, 0.0))

        margins = dict(nominal_operating_margins(self.config))
        self.assertLess(margins["four_state"], 0.0)
        self.assertGreater(margins["five_state"], 0.0)

    def test_margin_envelope_decision_boundaries_are_exact(self):
        self.assertEqual(
            classify_margin_envelope(MarginEnvelope(0.0, 0.2)), APPROVE
        )
        self.assertEqual(
            classify_margin_envelope(MarginEnvelope(-0.2, -1.0e-12)), REJECT
        )
        self.assertEqual(
            classify_margin_envelope(MarginEnvelope(-0.2, 0.0)),
            INSUFFICIENT_EVIDENCE,
        )
        self.assertEqual(classify_margin_envelope(None), INSUFFICIENT_EVIDENCE)

    def test_added_channels_are_prospective_and_cases_are_paired(self):
        cases = {
            policy.name: build_blinded_operating_case(
                "matched_four_state", 0, policy, self.config
            )
            for policy in default_fixed_policies()
        }
        initial_runs = tuple(case.acquisition_runs[0] for case in cases.values())
        self.assertTrue(all(run == initial_runs[0] for run in initial_runs[1:]))
        self.assertEqual(
            len({case.case_id.device_token for case in cases.values()}),
            1,
        )
        self.assertTrue(
            all(not hasattr(case, "truth_condition") for case in cases.values())
        )
        self.assertEqual(
            initial_runs[0].regime.channels,
            (COLD_EXCHANGER, HOT_EXCHANGER),
        )
        self.assertNotIn(
            VOLTAGE,
            cases[FIXED_VOLTAGE].acquisition_runs[0].regime.channels,
        )
        self.assertIn(
            VOLTAGE,
            cases[FIXED_VOLTAGE].acquisition_runs[1].regime.channels,
        )
        self.assertNotIn(
            COLD_FACE,
            cases[FIXED_FACE_TEMPERATURE].acquisition_runs[0].regime.channels,
        )
        self.assertIn(
            COLD_FACE,
            cases[FIXED_FACE_TEMPERATURE].acquisition_runs[1].regime.channels,
        )
        self.assertEqual(
            {name: len(case.acquisition_runs) for name, case in cases.items()},
            {
                STOP_NOW: 1,
                FIXED_THERMAL: 4,
                FIXED_VOLTAGE: 2,
                FIXED_FACE_TEMPERATURE: 2,
            },
        )
        for case in cases.values():
            self.assertEqual(case.final_regime.phase, FINAL_EVALUATION)
            self.assertEqual(case.final_regime.channels, ())
            self.assertFalse(hasattr(case, "final_observations"))

    def test_verification_cannot_change_acquisition_fits(self):
        policy = default_fixed_policies()[0]
        case = build_blinded_operating_case(
            "matched_four_state", 0, policy, self.config
        )
        shifted_values = tuple(
            ObservableValue(item.channel, item.time, item.value + 100.0)
            for item in case.verification_run.observations.values
        )
        shifted_observations = case.verification_run.observations._replace(
            values=shifted_values
        )
        shifted_case = BlindedOperatingCase(
            case_id=case.case_id,
            policy=case.policy,
            acquisition_runs=case.acquisition_runs,
            verification_run=OperatingRun(
                case.verification_run.regime, shifted_observations
            ),
            final_regime=case.final_regime,
        )

        original_fits = fit_acquisition_models(case, self.config)
        shifted_fits = fit_acquisition_models(shifted_case, self.config)
        self.assertEqual(original_fits, shifted_fits)

    def test_verification_uses_declared_bias_covariance_without_hiding_shape(self):
        policy = default_fixed_policies()[0]
        case = build_blinded_operating_case(
            "matched_four_state", 0, policy, self.config
        )
        fit = fit_acquisition_models(case, self.config).fits[0]
        _, prediction = verification_score(
            fit, case.verification_run, self.config
        )
        noise = dict(self.config.sensor.channel_noise)
        constant_values = tuple(
            ObservableValue(
                item.channel,
                item.time,
                item.value + 2.0 * noise[item.channel],
            )
            for item in prediction.values
        )
        shape_values = tuple(
            ObservableValue(
                item.channel,
                item.time,
                item.value
                + (2.0 if int(round(item.time)) % 2 == 0 else -2.0)
                * noise[item.channel],
            )
            for item in prediction.values
        )

        def with_values(values):
            observations = ObservableRun(
                name=case.verification_run.regime.name,
                current=case.verification_run.regime.current,
                values=values,
            )
            return OperatingRun(case.verification_run.regime, observations)

        constant_score, _ = verification_score(
            fit, with_values(constant_values), self.config
        )
        shape_score, _ = verification_score(
            fit, with_values(shape_values), self.config
        )
        self.assertLess(constant_score, self.config.verification_score_threshold)
        self.assertGreater(shape_score, self.config.verification_score_threshold)

    def test_revealed_truth_only_changes_postdecision_scoring(self):
        policy = default_fixed_policies()[0]
        case = build_blinded_operating_case(
            "matched_four_state", 0, policy, self.config
        )
        saved = SavedOperatingDecision(
            case_id=case.case_id,
            decision=APPROVE,
            decision_reason="verified_envelope_inside_band",
            margin_envelope=MarginEnvelope(0.1, 0.3),
            model_intervals=(),
            verifications=(),
            failures=(),
            decision_computation_seconds=0.0,
        )
        passing = RevealedOperatingOutcome(
            case.case_id, "matched_four_state", (301.5,), 0.2
        )
        violating = RevealedOperatingOutcome(
            case.case_id, "matched_four_state", (302.0,), -0.3
        )

        passing_score = score_saved_decision(
            saved, passing, case, self.config
        )
        violating_score = score_saved_decision(
            saved, violating, case, self.config
        )
        self.assertIs(passing_score.saved, saved)
        self.assertIs(violating_score.saved, saved)
        self.assertFalse(passing_score.false_approval)
        self.assertTrue(violating_score.false_approval)

        with self.assertRaisesRegex(ValueError, "margin does not match"):
            score_saved_decision(
                saved,
                passing._replace(true_margin=0.1),
                case,
                self.config,
            )
        other_case = build_blinded_operating_case(
            "matched_four_state", 0, default_fixed_policies()[1], self.config
        )
        with self.assertRaisesRegex(ValueError, "share identity"):
            score_saved_decision(saved, passing, other_case, self.config)

    def test_pilot_is_paired_and_accounts_for_every_diagnostic_run(self):
        self.assertEqual(len(self.result.trials), 8)
        expected_runs = {
            STOP_NOW: (1, 2),
            FIXED_THERMAL: (4, 5),
            FIXED_VOLTAGE: (2, 3),
            FIXED_FACE_TEMPERATURE: (2, 3),
        }
        for trial in self.result.trials:
            self.assertEqual(
                (
                    trial.acquisition_run_count,
                    trial.diagnostic_run_count,
                ),
                expected_runs[trial.saved.policy_name],
            )
            self.assertAlmostEqual(
                trial.total_diagnostic_energy,
                trial.acquisition_energy + trial.verification_energy,
            )
        for condition in ("matched_four_state", "extra_interface_mass"):
            margins = {
                trial.true_margin
                for trial in self.result.trials
                if trial.truth_condition == condition
            }
            self.assertEqual(len(margins), 1)

    def test_report_states_split_cost_and_synthetic_boundary(self):
        text = format_operating_decision_report(self.result)

        self.assertIn("final response is generated only after the decision", text)
        self.assertIn("modeled net diagnostic electrical energy", text)
        self.assertIn("N/A (0/0)", text)
        self.assertIn("not hardware validation", text)
        self.assertIn("decision-directed selector remain later", text)

    def test_empty_rate_denominator_is_not_reported_as_zero_error(self):
        empty = rate_estimate(0, 0)
        self.assertIsNone(empty.rate)
        self.assertIsNone(empty.lower_95)
        self.assertIsNone(empty.upper_95)

    def test_final_response_cannot_be_wrapped_as_a_fitting_run(self):
        regime = OperatingRegime(
            "final",
            FINAL_EVALUATION,
            final_operating_current(),
            (),
        )
        observations = ObservableRun("final", final_operating_current(), ())
        with self.assertRaisesRegex(ValueError, "final-evaluation response"):
            OperatingRun(regime, observations)

    def test_acquisition_policy_rejects_a_nonacquisition_addition(self):
        verification = OperatingRegime(
            "bad",
            "verification",
            final_operating_current(),
            (COLD_EXCHANGER, HOT_EXCHANGER),
        )
        from thermotwin.studies.operating_decision import FixedPolicy

        with self.assertRaisesRegex(ValueError, "acquisition regimes"):
            FixedPolicy(STOP_NOW, (verification,), 0)


if __name__ == "__main__":
    unittest.main()
