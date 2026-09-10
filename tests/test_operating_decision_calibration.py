import math
import unittest
from dataclasses import replace
from typing import NamedTuple

from thermotwin.studies.operating_decision import (
    APPROVE,
    CandidateVerification,
    FIXED_VOLTAGE,
    INSUFFICIENT_EVIDENCE,
    STOP_NOW,
    MarginEnvelope,
    SavedOperatingDecision,
    ScoredOperatingDecision,
    default_fixed_policies,
    RateEstimate,
)
from thermotwin.reports.operating_decision_calibration import _format_rate
from thermotwin.studies.operating_decision_calibration import (
    CostScenario,
    InitialDecisionSignal,
    OperatingDecisionCalibrationConfig,
    ProcedureOutcome,
    SelectorRule,
    _selector_choices,
    apply_margin_padding,
    block_conformal_padding,
    conformal_rank,
    expected_loss,
    initial_decision_signal,
    matched_pipeline_verification_gates,
    select_fixed_policy,
)
from thermotwin.studies.operating_decision_realism import (
    OperatingDecisionRealismConfig,
    Stage3CaseId,
    build_realistic_blinded_case,
    fit_realistic_acquisition_models,
)
from thermotwin.studies.sensor_model_discrimination import MODEL_NAMES


class _GateFit(NamedTuple):
    model_name: str
    reached_bound: bool = False


class OperatingDecisionCalibrationTests(unittest.TestCase):
    @staticmethod
    def _scored_decision(
        *,
        lower=0.10,
        upper=0.20,
        true_margin=0.15,
        decision=APPROVE,
    ):
        saved = SavedOperatingDecision(
            case_id=Stage3CaseId(
                "test-device",
                STOP_NOW,
                0,
                "stage3_final_v1",
            ),
            decision=decision,
            decision_reason="test",
            margin_envelope=MarginEnvelope(lower, upper),
            model_intervals=(),
            verifications=(),
            failures=(),
            decision_computation_seconds=0.1,
        )
        true_pass = true_margin >= 0.0
        return ScoredOperatingDecision(
            truth_condition="matched_four_state",
            saved=saved,
            true_margin=true_margin,
            true_pass=true_pass,
            false_approval=decision == APPROVE and not true_pass,
            false_rejection=False,
            interval_covered=lower <= true_margin <= upper,
            acquisition_run_count=1,
            diagnostic_run_count=2,
            energized_schedule_time_seconds=160.0,
            acquisition_energy=10.0,
            verification_energy=5.0,
            total_diagnostic_energy=15.0,
            extra_sensor_count=0,
        )

    def test_default_stage4_partitions_are_disjoint_and_reserve_evaluation(self):
        config = OperatingDecisionCalibrationConfig()
        self.assertEqual(config.calibration.sensor.trial_count, 20)
        self.assertEqual(config.gate_development.sensor.trial_count, 10)
        self.assertEqual(config.rehearsal.sensor.trial_count, 10)
        self.assertEqual(config.reserved_evaluation_trial_count, 50)
        self.assertEqual(config.calibration.sensor.first_seed, 10_191_001)
        self.assertEqual(config.rehearsal.sensor.first_seed, 20_191_001)
        self.assertEqual(config.reserved_evaluation_first_seed, 30_191_001)

        with self.assertRaisesRegex(ValueError, "seed namespaces overlap"):
            replace(
                config,
                rehearsal=replace(
                    config.rehearsal,
                    sensor=replace(
                        config.rehearsal.sensor,
                        first_seed=config.calibration.sensor.first_seed,
                    ),
                ),
            )

    def test_block_conformal_rank_uses_trial_blocks_not_family_rows(self):
        scores = tuple(float(index) for index in range(20))
        rank, padding = block_conformal_padding(scores, 0.90)
        self.assertEqual(rank, 19)
        self.assertEqual(padding, 18.0)
        self.assertEqual(conformal_rank(20, 0.90), 19)
        with self.assertRaisesRegex(ValueError, "too few calibration blocks"):
            conformal_rank(2, 0.90)

    def test_padding_can_only_widen_an_emitted_interval(self):
        raw = self._scored_decision()
        calibrated = apply_margin_padding(raw, 0.15)
        self.assertEqual(calibrated.saved.decision, INSUFFICIENT_EVIDENCE)
        self.assertAlmostEqual(calibrated.saved.margin_envelope.lower, -0.05)
        self.assertAlmostEqual(calibrated.saved.margin_envelope.upper, 0.35)
        self.assertLessEqual(
            calibrated.saved.margin_envelope.lower,
            raw.saved.margin_envelope.lower,
        )
        self.assertGreaterEqual(
            calibrated.saved.margin_envelope.upper,
            raw.saved.margin_envelope.upper,
        )
        self.assertFalse(calibrated.false_approval)
        self.assertFalse(calibrated.false_rejection)

    def test_selector_has_only_acquisition_interval_as_a_decision_signal(self):
        resolved = InitialDecisionSignal(
            candidate_models=tuple(sorted(MODEL_NAMES)),
            failed_fit_count=0,
            bound_hit_count=0,
            margin_envelope=MarginEnvelope(0.1, 0.3),
        )
        unresolved = resolved._replace(margin_envelope=MarginEnvelope(-0.1, 0.3))
        failed = resolved._replace(failed_fit_count=1, margin_envelope=None)
        self.assertEqual(select_fixed_policy(resolved), STOP_NOW)
        self.assertEqual(select_fixed_policy(unresolved), FIXED_VOLTAGE)
        self.assertEqual(select_fixed_policy(failed), FIXED_VOLTAGE)
        forbidden = {
            "truth_condition",
            "device_token",
            "trial_index",
            "verification_run",
            "true_margin",
            "final_response",
        }
        self.assertFalse(forbidden.intersection(InitialDecisionSignal._fields))

    def test_actual_initial_signal_contains_both_models_and_no_hidden_fields(self):
        base = OperatingDecisionRealismConfig()
        config = replace(
            base,
            sensor=replace(base.sensor, trial_count=1, fit_iterations=1),
        )
        case = build_realistic_blinded_case(
            "matched_four_state",
            0,
            default_fixed_policies()[0],
            config,
        )
        fit_set = fit_realistic_acquisition_models(case, config)
        signal = initial_decision_signal(fit_set, case.final_regime, config)
        self.assertEqual(set(signal.candidate_models), set(MODEL_NAMES))
        self.assertEqual(signal.failed_fit_count, 0)
        self.assertFalse(hasattr(signal, "truth_condition"))
        self.assertFalse(hasattr(signal, "trial_index"))

        indexed = {}
        for truth in (
            "matched_four_state",
            "extra_interface_mass",
            "temperature_dependent_contact",
        ):
            for policy in default_fixed_policies():
                verifications = tuple(
                    CandidateVerification(fit, 1.0, True, None)
                    for fit in fit_set.fits
                )
                saved = SavedOperatingDecision(
                    case_id=Stage3CaseId(
                        f"selector-{truth}",
                        policy.name,
                        0,
                        "stage3_final_v1",
                    ),
                    decision=INSUFFICIENT_EVIDENCE,
                    decision_reason="selector-test",
                    margin_envelope=None,
                    model_intervals=(),
                    verifications=verifications,
                    failures=(),
                    decision_computation_seconds=0.0,
                )
                indexed[(truth, 0, policy.name)] = ScoredOperatingDecision(
                    truth_condition=truth,
                    saved=saved,
                    true_margin=123.0,
                    true_pass=True,
                    false_approval=False,
                    false_rejection=False,
                    interval_covered=None,
                    acquisition_run_count=1 + len(policy.additional_regimes),
                    diagnostic_run_count=2 + len(policy.additional_regimes),
                    energized_schedule_time_seconds=160.0,
                    acquisition_energy=10.0,
                    verification_energy=5.0,
                    total_diagnostic_energy=15.0,
                    extra_sensor_count=policy.extra_sensor_count,
                )
        original_choices = _selector_choices(indexed, config, SelectorRule())
        mutated = {}
        for key, row in indexed.items():
            mutated_saved = row.saved._replace(
                verifications=(
                    tuple(
                        verification._replace(normalized_score=1.0e9)
                        for verification in row.saved.verifications
                    )
                    if key[2] == STOP_NOW
                    else ()
                )
            )
            mutated[key] = row._replace(
                saved=mutated_saved,
                true_margin=-1.0e9,
                true_pass=False,
            )
        self.assertEqual(
            original_choices,
            _selector_choices(mutated, config, SelectorRule()),
        )

    @staticmethod
    def _gate_trials(config):
        rows = []
        for truth_index, truth in enumerate(
            (
                "matched_four_state",
                "extra_interface_mass",
                "temperature_dependent_contact",
            )
        ):
            for trial_index in range(config.gate_development.sensor.trial_count):
                for policy_index, policy in enumerate(default_fixed_policies()):
                    verifications = tuple(
                        CandidateVerification(
                            _GateFit(model_name),
                            1.02
                            + 0.01 * policy_index
                            + 0.02 * trial_index
                            + 0.005 * truth_index,
                            True,
                            None,
                        )
                        for model_name in MODEL_NAMES
                    )
                    saved = SavedOperatingDecision(
                        case_id=Stage3CaseId(
                            f"gate-{truth}-{trial_index}",
                            policy.name,
                            trial_index,
                            "stage3_final_v1",
                        ),
                        decision=INSUFFICIENT_EVIDENCE,
                        decision_reason="gate-test",
                        margin_envelope=None,
                        model_intervals=(),
                        verifications=verifications,
                        failures=(),
                        decision_computation_seconds=0.0,
                    )
                    rows.append(
                        ScoredOperatingDecision(
                            truth_condition=truth,
                            saved=saved,
                            true_margin=0.01 * trial_index,
                            true_pass=True,
                            false_approval=False,
                            false_rejection=False,
                            interval_covered=None,
                            acquisition_run_count=1 + len(policy.additional_regimes),
                            diagnostic_run_count=2 + len(policy.additional_regimes),
                            energized_schedule_time_seconds=160.0,
                            acquisition_energy=10.0,
                            verification_energy=5.0,
                            total_diagnostic_energy=15.0,
                            extra_sensor_count=policy.extra_sensor_count,
                        )
                    )
        return tuple(rows)

    def test_matched_pipeline_gate_is_reproducible_and_keeps_noise_floor(self):
        defaults = OperatingDecisionCalibrationConfig()
        gate_development = replace(
            defaults.gate_development,
            sensor=replace(defaults.gate_development.sensor, trial_count=2),
        )
        config = replace(
            defaults,
            gate_development=gate_development,
            gate_target_block_retention=0.66,
            gate_monte_carlo_draws=1_000,
        )
        trials = self._gate_trials(config)
        first = matched_pipeline_verification_gates(trials, config)
        second = matched_pipeline_verification_gates(trials, config)
        self.assertEqual(first, second)
        by_policy = {item.policy_name: item for item in first}
        self.assertGreater(
            by_policy[FIXED_VOLTAGE].observation_count,
            by_policy[STOP_NOW].observation_count,
        )
        self.assertTrue(
            all(item.threshold >= item.noise_reference_threshold for item in first)
        )
        self.assertTrue(all(item.matched_development_rank == 2 for item in first))

    def test_expected_loss_keeps_errors_dominant_and_resources_visible(self):
        outcome = ProcedureOutcome(
            split="rehearsal",
            truth_condition="matched_four_state",
            trial_index=0,
            procedure_name="decision_directed_selector",
            selected_policy=FIXED_VOLTAGE,
            decision=INSUFFICIENT_EVIDENCE,
            true_margin=-0.1,
            true_pass=False,
            false_approval=False,
            false_rejection=False,
            raw_interval_lower=-0.2,
            raw_interval_upper=0.1,
            calibrated_interval_lower=-0.3,
            calibrated_interval_upper=0.2,
            raw_interval_covered=True,
            calibrated_interval_covered=True,
            additive_margin_padding=0.1,
            verified_candidate_count=2,
            score_rejected_candidate_count=0,
            numerical_failure_count=0,
            acquisition_run_count=2,
            diagnostic_run_count=3,
            energized_schedule_time_seconds=240.0,
            total_diagnostic_energy=22.5,
            extra_sensor_count=1,
            decision_computation_seconds=0.1,
        )
        scenario = CostScenario("test")
        loss = expected_loss(outcome, scenario, stop_energy=15.0)
        self.assertTrue(math.isclose(loss, 1.25))
        unsafe = outcome._replace(
            decision=APPROVE,
            false_approval=True,
        )
        self.assertGreater(
            expected_loss(unsafe, scenario, stop_energy=15.0),
            loss + 90.0,
        )

    def test_report_formats_clustered_or_calibration_rates_without_fake_ci(self):
        text = _format_rate(RateEstimate(1, 2, 0.5, None, None))
        self.assertIn("50.0%", text)
        self.assertIn("CI suppressed", text)


if __name__ == "__main__":
    unittest.main()
