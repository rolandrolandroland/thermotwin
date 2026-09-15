import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thermotwin.studies.operating_decision import (
    APPROVE,
    INSUFFICIENT_EVIDENCE,
    CandidateVerification,
    MarginInterval,
    NumericalFailure,
    SavedOperatingDecision,
    ScoredOperatingDecision,
    default_fixed_policies,
)
from thermotwin.studies.operating_decision_calibration import (
    STAGE4_PROCEDURES,
    ProcedureCalibration,
    VerificationGate,
    block_conformal_padding,
)
from thermotwin.studies.operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
    RealisticCandidateFit,
    Stage3CaseId,
)
from thermotwin.studies.operating_decision_replication import (
    CorrectedPartition,
    corrected_partition_config,
)
from thermotwin.studies.operating_decision_replication_calibration import (
    CorrectedParentCalibrationConfig,
    build_corrected_parent_artifact,
    calibrate_corrected_parent_procedures,
    calibrate_corrected_verification_gates,
    corrected_parent_artifact_from_payload,
    corrected_parent_artifact_payload,
    load_corrected_parent_artifact,
    rebuild_with_corrected_gate,
    save_corrected_parent_artifact,
    validate_corrected_parent_artifact,
)
from thermotwin.studies.operating_decision_replication_protocol import (
    CorrectedReplicationPlan,
    build_corrected_generator_freeze,
)
from thermotwin.studies.sensor_model_discrimination import (
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
)


class OperatingDecisionReplicationCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.partition = CorrectedPartition("r2_gate_development", 10, "unit_campaign")
        self.physical = corrected_partition_config(
            self.partition,
            OperatingDecisionRealismConfig(),
        )
        self.config = CorrectedParentCalibrationConfig(gate_monte_carlo_draws=1_000)

    @staticmethod
    def fit(model_name, *, converged=True):
        count = 4 if model_name == FOUR_STATE_MODEL else 5
        return RealisticCandidateFit(
            model_name=model_name,
            log_multipliers=(0.0,) * count,
            parameter_names=tuple(f"p{index}" for index in range(count)),
            physical_values=(0.25, 50.0, 1.5),
            interface_mass=None if model_name == FOUR_STATE_MODEL else 20.0,
            series_resistance=0.1,
            face_sensor=None,
            objective=1.0,
            covariance=tuple(
                tuple(1.0 if row == column else 0.0 for column in range(count))
                for row in range(count)
            ),
            reached_bound=False,
            evaluation_count=20,
            converged=converged,
            termination_reason=(
                "scaled_projected_gradient_tolerance"
                if converged
                else "fixed_iteration_limit"
            ),
            completed_iterations=6,
            accepted_iterations=5,
            scaled_gradient_infinity_norm=1.0e-6 if converged else 0.1,
            last_step_infinity_norm=1.0e-7,
            last_relative_objective_reduction=1.0e-10,
        )

    def trials(self, *, converged=True):
        rows = []
        for truth_index, truth in enumerate(STAGE3_TRUTH_CONDITIONS):
            for block in range(self.partition.block_count):
                for policy_index, policy in enumerate(default_fixed_policies()):
                    fits = (self.fit(FOUR_STATE_MODEL, converged=converged), self.fit(FIVE_STATE_MODEL, converged=converged))
                    verifications = tuple(
                        CandidateVerification(
                            fit,
                            0.80 + 0.01 * truth_index + 0.001 * block + 0.02 * policy_index,
                            converged,
                            None if converged else "optimizer_not_converged",
                        )
                        for fit in fits
                    )
                    saved = SavedOperatingDecision(
                        case_id=Stage3CaseId(
                            f"device-{truth}-{block}",
                            policy.name,
                            block,
                            "operating_decision_corrected_final_v2",
                        ),
                        decision=INSUFFICIENT_EVIDENCE,
                        decision_reason="unit",
                        margin_envelope=None,
                        model_intervals=(),
                        verifications=verifications,
                        failures=(),
                        decision_computation_seconds=0.1,
                    )
                    rows.append(
                        ScoredOperatingDecision(
                            truth_condition=truth,
                            saved=saved,
                            true_margin=0.1 + block * 0.001,
                            true_pass=True,
                            false_approval=False,
                            false_rejection=False,
                            interval_covered=None,
                            acquisition_run_count=1 + len(policy.additional_regimes),
                            diagnostic_run_count=2 + len(policy.additional_regimes),
                            energized_schedule_time_seconds=160.0,
                            acquisition_energy=50.0,
                            verification_energy=20.0,
                            total_diagnostic_energy=70.0,
                            extra_sensor_count=policy.extra_sensor_count,
                        )
                    )
        return tuple(rows)

    def test_gate_calibration_uses_semantic_streams_and_matched_blocks(self):
        gates, keys = calibrate_corrected_verification_gates(
            self.trials(),
            self.physical,
            self.partition,
            self.config,
        )
        self.assertEqual(len(gates), 4)
        self.assertEqual(len(set(keys)), 4)
        for gate, key in zip(gates, keys):
            self.assertEqual(gate.matched_development_block_count, 10)
            self.assertEqual(gate.matched_development_rank, 10)
            self.assertEqual(key.campaign, "unit_campaign")
            self.assertEqual(key.partition, "r2_gate_development")
            self.assertTrue(math.isfinite(gate.threshold))

    def test_gate_calibration_fails_closed_on_unreliable_matched_fit(self):
        cases = (
            {"converged": False, "reached_bound": False, "score": 0.8},
            {"converged": True, "reached_bound": True, "score": 0.8},
            {"converged": True, "reached_bound": False, "score": math.nan},
            {"converged": True, "reached_bound": False, "score": math.inf},
        )
        for case in cases:
            with self.subTest(case=case):
                rows = list(self.trials())
                trial = rows[0]
                verifications = list(trial.saved.verifications)
                matched = verifications[0]
                fit = matched.fit._replace(
                    converged=case["converged"],
                    reached_bound=case["reached_bound"],
                )
                verifications[0] = matched._replace(
                    fit=fit,
                    normalized_score=case["score"],
                )
                rows[0] = trial._replace(
                    saved=trial.saved._replace(verifications=tuple(verifications))
                )
                with self.assertRaisesRegex(ValueError, "unreliable"):
                    calibrate_corrected_verification_gates(
                        rows,
                        self.physical,
                        self.partition,
                        self.config,
                    )

    def test_corrected_gate_rebuild_excludes_unreliable_candidate(self):
        gate = VerificationGate(
            policy_name="stop_now",
            observation_count=42,
            matched_development_block_count=10,
            matched_development_rank=10,
            target_block_retention=0.90,
            matched_block_scores=(1.0,) * 10,
            noise_reference_quantile=0.99,
            noise_reference_threshold=1.0,
            threshold=2.0,
            monte_carlo_draws=1_000,
            seed=1,
        )
        base = self.trials()[0]
        cases = (
            (False, False, 0.8, "optimizer_not_converged"),
            (True, True, 0.8, "fit_reached_bound"),
            (True, True, math.nan, "fit_reached_bound"),
        )
        interval = MarginInterval(FIVE_STATE_MODEL, 0.2, 0.01, 0.18, 0.22)
        for converged, reached_bound, score, expected_reason in cases:
            with self.subTest(
                converged=converged,
                reached_bound=reached_bound,
                score=score,
            ):
                verifications = list(base.saved.verifications)
                original = verifications[0]
                verifications[0] = original._replace(
                    fit=original.fit._replace(
                        converged=converged,
                        reached_bound=reached_bound,
                    ),
                    normalized_score=score,
                )
                trial = base._replace(
                    saved=base.saved._replace(
                        verifications=tuple(verifications),
                        failures=(),
                    )
                )
                with patch(
                    "thermotwin.studies.operating_decision_replication_calibration.forecast_realistic_margin_interval",
                    return_value=interval,
                ):
                    rebuilt = rebuild_with_corrected_gate(
                        trial,
                        gate,
                        self.physical,
                    )
                self.assertEqual(rebuilt.saved.decision, APPROVE)
                self.assertEqual(
                    rebuilt.saved.decision_reason,
                    "corrected_verified_envelope_inside_band",
                )
                self.assertEqual(rebuilt.saved.margin_envelope, (0.18, 0.22))
                self.assertEqual(rebuilt.saved.model_intervals, (interval,))
                self.assertEqual(rebuilt.saved.failures, ())
                self.assertFalse(rebuilt.saved.verifications[0].passed)
                self.assertEqual(
                    rebuilt.saved.verifications[0].failure_reason,
                    expected_reason,
                )
                self.assertTrue(rebuilt.saved.verifications[1].passed)

    def test_corrected_gate_rebuild_keeps_verification_failures_case_fatal(self):
        gate = VerificationGate(
            policy_name="stop_now",
            observation_count=42,
            matched_development_block_count=10,
            matched_development_rank=10,
            target_block_retention=0.90,
            matched_block_scores=(1.0,) * 10,
            noise_reference_quantile=0.99,
            noise_reference_threshold=1.0,
            threshold=2.0,
            monte_carlo_draws=1_000,
            seed=1,
        )
        base = self.trials()[0]
        interval = MarginInterval(FIVE_STATE_MODEL, 0.2, 0.01, 0.18, 0.22)
        for converged, reached_bound, score, original_reason, expected_reason in (
            (True, False, math.nan, "nonfinite_score", "nonfinite_score"),
            (False, False, math.nan, "nonfinite_score", "nonfinite_score"),
            (
                True,
                True,
                math.inf,
                "verification_failure",
                "verification_failure",
            ),
        ):
            with self.subTest(
                converged=converged,
                reached_bound=reached_bound,
                original_reason=original_reason,
            ):
                verifications = list(base.saved.verifications)
                first = verifications[0]
                verifications[0] = first._replace(
                    fit=first.fit._replace(
                        converged=converged,
                        reached_bound=reached_bound,
                    ),
                    normalized_score=score,
                    passed=False,
                    failure_reason=original_reason,
                )
                trial = base._replace(
                    saved=base.saved._replace(
                        verifications=tuple(verifications),
                        failures=(),
                    )
                )
                with patch(
                    "thermotwin.studies.operating_decision_replication_calibration.forecast_realistic_margin_interval",
                    return_value=interval,
                ):
                    rebuilt = rebuild_with_corrected_gate(trial, gate, self.physical)
                self.assertEqual(rebuilt.saved.decision, INSUFFICIENT_EVIDENCE)
                self.assertEqual(
                    rebuilt.saved.decision_reason,
                    "verification_failure",
                )
                self.assertIsNone(rebuilt.saved.margin_envelope)
                self.assertEqual(rebuilt.saved.model_intervals, ())
                self.assertEqual(
                    rebuilt.saved.failures,
                    (
                        NumericalFailure(
                            FOUR_STATE_MODEL,
                            "verification",
                            expected_reason,
                        ),
                    ),
                )

    def test_corrected_gate_rebuild_abstains_when_all_candidates_are_excluded(self):
        gate = VerificationGate(
            policy_name="stop_now",
            observation_count=42,
            matched_development_block_count=10,
            matched_development_rank=10,
            target_block_retention=0.90,
            matched_block_scores=(1.0,) * 10,
            noise_reference_quantile=0.99,
            noise_reference_threshold=1.0,
            threshold=2.0,
            monte_carlo_draws=1_000,
            seed=1,
        )
        base = self.trials()[0]
        first, second = base.saved.verifications
        trial = base._replace(
            saved=base.saved._replace(
                verifications=(
                    first._replace(fit=first.fit._replace(converged=False)),
                    second._replace(fit=second.fit._replace(reached_bound=True)),
                ),
                failures=(),
            )
        )
        with patch(
            "thermotwin.studies.operating_decision_replication_calibration.forecast_realistic_margin_interval"
        ) as forecast:
            rebuilt = rebuild_with_corrected_gate(trial, gate, self.physical)
        forecast.assert_not_called()
        self.assertEqual(rebuilt.saved.decision, INSUFFICIENT_EVIDENCE)
        self.assertEqual(
            rebuilt.saved.decision_reason,
            "no_corrected_verified_candidate",
        )
        self.assertIsNone(rebuilt.saved.margin_envelope)
        self.assertEqual(rebuilt.saved.model_intervals, ())
        self.assertEqual(rebuilt.saved.failures, ())
        self.assertEqual(
            tuple(item.failure_reason for item in rebuilt.saved.verifications),
            ("optimizer_not_converged", "fit_reached_bound"),
        )

    def test_all_nonconverged_candidates_emit_no_parent_calibration_intervals(self):
        gates, _ = calibrate_corrected_verification_gates(
            self.trials(),
            self.physical,
            self.partition,
            self.config,
        )
        calibration_partition = CorrectedPartition(
            "r2_parent_calibration",
            10,
            "unit_campaign",
        )
        calibration_physical = corrected_partition_config(
            calibration_partition,
            OperatingDecisionRealismConfig(),
        )
        calibrations = calibrate_corrected_parent_procedures(
            self.trials(converged=False),
            gates,
            calibration_physical,
            calibration_partition,
            self.config,
        )
        self.assertEqual({item.procedure_name for item in calibrations}, set(STAGE4_PROCEDURES))
        self.assertTrue(all(item.additive_margin_padding == 0.0 for item in calibrations))
        self.assertTrue(all(item.block_nonconformity == (0.0,) * 10 for item in calibrations))
        self.assertTrue(all(item.emitted_interval_count == 0 for item in calibrations))
        self.assertTrue(all(item.raw_interval_covered_count == 0 for item in calibrations))
        self.assertTrue(
            all(item.calibrated_interval_covered_count == 0 for item in calibrations)
        )

    def test_artifact_binds_generator_partitions_gates_and_order_statistics(self):
        gates, keys = calibrate_corrected_verification_gates(
            self.trials(),
            self.physical,
            self.partition,
            self.config,
        )
        plan = CorrectedReplicationPlan(
            campaign="unit_campaign",
            gate_development_blocks=10,
            parent_calibration_blocks=10,
            parent_rehearsal_blocks=10,
            guard_development_blocks=10,
            guard_calibration_blocks=10,
            reserved_evaluation_blocks=10,
            bootstrap_draws=1_000,
        )
        scores = (0.0,) * 10
        rank, padding = block_conformal_padding(scores, self.config.target_block_coverage)
        calibrations = tuple(
            ProcedureCalibration(
                procedure_name=name,
                block_count=10,
                conformal_rank=rank,
                target_block_coverage=self.config.target_block_coverage,
                additive_margin_padding=padding,
                block_nonconformity=scores,
                emitted_interval_count=0,
                raw_interval_covered_count=0,
                calibrated_interval_covered_count=0,
            )
            for name in STAGE4_PROCEDURES
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "generator.py").write_text("VERSION = 2\n", encoding="utf-8")
            freeze = build_corrected_generator_freeze(
                root,
                OperatingDecisionRealismConfig(),
                plan,
                source_paths=("generator.py",),
            )
            artifact = build_corrected_parent_artifact(
                freeze,
                self.config,
                "1" * 64,
                "2" * 64,
                keys,
                gates,
                calibrations,
            )
            validate_corrected_parent_artifact(artifact, freeze, self.config)
            self.assertEqual(artifact.generator_freeze_digest, freeze.artifact_digest)
            self.assertEqual(artifact.reserved_evaluation.name, "r2_reserved_evaluation")
            payload = corrected_parent_artifact_payload(
                artifact,
                freeze,
                self.config,
            )
            loaded = corrected_parent_artifact_from_payload(payload)
            self.assertEqual(loaded.artifact, artifact)
            destination = root / "parent.json"
            save_corrected_parent_artifact(
                artifact,
                freeze,
                self.config,
                destination,
            )
            self.assertEqual(
                load_corrected_parent_artifact(destination).artifact,
                artifact,
            )
            with self.assertRaises(FileExistsError):
                save_corrected_parent_artifact(
                    artifact,
                    freeze,
                    self.config,
                    destination,
                )
            with self.assertRaisesRegex(ValueError, "fields changed"):
                validate_corrected_parent_artifact(
                    artifact._replace(protocol_digest="0" * 64),
                    freeze,
                    self.config,
                )
            invalid = calibrations[0]._replace(
                emitted_interval_count=0,
                calibrated_interval_covered_count=1,
            )
            with self.assertRaisesRegex(ValueError, "calibrated coverage exceeds"):
                build_corrected_parent_artifact(
                    freeze,
                    self.config,
                    "1" * 64,
                    "2" * 64,
                    keys,
                    gates,
                    (invalid, *calibrations[1:]),
                )


if __name__ == "__main__":
    unittest.main()
