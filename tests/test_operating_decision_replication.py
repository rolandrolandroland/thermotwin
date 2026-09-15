import math
from pathlib import Path
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from thermotwin.studies.operating_decision import (
    APPROVE,
    FIXED_THERMAL,
    INSUFFICIENT_EVIDENCE,
    STOP_NOW,
    CandidateVerification,
    MarginEnvelope,
    MarginInterval,
    SavedOperatingDecision,
    default_fixed_policies,
)
from thermotwin.studies.operating_decision_random_streams import (
    RandomStream,
    RandomStreamRegistry,
    audit_stream_uses,
    device_truth_stream,
    observation_stream,
)
from thermotwin.studies.operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
)
from thermotwin.studies.sensor_model_discrimination import (
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
)
from thermotwin.studies.operating_decision_replication import (
    CORRECTED_FIT_ITERATIONS,
    CORRECTED_GENERATOR_VERSION,
    CORRECTED_REPLICATION_CONFIG,
    CorrectedPartition,
    CorrectedPartitionResult,
    RESERVED_EVALUATION_PARTITION_NAME,
    _ReservedEvaluationRequest,
    _authorize_reserved_evaluation,
    _run_corrected_block,
    _run_corrected_partition,
    build_corrected_blinded_case,
    corrected_case_id,
    corrected_partition_config,
    corrected_truth_for_block,
    decide_corrected_blinded_case,
    score_corrected_saved_decision,
    run_corrected_partition,
)


class OperatingDecisionReplicationTests(unittest.TestCase):
    def setUp(self):
        self.partition = CorrectedPartition("unit_development", 2, "unit_campaign")
        base = OperatingDecisionRealismConfig()
        self.config = corrected_partition_config(
            self.partition,
            replace(
                base,
                sensor=replace(base.sensor, fit_iterations=1),
            ),
        )

    def _cases_for_block(self, block=0):
        registry = RandomStreamRegistry()
        truth = corrected_truth_for_block(
            self.partition,
            block,
            self.config,
            registry,
        )
        cases = {}
        for family in STAGE3_TRUTH_CONDITIONS:
            for policy in default_fixed_policies():
                cases[(family, policy.name)] = build_corrected_blinded_case(
                    family,
                    block,
                    policy,
                    truth,
                    self.partition,
                    self.config,
                    registry,
                )
        return truth, cases, registry.audit()

    def test_corrected_streams_remove_run_channel_and_adjacent_block_collisions(self):
        initial_hot = observation_stream(
            campaign=self.partition.campaign,
            partition=self.partition.name,
            block=0,
            purpose="white_noise",
            family=STAGE3_TRUTH_CONDITIONS[0],
            run="initial_0.8A_20s|temporary_face_sensor=0",
            channel="hot_exchanger",
        )
        thermal_cold = observation_stream(
            campaign=self.partition.campaign,
            partition=self.partition.name,
            block=0,
            purpose="white_noise",
            family=STAGE3_TRUTH_CONDITIONS[0],
            run="thermal_0.6A_30s|temporary_face_sensor=0",
            channel="cold_exchanger",
        )
        verification_hot = observation_stream(
            campaign=self.partition.campaign,
            partition=self.partition.name,
            block=0,
            purpose="run_bias",
            family=STAGE3_TRUTH_CONDITIONS[0],
            run="fixed_verification|temporary_face_sensor=0",
            channel="hot_exchanger",
        )
        next_device = device_truth_stream(
            campaign=self.partition.campaign,
            partition=self.partition.name,
            block=1,
            purpose="physical_parameter_0",
        )
        self.assertEqual(len({
            initial_hot.seed,
            thermal_cold.seed,
            verification_hot.seed,
            next_device.seed,
        }), 4)

    def test_truth_and_measurements_are_paired_only_where_declared(self):
        truth, cases, audit = self._cases_for_block()
        self.assertTrue(audit.ok)
        self.assertGreater(len(audit.declared_pairings), 0)
        stop = cases[(STAGE3_TRUTH_CONDITIONS[0], STOP_NOW)]
        thermal = cases[(STAGE3_TRUTH_CONDITIONS[0], FIXED_THERMAL)]
        self.assertEqual(
            stop.acquisition_runs[0].observations,
            thermal.acquisition_runs[0].observations,
        )
        other_family = cases[(STAGE3_TRUTH_CONDITIONS[1], STOP_NOW)]
        self.assertNotEqual(
            stop.acquisition_runs[0].observations,
            other_family.acquisition_runs[0].observations,
        )
        self.assertTrue(all(value > 0.0 for value in truth.physical_values))

    def test_partition_and_protocol_change_device_identity(self):
        first = corrected_case_id(
            self.partition,
            STAGE3_TRUTH_CONDITIONS[0],
            0,
            STOP_NOW,
            self.config,
        )
        changed_partition = corrected_case_id(
            CorrectedPartition("other", 2, "unit_campaign"),
            STAGE3_TRUTH_CONDITIONS[0],
            0,
            STOP_NOW,
            self.config,
        )
        self.assertNotEqual(first.device_token, changed_partition.device_token)
        self.assertIn("v4", CORRECTED_GENERATOR_VERSION)

    def test_corrected_numerical_config_separates_truth_and_fit_support(self):
        self.assertEqual(
            CORRECTED_REPLICATION_CONFIG.sensor.fit_iterations,
            CORRECTED_FIT_ITERATIONS,
        )
        self.assertEqual(CORRECTED_FIT_ITERATIONS, 12)
        for truth_bounds, fit_bounds in (
            (
                CORRECTED_REPLICATION_CONFIG.series_resistance_bounds,
                CORRECTED_REPLICATION_CONFIG.series_resistance_fit_bounds,
            ),
            (
                CORRECTED_REPLICATION_CONFIG.face_sensor_capacitance_bounds,
                CORRECTED_REPLICATION_CONFIG.face_sensor_capacitance_fit_bounds,
            ),
            (
                CORRECTED_REPLICATION_CONFIG.face_sensor_response_bounds,
                CORRECTED_REPLICATION_CONFIG.face_sensor_response_fit_bounds,
            ),
        ):
            self.assertLess(fit_bounds[0], truth_bounds[0])
            self.assertGreater(fit_bounds[1], truth_bounds[1])

    def test_saved_decision_is_scored_with_realized_not_nominal_energy(self):
        registry = RandomStreamRegistry()
        truth = corrected_truth_for_block(
            self.partition,
            0,
            self.config,
            registry,
        )
        policy = next(item for item in default_fixed_policies() if item.name == STOP_NOW)
        case = build_corrected_blinded_case(
            STAGE3_TRUTH_CONDITIONS[0],
            0,
            policy,
            truth,
            self.partition,
            self.config,
            registry,
        )
        saved = decide_corrected_blinded_case(case, config=self.config)
        for verification in saved.verifications:
            if not verification.fit.converged:
                self.assertFalse(verification.passed)
                self.assertEqual(
                    verification.failure_reason,
                    "optimizer_not_converged",
                )
        record = score_corrected_saved_decision(
            saved,
            truth_condition=STAGE3_TRUTH_CONDITIONS[0],
            block=0,
            case=case,
            truth=truth,
            partition=self.partition,
            config=self.config,
        )
        self.assertTrue(math.isfinite(record.scored.total_diagnostic_energy))
        self.assertEqual(
            record.scored.total_diagnostic_energy,
            record.realized_energy.total_diagnostic_energy,
        )
        self.assertNotEqual(
            record.nominal_selection_energy,
            record.scored.total_diagnostic_energy,
        )

    def test_corrected_decision_excludes_one_unusable_candidate(self):
        _, cases, _ = self._cases_for_block()
        case = cases[(STAGE3_TRUTH_CONDITIONS[0], STOP_NOW)]
        good_fit = SimpleNamespace(
            model_name=FOUR_STATE_MODEL,
            converged=True,
            reached_bound=False,
        )
        good_interval = MarginInterval(
            FOUR_STATE_MODEL,
            0.15,
            0.01,
            0.10,
            0.20,
        )
        for converged, reached_bound, reason in (
            (False, False, "optimizer_not_converged"),
            (True, True, "fit_reached_bound"),
        ):
            with self.subTest(reason=reason):
                bad_fit = SimpleNamespace(
                    model_name=FIVE_STATE_MODEL,
                    converged=converged,
                    reached_bound=reached_bound,
                )
                bad_interval = MarginInterval(
                    FIVE_STATE_MODEL,
                    -0.15,
                    0.01,
                    -0.20,
                    -0.10,
                )
                saved = SavedOperatingDecision(
                    case_id=case.case_id,
                    decision=INSUFFICIENT_EVIDENCE,
                    decision_reason="unit",
                    margin_envelope=MarginEnvelope(-0.20, 0.20),
                    model_intervals=(good_interval, bad_interval),
                    verifications=(
                        CandidateVerification(good_fit, 0.5, True, None),
                        CandidateVerification(bad_fit, 0.5, True, None),
                    ),
                    failures=(),
                    decision_computation_seconds=0.0,
                )
                with patch(
                    "thermotwin.studies.operating_decision_replication."
                    "decide_realistic_blinded_case",
                    return_value=saved,
                ):
                    corrected = decide_corrected_blinded_case(
                        case,
                        config=self.config,
                    )
                self.assertEqual(corrected.decision, APPROVE)
                self.assertEqual(corrected.margin_envelope, MarginEnvelope(0.10, 0.20))
                self.assertEqual(corrected.model_intervals, (good_interval,))
                self.assertEqual(corrected.failures, ())
                excluded = next(
                    item
                    for item in corrected.verifications
                    if item.fit.model_name == FIVE_STATE_MODEL
                )
                self.assertFalse(excluded.passed)
                self.assertEqual(excluded.failure_reason, reason)

    def test_corrected_decision_abstains_when_every_candidate_is_excluded(self):
        _, cases, _ = self._cases_for_block()
        case = cases[(STAGE3_TRUTH_CONDITIONS[0], STOP_NOW)]
        fits = (
            SimpleNamespace(
                model_name=FOUR_STATE_MODEL,
                converged=False,
                reached_bound=False,
            ),
            SimpleNamespace(
                model_name=FIVE_STATE_MODEL,
                converged=True,
                reached_bound=True,
            ),
        )
        intervals = tuple(
            MarginInterval(fit.model_name, 0.15, 0.01, 0.10, 0.20)
            for fit in fits
        )
        saved = SavedOperatingDecision(
            case_id=case.case_id,
            decision=APPROVE,
            decision_reason="unit",
            margin_envelope=MarginEnvelope(0.10, 0.20),
            model_intervals=intervals,
            verifications=tuple(
                CandidateVerification(fit, 0.5, True, None) for fit in fits
            ),
            failures=(),
            decision_computation_seconds=0.0,
        )
        with patch(
            "thermotwin.studies.operating_decision_replication."
            "decide_realistic_blinded_case",
            return_value=saved,
        ):
            corrected = decide_corrected_blinded_case(case, config=self.config)
        self.assertEqual(corrected.decision, INSUFFICIENT_EVIDENCE)
        self.assertEqual(
            corrected.decision_reason,
            "no_corrected_verified_candidate",
        )
        self.assertIsNone(corrected.margin_envelope)
        self.assertEqual(corrected.model_intervals, ())
        self.assertEqual(corrected.failures, ())

    def test_malformed_partition_and_mismatched_config_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive block"):
            CorrectedPartition("bad", 0)
        mismatched = replace(
            self.config,
            sensor=replace(self.config.sensor, trial_count=1),
        )
        with self.assertRaisesRegex(ValueError, "trial count"):
            corrected_truth_for_block(
                self.partition,
                0,
                mismatched,
                RandomStreamRegistry(),
            )

    def test_public_runner_keeps_reserved_evaluation_sealed(self):
        reserved = CorrectedPartition(
            RESERVED_EVALUATION_PARTITION_NAME,
            1,
            self.partition.campaign,
        )
        with self.assertRaisesRegex(ValueError, "manifest-bound artifact"):
            run_corrected_partition(reserved)
        with patch(
            "thermotwin.studies.operating_decision_replication._validate_block"
        ) as validate:
            with self.assertRaisesRegex(
                ValueError,
                "committed-artifact authorization",
            ):
                _run_corrected_partition(reserved)
        validate.assert_not_called()

        reserved_config = corrected_partition_config(reserved, self.config)
        registry = RandomStreamRegistry()
        with self.assertRaisesRegex(ValueError, "committed-artifact authorization"):
            corrected_truth_for_block(reserved, 0, reserved_config, registry)
        truth = corrected_truth_for_block(
            self.partition,
            0,
            self.config,
            RandomStreamRegistry(),
        )
        with self.assertRaisesRegex(ValueError, "committed-artifact authorization"):
            build_corrected_blinded_case(
                STAGE3_TRUTH_CONDITIONS[0],
                0,
                default_fixed_policies()[0],
                truth,
                reserved,
                reserved_config,
                RandomStreamRegistry(),
            )
        with patch(
            "thermotwin.studies.operating_decision_replication."
            "corrected_truth_for_block"
        ) as generate_truth:
            with self.assertRaisesRegex(
                ValueError,
                "committed-artifact authorization",
            ):
                _run_corrected_block(reserved, 0, reserved_config)
        generate_truth.assert_not_called()

        forged_request = _ReservedEvaluationRequest(
            str(Path(__file__).resolve().parents[1]),
            str(Path(__file__).resolve().parents[1] / ".gitignore"),
            "0" * 40,
            (".gitignore",),
        )
        with patch(
            "thermotwin.studies.operating_decision_replication."
            "corrected_truth_for_block"
        ) as generate_truth:
            with self.assertRaises(ValueError):
                _run_corrected_block(
                    reserved,
                    0,
                    reserved_config,
                    _reserved_request=forged_request,
                )
        generate_truth.assert_not_called()

    def test_reserved_authorization_performs_git_verification_itself(self):
        loaded = SimpleNamespace(
            generator_freeze=SimpleNamespace(source_manifest=SimpleNamespace())
        )
        with patch(
            "thermotwin.studies.operating_decision_replication_guard."
            "load_corrected_guard_artifact",
            return_value=loaded,
        ), patch(
            "thermotwin.studies.operating_decision_replication."
            "verify_committed_artifact_chain",
            side_effect=ValueError("path is not committed at HEAD"),
        ) as verify:
            with self.assertRaisesRegex(ValueError, "not committed"):
                _authorize_reserved_evaluation(
                    CorrectedPartition(
                        RESERVED_EVALUATION_PARTITION_NAME,
                        1,
                        "unit_campaign",
                    ),
                    "/repository",
                    "guard.json",
                    self.config,
                )
        verify.assert_called_once_with(
            "/repository",
            "guard.json",
            loaded.generator_freeze.source_manifest,
        )

    def test_corrected_fit_records_convergence_before_decision(self):
        converged_config = corrected_partition_config(
            self.partition,
            OperatingDecisionRealismConfig(),
        )
        registry = RandomStreamRegistry()
        truth = corrected_truth_for_block(
            self.partition,
            0,
            converged_config,
            registry,
        )
        policy = next(item for item in default_fixed_policies() if item.name == STOP_NOW)
        case = build_corrected_blinded_case(
            STAGE3_TRUTH_CONDITIONS[0],
            0,
            policy,
            truth,
            self.partition,
            converged_config,
            registry,
        )
        saved = decide_corrected_blinded_case(case, config=converged_config)
        self.assertTrue(saved.verifications)
        self.assertTrue(all(item.fit.converged for item in saved.verifications))

    def test_partition_result_recomputes_identity_namespace_and_stream_audit(self):
        partition = CorrectedPartition("identity_unit", 1, "unit_campaign")
        config = corrected_partition_config(partition, self.config)
        registry = RandomStreamRegistry()
        truth = corrected_truth_for_block(partition, 0, config, registry)
        records = []
        for truth_condition in STAGE3_TRUTH_CONDITIONS:
            for policy in default_fixed_policies():
                case = build_corrected_blinded_case(
                    truth_condition,
                    0,
                    policy,
                    truth,
                    partition,
                    config,
                    registry,
                )
                saved = SavedOperatingDecision(
                    case_id=case.case_id,
                    decision=INSUFFICIENT_EVIDENCE,
                    decision_reason="unit_identity_check",
                    margin_envelope=None,
                    model_intervals=(),
                    verifications=(),
                    failures=(),
                    decision_computation_seconds=0.0,
                )
                records.append(
                    score_corrected_saved_decision(
                        saved,
                        truth_condition=truth_condition,
                        block=0,
                        case=case,
                        truth=truth,
                        partition=partition,
                        config=config,
                    )
                )
        uses = registry.uses
        audit = audit_stream_uses(uses)
        result = CorrectedPartitionResult(
            partition,
            config,
            tuple(records),
            uses,
            audit,
        )
        foreign_partition = CorrectedPartition("foreign", 1, "unit_campaign")
        foreign_record = replace(result.records[0], partition=foreign_partition)
        with self.assertRaisesRegex(ValueError, "different partition"):
            replace(
                result,
                records=(foreign_record, *result.records[1:]),
            )
        foreign_key = replace(result.random_stream_uses[0].stream.key, partition="foreign")
        foreign_use = replace(
            result.random_stream_uses[0],
            stream=RandomStream(foreign_key),
        )
        changed_uses = (foreign_use, *result.random_stream_uses[1:])
        with self.assertRaisesRegex(ValueError, "different namespace"):
            replace(
                result,
                random_stream_uses=changed_uses,
                random_stream_audit=audit_stream_uses(changed_uses),
            )
        forged_audit = replace(
            result.random_stream_audit,
            unique_seed_count=result.random_stream_audit.unique_seed_count + 1,
        )
        with self.assertRaisesRegex(ValueError, "recomputed faithfully"):
            replace(result, random_stream_audit=forged_audit)
        changed_physics = replace(
            config,
            series_resistance_nominal=config.series_resistance_nominal * 1.01,
        )
        with self.assertRaisesRegex(ValueError, "identity is not reproducible"):
            replace(result, config=changed_physics)


if __name__ == "__main__":
    unittest.main()
