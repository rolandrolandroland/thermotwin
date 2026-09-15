import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from thermotwin.studies.operating_decision import (
    INSUFFICIENT_EVIDENCE,
    STOP_NOW,
    CandidateVerification,
    MarginEnvelope,
    SavedOperatingDecision,
    ScoredOperatingDecision,
    default_fixed_policies,
)
from thermotwin.studies.operating_decision_calibration import (
    DECISION_DIRECTED_SELECTOR,
    STAGE4_PROCEDURES,
    InitialDecisionSignal,
    ProcedureCalibration,
    block_conformal_padding,
)
from thermotwin.studies.operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
    RealisticCandidateFit,
    Stage3CaseId,
    _parameter_spec,
)
from thermotwin.studies.operating_decision_replication import (
    CorrectedPartition,
    corrected_partition_config,
)
from thermotwin.studies.operating_decision_replication_calibration import (
    CorrectedParentCalibrationConfig,
    build_corrected_parent_artifact,
    calibrate_corrected_verification_gates,
)
from thermotwin.studies.operating_decision_replication_guard import (
    EARLY_MISMATCH_ABSTENTION,
    GUARD_DEVELOPMENT_SPLIT,
    MISMATCH_GUARDED_SELECTOR,
    AcquisitionAdequacySignal,
    CorrectedGuardCalibrationConfig,
    _early_abstention_trial,
    build_corrected_guard_artifact,
    calibrate_corrected_acquisition_guard,
    corrected_guard_artifact_from_payload,
    corrected_guard_artifact_payload,
    evaluate_corrected_reserved_cohort,
    select_corrected_action,
)
from thermotwin.studies.operating_decision_replication_protocol import (
    CorrectedReplicationPlan,
    build_corrected_generator_freeze,
)
from thermotwin.studies.sensor_model_discrimination import (
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
)


class OperatingDecisionReplicationGuardTests(unittest.TestCase):
    def setUp(self):
        self.partition = CorrectedPartition(
            GUARD_DEVELOPMENT_SPLIT,
            10,
            "unit_campaign",
        )
        self.physical = corrected_partition_config(
            self.partition,
            OperatingDecisionRealismConfig(),
        )

    def fit(self, model_name, objective):
        spec = _parameter_spec(model_name, False, self.physical)
        size = len(spec.names)
        return RealisticCandidateFit(
            model_name=model_name,
            log_multipliers=(0.0,) * size,
            parameter_names=spec.names,
            physical_values=self.physical.sensor.fit.nominal_values,
            interface_mass=(
                None
                if model_name == FOUR_STATE_MODEL
                else self.physical.sensor.interface_mass_nominal
            ),
            series_resistance=self.physical.series_resistance_nominal,
            face_sensor=None,
            objective=objective,
            covariance=tuple(
                tuple(0.0 for _ in range(size)) for _ in range(size)
            ),
            reached_bound=False,
            evaluation_count=10,
            converged=True,
            termination_reason="scaled_projected_gradient_tolerance",
            completed_iterations=3,
            accepted_iterations=2,
            scaled_gradient_infinity_norm=1.0e-7,
            last_step_infinity_norm=1.0e-8,
            last_relative_objective_reduction=1.0e-10,
        )

    def trials(self, *, family_c_objective=1.0):
        rows = []
        for truth in STAGE3_TRUTH_CONDITIONS:
            objective = family_c_objective if truth == STAGE3_TRUTH_CONDITIONS[2] else 1.0
            for block in range(self.partition.block_count):
                for policy in default_fixed_policies():
                    fits = (
                        self.fit(FOUR_STATE_MODEL, objective + block * 0.001),
                        self.fit(FIVE_STATE_MODEL, objective + 0.05 + block * 0.001),
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
                        margin_envelope=MarginEnvelope(-0.1, 0.1),
                        model_intervals=(),
                        verifications=tuple(
                            CandidateVerification(fit, 0.8, True, None)
                            for fit in fits
                        ),
                        failures=(),
                        decision_computation_seconds=0.1,
                    )
                    rows.append(
                        ScoredOperatingDecision(
                            truth_condition=truth,
                            saved=saved,
                            true_margin=0.02,
                            true_pass=True,
                            false_approval=False,
                            false_rejection=False,
                            interval_covered=True,
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

    @staticmethod
    def no_forecast_signal(*_arguments, **_keywords):
        return InitialDecisionSignal(
            candidate_models=tuple(sorted((FOUR_STATE_MODEL, FIVE_STATE_MODEL))),
            failed_fit_count=0,
            bound_hit_count=0,
            margin_envelope=None,
        )

    def test_guard_threshold_uses_matched_families_only(self):
        config = CorrectedGuardCalibrationConfig()
        with patch(
            "thermotwin.studies.operating_decision_replication_guard.initial_decision_signal",
            side_effect=self.no_forecast_signal,
        ):
            ordinary = calibrate_corrected_acquisition_guard(
                self.trials(family_c_objective=1.0),
                self.physical,
                self.partition,
                config,
            )
            changed_c = calibrate_corrected_acquisition_guard(
                self.trials(family_c_objective=1_000.0),
                self.physical,
                self.partition,
                config,
            )
        self.assertEqual(ordinary, changed_c)
        self.assertEqual(ordinary.block_count, 10)
        self.assertEqual(len(ordinary.matched_block_scores), 10)

    def test_selector_only_adds_early_abstention(self):
        guard = type("Guard", (), {"threshold": 1.0})()
        signal = AcquisitionAdequacySignal(
            candidate_models=tuple(sorted((FOUR_STATE_MODEL, FIVE_STATE_MODEL))),
            candidate_data_scores=((FOUR_STATE_MODEL, 2.0),),
            best_candidate_data_score=2.0,
            failed_fit_count=0,
            bound_hit_count=0,
            nonconverged_fit_count=0,
            margin_envelope=MarginEnvelope(-0.1, 0.1),
        )
        self.assertEqual(
            select_corrected_action(signal, guard),
            EARLY_MISMATCH_ABSTENTION,
        )
        resolved = signal._replace(
            best_candidate_data_score=0.5,
            margin_envelope=MarginEnvelope(0.1, 0.2),
        )
        self.assertEqual(select_corrected_action(resolved, guard), STOP_NOW)

    def test_early_abstention_uses_realized_acquisition_energy(self):
        stop = next(
            item for item in self.trials() if item.saved.policy_name == STOP_NOW
        )
        guarded = _early_abstention_trial(stop)
        self.assertEqual(guarded.saved.decision, INSUFFICIENT_EVIDENCE)
        self.assertEqual(guarded.diagnostic_run_count, 1)
        self.assertEqual(guarded.total_diagnostic_energy, stop.acquisition_energy)
        self.assertEqual(guarded.verification_energy, 0.0)

    def test_reserved_evaluator_requires_committed_artifact_before_generation(self):
        loaded = SimpleNamespace(
            generator_freeze=SimpleNamespace(
                source_manifest=object(),
                plan=SimpleNamespace(
                    partition=lambda name: CorrectedPartition(name, 1)
                ),
            )
        )
        with patch(
            "thermotwin.studies.operating_decision_replication_guard.load_corrected_guard_artifact",
            return_value=loaded,
        ), patch(
            "thermotwin.studies.operating_decision_replication_guard."
            "_authorize_reserved_evaluation",
            side_effect=ValueError("path is not committed at HEAD"),
        ), patch(
            "thermotwin.studies.operating_decision_replication_guard._run_corrected_partition"
        ) as runner:
            with self.assertRaisesRegex(ValueError, "not committed"):
                evaluate_corrected_reserved_cohort(
                    "uncommitted.json",
                    Path(__file__).resolve().parents[1],
                )
        runner.assert_not_called()

    def test_final_artifact_round_trip_binds_parent_evidence_and_bootstrap_keys(self):
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
        parent_config = CorrectedParentCalibrationConfig(
            gate_monte_carlo_draws=1_000
        )
        gate_partition = plan.partition("r2_gate_development")
        gate_physical = corrected_partition_config(
            gate_partition,
            OperatingDecisionRealismConfig(),
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
            gates, keys = calibrate_corrected_verification_gates(
                self.trials(),
                gate_physical,
                gate_partition,
                parent_config,
            )
            rank, padding = block_conformal_padding((0.0,) * 10, 0.90)
            parent_calibrations = tuple(
                ProcedureCalibration(
                    procedure_name=name,
                    block_count=10,
                    conformal_rank=rank,
                    target_block_coverage=0.90,
                    additive_margin_padding=padding,
                    block_nonconformity=(0.0,) * 10,
                    emitted_interval_count=0,
                    raw_interval_covered_count=0,
                    calibrated_interval_covered_count=0,
                )
                for name in STAGE4_PROCEDURES
            )
            parent = build_corrected_parent_artifact(
                freeze,
                parent_config,
                "1" * 64,
                "2" * 64,
                keys,
                gates,
                parent_calibrations,
            )
            with patch(
                "thermotwin.studies.operating_decision_replication_guard.initial_decision_signal",
                side_effect=self.no_forecast_signal,
            ):
                guard = calibrate_corrected_acquisition_guard(
                    self.trials(),
                    self.physical,
                    self.partition,
                )
            guarded_calibration = ProcedureCalibration(
                procedure_name=MISMATCH_GUARDED_SELECTOR,
                block_count=10,
                conformal_rank=rank,
                target_block_coverage=0.90,
                additive_margin_padding=parent.procedure_for(
                    DECISION_DIRECTED_SELECTOR
                ).additive_margin_padding,
                block_nonconformity=(0.0,) * 10,
                emitted_interval_count=0,
                raw_interval_covered_count=0,
                calibrated_interval_covered_count=0,
            )
            guard_config = CorrectedGuardCalibrationConfig()
            artifact = build_corrected_guard_artifact(
                freeze,
                parent,
                guard_config,
                "3" * 64,
                "4" * 64,
                guard,
                guarded_calibration,
            )
            invalid_calibration = guarded_calibration._replace(
                emitted_interval_count=0,
                calibrated_interval_covered_count=1,
            )
            with self.assertRaisesRegex(ValueError, "calibrated coverage exceeds"):
                build_corrected_guard_artifact(
                    freeze,
                    parent,
                    guard_config,
                    "3" * 64,
                    "4" * 64,
                    guard,
                    invalid_calibration,
                )
            payload = corrected_guard_artifact_payload(
                artifact,
                freeze,
                parent_config,
                parent,
                guard_config,
            )
            loaded = corrected_guard_artifact_from_payload(
                json.loads(json.dumps(payload))
            )
            self.assertEqual(loaded.artifact, artifact)
            self.assertEqual(artifact.parent_protocol_digest, parent.protocol_digest)
            self.assertEqual(artifact.guard_development_evidence_digest, "3" * 64)
            self.assertEqual(
                len(set(artifact.bootstrap_random_stream_keys)),
                len(parent.cost_scenarios),
            )


if __name__ == "__main__":
    unittest.main()
