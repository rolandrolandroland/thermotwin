import math
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import fields, replace
from unittest.mock import patch

from thermotwin.studies.operating_decision import (
    APPROVE,
    CandidateVerification,
    FIXED_VOLTAGE,
    INSUFFICIENT_EVIDENCE,
    REJECT,
    RUN_DURATION_SECONDS,
    STOP_NOW,
    MarginEnvelope,
    SavedOperatingDecision,
    ScoredOperatingDecision,
    default_fixed_policies,
)
from thermotwin.reports.operating_decision_final_evaluation import (
    _preflight_output_paths,
    final_result_payload,
    main as final_evaluation_main,
)
from thermotwin.studies.operating_decision_calibration import (
    DECISION_DIRECTED_SELECTOR,
    InitialDecisionSignal,
    ProcedureCalibration,
    ProcedureOutcome,
)
from thermotwin.studies.operating_decision_final_evaluation import (
    EARLY_MISMATCH_ABSTENTION,
    FINAL_EVALUATION_SPLIT,
    MISMATCH_GUARDED_SELECTOR,
    AcquisitionAdequacySignal,
    AcquisitionGuardCalibration,
    FinalEvaluationResult,
    OperatingDecisionFinalConfig,
    RevisedSelectorRule,
    _artifact_digest,
    _build_artifact,
    _calibrate_revised_procedure,
    _early_abstention_trial,
    _paired_loss_comparisons,
    _revised_choices,
    _validate_monotone_revision,
    acquisition_adequacy_signal,
    acquisition_data_score,
    calibrate_acquisition_guard,
    evaluate_reserved_stage5,
    freeze_revised_selector,
    load_revised_artifact,
    load_stage4_calibration_artifact,
    save_revised_artifact,
    select_revised_action,
)
from thermotwin.studies.operating_decision_realism import (
    OperatingDecisionRealismConfig,
    RealisticCandidateFit,
    Stage3CaseId,
    _parameter_spec,
    build_realistic_blinded_case,
    fit_realistic_acquisition_models,
)
from thermotwin.studies.sensor_model_discrimination import (
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
    MODEL_NAMES,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT_ARTIFACT = ROOT / "thermotwin" / "OPERATING_DECISION_CALIBRATION_ARTIFACT.json"


class OperatingDecisionFinalEvaluationTests(unittest.TestCase):
    @staticmethod
    def _fit(model_name: str, objective: float = 1.0) -> RealisticCandidateFit:
        config = OperatingDecisionRealismConfig()
        spec = _parameter_spec(model_name, False, config)
        size = len(spec.names)
        return RealisticCandidateFit(
            model_name=model_name,
            log_multipliers=(0.0,) * size,
            parameter_names=spec.names,
            physical_values=config.sensor.fit.nominal_values,
            interface_mass=(
                config.sensor.interface_mass_nominal
                if model_name == FIVE_STATE_MODEL
                else None
            ),
            series_resistance=config.series_resistance_nominal,
            face_sensor=None,
            objective=objective,
            covariance=tuple(
                tuple(1.0 if row == column else 0.0 for column in range(size))
                for row in range(size)
            ),
            reached_bound=False,
            evaluation_count=1,
        )

    @staticmethod
    def _guard(threshold: float = 0.9) -> AcquisitionGuardCalibration:
        return AcquisitionGuardCalibration(
            target_matched_block_retention=0.90,
            block_count=20,
            conformal_rank=19,
            matched_block_scores=(0.9,) * 20,
            threshold=threshold,
        )

    @staticmethod
    def _scored_stop() -> ScoredOperatingDecision:
        saved = SavedOperatingDecision(
            case_id=Stage3CaseId("device", STOP_NOW, 0, "stage3_final_v1"),
            decision=APPROVE,
            decision_reason="test",
            margin_envelope=MarginEnvelope(0.1, 0.2),
            model_intervals=(),
            verifications=(),
            failures=(),
            decision_computation_seconds=0.1,
        )
        return ScoredOperatingDecision(
            truth_condition="matched_four_state",
            saved=saved,
            true_margin=0.15,
            true_pass=True,
            false_approval=False,
            false_rejection=False,
            interval_covered=True,
            acquisition_run_count=1,
            diagnostic_run_count=2,
            energized_schedule_time_seconds=120.0,
            acquisition_energy=20.0,
            verification_energy=10.0,
            total_diagnostic_energy=30.0,
            extra_sensor_count=0,
        )

    @staticmethod
    def _valid_calibration(parent, *, emitted: int = 0) -> ProcedureCalibration:
        return ProcedureCalibration(
            procedure_name=MISMATCH_GUARDED_SELECTOR,
            block_count=20,
            conformal_rank=19,
            target_block_coverage=0.90,
            additive_margin_padding=parent.procedure_for(
                DECISION_DIRECTED_SELECTOR
            ).additive_margin_padding,
            block_nonconformity=(0.0,) * 20,
            emitted_interval_count=emitted,
            raw_interval_covered_count=emitted,
            calibrated_interval_covered_count=emitted,
        )

    @classmethod
    def _fit_for_score(
        cls,
        model_name: str,
        score: float,
        config: OperatingDecisionRealismConfig,
    ) -> RealisticCandidateFit:
        fit = cls._fit(model_name, objective=0.0)
        sample_count = int(RUN_DURATION_SECONDS / config.sensor.sampling_interval) + 1
        observation_count = 2 * sample_count
        full_residual_count = observation_count + 2 + len(fit.parameter_names)
        return fit._replace(
            objective=score * observation_count / full_residual_count
        )

    @staticmethod
    def _procedure_outcome(
        procedure_name: str,
        *,
        truth_condition: str = "matched_four_state",
        trial_index: int = 0,
        selected_policy: str = STOP_NOW,
        decision: str = APPROVE,
        diagnostic_run_count: int = 1,
        energy: float = 10.0,
        extra_sensor_count: int = 0,
    ) -> ProcedureOutcome:
        return ProcedureOutcome(
            split=FINAL_EVALUATION_SPLIT,
            truth_condition=truth_condition,
            trial_index=trial_index,
            procedure_name=procedure_name,
            selected_policy=selected_policy,
            decision=decision,
            true_margin=0.1,
            true_pass=True,
            false_approval=False,
            false_rejection=False,
            raw_interval_lower=0.05,
            raw_interval_upper=0.15,
            calibrated_interval_lower=0.04,
            calibrated_interval_upper=0.16,
            raw_interval_covered=True,
            calibrated_interval_covered=True,
            additive_margin_padding=0.01,
            verified_candidate_count=2,
            score_rejected_candidate_count=0,
            numerical_failure_count=0,
            acquisition_run_count=1,
            diagnostic_run_count=diagnostic_run_count,
            energized_schedule_time_seconds=RUN_DURATION_SECONDS
            * diagnostic_run_count,
            total_diagnostic_energy=energy,
            extra_sensor_count=extra_sensor_count,
            decision_computation_seconds=0.1,
        )

    def test_default_partitions_preserve_and_do_not_overlap_stage5(self):
        config = OperatingDecisionFinalConfig()
        self.assertEqual(config.guard_development.sensor.first_seed, 50_191_001)
        self.assertEqual(config.recalibration.sensor.first_seed, 60_191_001)
        self.assertEqual(config.evaluation.sensor.first_seed, 30_191_001)
        self.assertEqual(config.evaluation.sensor.trial_count, 50)
        with self.assertRaisesRegex(ValueError, "seeds overlap"):
            replace(
                config,
                recalibration=replace(
                    config.recalibration,
                    sensor=replace(
                        config.recalibration.sensor,
                        first_seed=config.guard_development.sensor.first_seed,
                    ),
                ),
            )

    def test_parent_artifact_is_the_frozen_stage4_protocol(self):
        artifact = load_stage4_calibration_artifact(PARENT_ARTIFACT)
        self.assertEqual(
            artifact.protocol_digest,
            "0c87ccfdbec5b1047490a3e2408b2a599cfa71687f18ef55076724108e5b3b68",
        )
        self.assertEqual(artifact.reserved_evaluation_first_seed, 30_191_001)
        self.assertEqual(artifact.reserved_evaluation_block_count, 50)

    def test_parent_artifact_recomputes_digest_instead_of_trusting_label(self):
        payload = json.loads(PARENT_ARTIFACT.read_text(encoding="utf-8"))
        payload["verification_gates"][0]["threshold"] += 0.01
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered-parent.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contents do not match"):
                load_stage4_calibration_artifact(path)

    def test_acquisition_score_removes_prior_terms_and_uses_observation_count(self):
        config = OperatingDecisionRealismConfig()
        fit = self._fit(FOUR_STATE_MODEL, objective=1.0)
        sample_count = 81
        expected = (2 * sample_count + 2 + len(fit.parameter_names)) / (
            2 * sample_count
        )
        self.assertAlmostEqual(acquisition_data_score(fit, config), expected)
        moved = fit._replace(log_multipliers=(0.1,) * len(fit.log_multipliers))
        self.assertLess(acquisition_data_score(moved, config), expected)

    def test_actual_guard_signal_ignores_verification_and_revealed_truth(self):
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
        signal = acquisition_adequacy_signal(fit_set, config)
        self.assertEqual(set(signal.candidate_models), set(MODEL_NAMES))
        self.assertEqual(
            {name for name, _ in signal.candidate_data_scores},
            set(MODEL_NAMES),
        )

        indexed = {}
        for truth in (
            "matched_four_state",
            "extra_interface_mass",
            "temperature_dependent_contact",
        ):
            for policy in default_fixed_policies():
                saved = SavedOperatingDecision(
                    case_id=Stage3CaseId(
                        f"guard-{truth}",
                        policy.name,
                        0,
                        "stage3_final_v1",
                    ),
                    decision=INSUFFICIENT_EVIDENCE,
                    decision_reason="guard-test",
                    margin_envelope=None,
                    model_intervals=(),
                    verifications=tuple(
                        CandidateVerification(fit, 1.0, True, None)
                        for fit in fit_set.fits
                    ),
                    failures=fit_set.failures,
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
                    acquisition_run_count=1,
                    diagnostic_run_count=2,
                    energized_schedule_time_seconds=2 * RUN_DURATION_SECONDS,
                    acquisition_energy=10.0,
                    verification_energy=5.0,
                    total_diagnostic_energy=15.0,
                    extra_sensor_count=policy.extra_sensor_count,
                )
        guard = self._guard(1.0e9)
        original = _revised_choices(indexed, config, guard, RevisedSelectorRule())
        mutated = {
            key: row._replace(
                saved=row.saved._replace(
                    decision=REJECT,
                    verifications=tuple(
                        verification._replace(normalized_score=1.0e9, passed=False)
                        for verification in row.saved.verifications
                    ),
                ),
                true_margin=-1.0e9,
                true_pass=False,
            )
            for key, row in indexed.items()
        }
        self.assertEqual(
            original,
            _revised_choices(mutated, config, guard, RevisedSelectorRule()),
        )

    def test_selector_input_excludes_truth_and_later_data(self):
        forbidden = {
            "truth_condition",
            "trial_index",
            "device_token",
            "verification_score",
            "verification_run",
            "selected_policy",
            "true_margin",
            "final_response",
        }
        self.assertFalse(forbidden.intersection(AcquisitionAdequacySignal._fields))

    def test_revised_rule_has_three_auditable_branches(self):
        base = AcquisitionAdequacySignal(
            candidate_models=tuple(sorted(MODEL_NAMES)),
            candidate_data_scores=((FOUR_STATE_MODEL, 0.8), (FIVE_STATE_MODEL, 0.9)),
            best_candidate_data_score=0.8,
            failed_fit_count=0,
            bound_hit_count=0,
            margin_envelope=MarginEnvelope(0.1, 0.2),
        )
        guard = self._guard(1.0)
        self.assertEqual(select_revised_action(base, guard), STOP_NOW)
        self.assertEqual(
            select_revised_action(
                base._replace(margin_envelope=MarginEnvelope(-0.1, 0.2)),
                guard,
            ),
            FIXED_VOLTAGE,
        )
        self.assertEqual(
            select_revised_action(
                base._replace(best_candidate_data_score=math.nextafter(1.0, math.inf)),
                guard,
            ),
            EARLY_MISMATCH_ABSTENTION,
        )
        self.assertEqual(
            select_revised_action(
                base._replace(
                    candidate_data_scores=(
                        (FOUR_STATE_MODEL, math.inf),
                        (FIVE_STATE_MODEL, math.inf),
                    ),
                    best_candidate_data_score=math.inf,
                    failed_fit_count=2,
                    margin_envelope=None,
                ),
                guard,
            ),
            EARLY_MISMATCH_ABSTENTION,
        )

    def test_guard_threshold_is_inclusive_for_matched_retention(self):
        signal = AcquisitionAdequacySignal(
            candidate_models=tuple(sorted(MODEL_NAMES)),
            candidate_data_scores=((FOUR_STATE_MODEL, 1.0), (FIVE_STATE_MODEL, 1.1)),
            best_candidate_data_score=1.0,
            failed_fit_count=0,
            bound_hit_count=0,
            margin_envelope=MarginEnvelope(-0.1, 0.2),
        )
        self.assertEqual(select_revised_action(signal, self._guard(1.0)), FIXED_VOLTAGE)

    def test_guard_calibration_uses_matched_family_block_maximum_and_rank(self):
        config = OperatingDecisionFinalConfig()
        trials = []
        expected_scores = []
        for trial_index in range(config.guard_development.sensor.trial_count):
            family_scores = (0.10 + trial_index, 0.60 + trial_index)
            expected_scores.append(max(family_scores))
            for truth, score in zip(
                ("matched_four_state", "extra_interface_mass"),
                family_scores,
            ):
                fits = tuple(
                    self._fit_for_score(model_name, score + model_index, config.guard_development)
                    for model_index, model_name in enumerate(sorted(MODEL_NAMES))
                )
                for policy in default_fixed_policies():
                    saved = SavedOperatingDecision(
                        case_id=Stage3CaseId(
                            f"calibrate-{truth}-{trial_index}",
                            policy.name,
                            trial_index,
                            "stage3_final_v1",
                        ),
                        decision=INSUFFICIENT_EVIDENCE,
                        decision_reason="guard-calibration-test",
                        margin_envelope=None,
                        model_intervals=(),
                        verifications=tuple(
                            CandidateVerification(fit, 1.0, True, None) for fit in fits
                        ),
                        failures=(),
                        decision_computation_seconds=0.0,
                    )
                    trials.append(
                        ScoredOperatingDecision(
                            truth_condition=truth,
                            saved=saved,
                            true_margin=0.0,
                            true_pass=True,
                            false_approval=False,
                            false_rejection=False,
                            interval_covered=None,
                            acquisition_run_count=1,
                            diagnostic_run_count=1,
                            energized_schedule_time_seconds=RUN_DURATION_SECONDS,
                            acquisition_energy=10.0,
                            verification_energy=0.0,
                            total_diagnostic_energy=10.0,
                            extra_sensor_count=policy.extra_sensor_count,
                        )
                    )
        initial = InitialDecisionSignal(
            candidate_models=tuple(sorted(MODEL_NAMES)),
            failed_fit_count=0,
            bound_hit_count=0,
            margin_envelope=MarginEnvelope(-0.1, 0.1),
        )
        with patch(
            "thermotwin.studies.operating_decision_final_evaluation.initial_decision_signal",
            return_value=initial,
        ):
            guard = calibrate_acquisition_guard(trials, config)
        self.assertEqual(guard.conformal_rank, 19)
        for actual, expected in zip(guard.matched_block_scores, expected_scores):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(guard.threshold, sorted(expected_scores)[18])

    def test_early_guard_abstention_counts_only_common_acquisition(self):
        raw = self._scored_stop()
        guarded = _early_abstention_trial(raw)
        self.assertEqual(guarded.saved.decision, INSUFFICIENT_EVIDENCE)
        self.assertEqual(guarded.saved.decision_reason, "acquisition_mismatch_guard")
        self.assertIsNone(guarded.saved.margin_envelope)
        self.assertEqual(guarded.diagnostic_run_count, 1)
        self.assertEqual(
            guarded.energized_schedule_time_seconds,
            RUN_DURATION_SECONDS,
        )
        self.assertEqual(guarded.verification_energy, 0.0)
        self.assertEqual(guarded.total_diagnostic_energy, raw.acquisition_energy)

    def test_recalibration_uses_three_family_block_maximum_and_parent_floor(self):
        parent = load_stage4_calibration_artifact(PARENT_ARTIFACT)
        config = OperatingDecisionFinalConfig()
        indexed = {}
        choices = {}
        expected_block_scores = []
        for trial_index in range(config.recalibration.sensor.trial_count):
            scores = (
                0.001 * trial_index,
                0.010 + 0.001 * trial_index,
                0.020 + 0.001 * trial_index,
            )
            expected_block_scores.append(max(scores))
            for truth, score in zip(
                (
                    "matched_four_state",
                    "extra_interface_mass",
                    "temperature_dependent_contact",
                ),
                scores,
            ):
                saved = SavedOperatingDecision(
                    case_id=Stage3CaseId(
                        f"padding-{truth}-{trial_index}",
                        STOP_NOW,
                        trial_index,
                        "stage3_final_v1",
                    ),
                    decision=APPROVE,
                    decision_reason="padding-test",
                    margin_envelope=MarginEnvelope(score, score + 0.1),
                    model_intervals=(),
                    verifications=(),
                    failures=(),
                    decision_computation_seconds=0.0,
                )
                indexed[(truth, trial_index, STOP_NOW)] = ScoredOperatingDecision(
                    truth_condition=truth,
                    saved=saved,
                    true_margin=0.0,
                    true_pass=True,
                    false_approval=False,
                    false_rejection=False,
                    interval_covered=score == 0.0,
                    acquisition_run_count=1,
                    diagnostic_run_count=1,
                    energized_schedule_time_seconds=RUN_DURATION_SECONDS,
                    acquisition_energy=10.0,
                    verification_energy=0.0,
                    total_diagnostic_energy=10.0,
                    extra_sensor_count=0,
                )
                choices[(truth, trial_index)] = STOP_NOW
        with (
            patch(
                "thermotwin.studies.operating_decision_final_evaluation._validate_paired_trials",
                return_value=indexed,
            ),
            patch(
                "thermotwin.studies.operating_decision_final_evaluation._gate_trials",
                return_value=indexed,
            ),
            patch(
                "thermotwin.studies.operating_decision_final_evaluation._revised_choices",
                return_value=choices,
            ),
        ):
            calibration = _calibrate_revised_procedure(
                (),
                parent,
                config,
                self._guard(),
            )
        self.assertEqual(calibration.conformal_rank, 19)
        for actual, expected in zip(
            calibration.block_nonconformity,
            expected_block_scores,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            calibration.additive_margin_padding,
            parent.procedure_for(
                DECISION_DIRECTED_SELECTOR
            ).additive_margin_padding,
        )

    def test_revised_artifact_round_trip_is_exact(self):
        parent = load_stage4_calibration_artifact(PARENT_ARTIFACT)
        config = OperatingDecisionFinalConfig()
        calibration = ProcedureCalibration(
            procedure_name="mismatch_guarded_selector_v2",
            block_count=20,
            conformal_rank=19,
            target_block_coverage=0.90,
            additive_margin_padding=parent.procedure_for(
                "decision_directed_selector"
            ).additive_margin_padding,
            block_nonconformity=(0.0,) * 20,
            emitted_interval_count=30,
            raw_interval_covered_count=29,
            calibrated_interval_covered_count=30,
        )
        artifact = _build_artifact(
            source_revision="test-revision",
            parent=parent,
            config=config,
            guard=self._guard(),
            calibration=calibration,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            save_revised_artifact(artifact, path)
            self.assertEqual(load_revised_artifact(path), artifact)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["analysis"]["paired_block_bootstrap"],
                {"draws": 20_000, "seed": 70_191_001},
            )
            self.assertEqual(
                [item["name"] for item in payload["analysis"]["cost_scenarios"]],
                [item.name for item in config.cost_scenarios],
            )
            with self.assertRaises(FileExistsError):
                save_revised_artifact(artifact, path)

    def test_evaluator_rejects_artifact_mismatch_before_generation(self):
        parent = load_stage4_calibration_artifact(PARENT_ARTIFACT)
        config = OperatingDecisionFinalConfig()
        calibration = ProcedureCalibration(
            procedure_name="mismatch_guarded_selector_v2",
            block_count=20,
            conformal_rank=19,
            target_block_coverage=0.90,
            additive_margin_padding=parent.procedure_for(
                "decision_directed_selector"
            ).additive_margin_padding,
            block_nonconformity=(0.0,) * 20,
            emitted_interval_count=0,
            raw_interval_covered_count=0,
            calibrated_interval_covered_count=0,
        )
        artifact = _build_artifact(
            source_revision="test-revision",
            parent=parent,
            config=config,
            guard=self._guard(),
            calibration=calibration,
        )._replace(protocol_digest="corrupt")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.json"
            save_revised_artifact(artifact, path)
            with patch(
                "thermotwin.studies.operating_decision_final_evaluation._run_truths"
            ) as generator:
                with self.assertRaisesRegex(ValueError, "artifact digest"):
                    evaluate_reserved_stage5(path, parent, config=config)
                generator.assert_not_called()

    def test_valid_serialized_artifact_crosses_generation_boundary(self):
        parent = load_stage4_calibration_artifact(PARENT_ARTIFACT)
        config = OperatingDecisionFinalConfig()
        calibration = ProcedureCalibration(
            procedure_name="mismatch_guarded_selector_v2",
            block_count=20,
            conformal_rank=19,
            target_block_coverage=0.90,
            additive_margin_padding=parent.procedure_for(
                "decision_directed_selector"
            ).additive_margin_padding,
            block_nonconformity=(0.0,) * 20,
            emitted_interval_count=0,
            raw_interval_covered_count=0,
            calibrated_interval_covered_count=0,
        )
        artifact = _build_artifact(
            source_revision="test-revision",
            parent=parent,
            config=config,
            guard=self._guard(),
            calibration=calibration,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.json"
            save_revised_artifact(artifact, path)
            with patch(
                "thermotwin.studies.operating_decision_final_evaluation._run_truths",
                side_effect=RuntimeError("generation boundary reached"),
            ) as generator:
                with self.assertRaisesRegex(RuntimeError, "generation boundary"):
                    evaluate_reserved_stage5(path, parent, config=config)
                generator.assert_called_once()

    def test_bootstrap_change_is_rejected_before_stage5_generation(self):
        parent = load_stage4_calibration_artifact(PARENT_ARTIFACT)
        config = OperatingDecisionFinalConfig()
        artifact = _build_artifact(
            source_revision="test-revision",
            parent=parent,
            config=config,
            guard=self._guard(),
            calibration=self._valid_calibration(parent),
        )
        changed = replace(config, bootstrap_seed=80_191_001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            save_revised_artifact(artifact, path)
            with patch(
                "thermotwin.studies.operating_decision_final_evaluation._run_truths"
            ) as generator:
                with self.assertRaisesRegex(ValueError, "digest or protocol"):
                    evaluate_reserved_stage5(path, parent, config=changed)
                generator.assert_not_called()

    def test_semantically_invalid_artifact_is_rejected_even_with_matching_digest(self):
        parent = load_stage4_calibration_artifact(PARENT_ARTIFACT)
        config = OperatingDecisionFinalConfig()
        calibration = self._valid_calibration(parent)
        valid = _build_artifact(
            source_revision="test-revision",
            parent=parent,
            config=config,
            guard=self._guard(),
            calibration=calibration,
        )
        cases = (
            (
                "bad-guard.json",
                valid.guard._replace(threshold=999.0),
                calibration,
                "guard threshold",
            ),
            (
                "bad-padding.json",
                valid.guard,
                calibration._replace(additive_margin_padding=0.0),
                "procedure padding",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for filename, guard, candidate_calibration, message in cases:
                with self.subTest(filename=filename):
                    digest = _artifact_digest(
                        source_revision=valid.implementation_revision,
                        parent=parent,
                        config=config,
                        guard=guard,
                        calibration=candidate_calibration,
                    )
                    artifact = valid._replace(
                        protocol_digest=digest,
                        guard=guard,
                        procedure_calibration=candidate_calibration,
                    )
                    path = Path(directory) / filename
                    save_revised_artifact(artifact, path)
                    with patch(
                        "thermotwin.studies.operating_decision_final_evaluation._run_truths"
                    ) as generator:
                        with self.assertRaisesRegex(ValueError, message):
                            evaluate_reserved_stage5(path, parent, config=config)
                        generator.assert_not_called()

    def test_cli_preflight_refuses_output_overwrite_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "existing-result.json"
            result_path.write_text("already generated\n", encoding="utf-8")
            with patch(
                "thermotwin.reports.operating_decision_final_evaluation.evaluate_reserved_stage5"
            ) as evaluator:
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    final_evaluation_main(
                        (
                            "evaluate",
                            "--parent-artifact",
                            str(PARENT_ARTIFACT),
                            "--artifact",
                            str(Path(directory) / "artifact.json"),
                            "--result-output",
                            str(result_path),
                        )
                    )
                evaluator.assert_not_called()
        with self.assertRaisesRegex(ValueError, "aliases an input"):
            _preflight_output_paths(
                input_paths=(PARENT_ARTIFACT,),
                output_paths=(PARENT_ARTIFACT,),
            )

    def test_monotone_audit_rejects_a_new_or_reversed_decision(self):
        parent = self._procedure_outcome(DECISION_DIRECTED_SELECTOR)
        safe_abstention = self._procedure_outcome(
            MISMATCH_GUARDED_SELECTOR,
            selected_policy=EARLY_MISMATCH_ABSTENTION,
            decision=INSUFFICIENT_EVIDENCE,
        )
        _validate_monotone_revision((parent, safe_abstention))
        reversed_decision = safe_abstention._replace(
            selected_policy=STOP_NOW,
            decision=REJECT,
        )
        with self.assertRaisesRegex(ValueError, "introduced or reversed"):
            _validate_monotone_revision((parent, reversed_decision))

    def test_paired_bootstrap_is_deterministic_and_final_json_is_finite(self):
        config = replace(OperatingDecisionFinalConfig(), bootstrap_draws=1_000)
        outcomes = []
        for truth in (
            "matched_four_state",
            "extra_interface_mass",
            "temperature_dependent_contact",
        ):
            for trial_index in range(config.evaluation.sensor.trial_count):
                outcomes.extend(
                    (
                        self._procedure_outcome(
                            STOP_NOW,
                            truth_condition=truth,
                            trial_index=trial_index,
                        ),
                        self._procedure_outcome(
                            DECISION_DIRECTED_SELECTOR,
                            truth_condition=truth,
                            trial_index=trial_index,
                            selected_policy=FIXED_VOLTAGE,
                            decision=INSUFFICIENT_EVIDENCE,
                            diagnostic_run_count=3,
                            energy=30.0,
                        ),
                        self._procedure_outcome(
                            MISMATCH_GUARDED_SELECTOR,
                            truth_condition=truth,
                            trial_index=trial_index,
                            selected_policy=EARLY_MISMATCH_ABSTENTION,
                            decision=INSUFFICIENT_EVIDENCE,
                            diagnostic_run_count=1,
                            energy=10.0,
                        ),
                    )
                )
        first = _paired_loss_comparisons(outcomes, config)
        self.assertEqual(first, _paired_loss_comparisons(outcomes, config))
        self.assertTrue(all(item.revised_minus_parent_mean < 0.0 for item in first))

        parent_artifact = load_stage4_calibration_artifact(PARENT_ARTIFACT)
        artifact = _build_artifact(
            source_revision="test-revision",
            parent=parent_artifact,
            config=config,
            guard=self._guard(),
            calibration=self._valid_calibration(parent_artifact),
        )
        result = FinalEvaluationResult(
            config=config,
            artifact=artifact,
            outcomes=tuple(outcomes),
            summaries=(),
            paired_loss_comparisons=first,
            guard_summaries=(),
        )
        payload = final_result_payload(result)
        json.dumps(payload, allow_nan=False)
        self.assertEqual(
            payload["artifact"]["analysis"]["paired_block_bootstrap"]["draws"],
            1_000,
        )
        self.assertTrue(payload["evaluation"]["instantiated"])

    def test_rule_has_no_learned_family_specific_parameters(self):
        self.assertEqual(
            tuple(item.name for item in fields(RevisedSelectorRule)),
            ("mismatch_policy", "resolved_policy", "fallback_policy"),
        )


if __name__ == "__main__":
    unittest.main()
