import inspect
import json
import math
import unittest
from typing import NamedTuple
from unittest.mock import patch

from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.studies.operating_decision import (
    FINAL_EVALUATION,
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    POLICY_NAMES,
    STOP_NOW,
    MarginEnvelope,
    MarginInterval,
    NumericalFailure,
    OperatingRegime,
    default_fixed_policies,
)
from thermotwin.studies.operating_decision_calibration import (
    STAGE4_PROCEDURES,
    SelectorRule,
    select_fixed_policy,
)
from thermotwin.studies.operating_decision_prospective import (
    PROSPECTIVE_ACTIONS,
    PROSPECTIVE_FOUR_ACTION_SELECTOR,
    CandidateExclusion,
    ProspectiveAcquisitionSnapshot,
    ProspectiveActionEvaluation,
    ProspectiveDevelopmentOffsets,
    ProspectiveSelectorRule,
    build_prospective_acquisition_snapshot,
    prospective_selector_protocol_digest,
    prospective_selector_rule_from_payload,
    prospective_selector_rule_payload,
    select_prospective_action,
    uncomputed_action_evaluations,
)
from thermotwin.studies.operating_decision_replication_guard import (
    CORRECTED_FINAL_PROCEDURES,
)
from thermotwin.studies.operating_decision_realism import (
    OperatingDecisionRealismConfig,
    RealisticAcquisitionFitSet,
)
from thermotwin.studies.sensor_model_discrimination import (
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
)


class _Fit(NamedTuple):
    model_name: str
    reached_bound: bool
    converged: bool


def _evaluation(
    policy_name,
    *,
    before=1.0,
    after=0.5,
    cost=1.0,
    eligible=True,
    raw_before=None,
    raw_after=None,
    development_offset=None,
):
    if policy_name == STOP_NOW:
        return ProspectiveActionEvaluation(STOP_NOW, True)
    if not eligible:
        return ProspectiveActionEvaluation(policy_name, False, "unavailable")
    return ProspectiveActionEvaluation(
        policy_name=policy_name,
        eligible=True,
        uncertainty_before=before,
        expected_uncertainty_after=after,
        declared_cost=cost,
        prospective_draw_count=8,
        raw_uncertainty_before=raw_before,
        raw_expected_uncertainty_after=raw_after,
        development_offset=development_offset,
    )


def _evaluations(*, thermal=0.4, voltage=0.8, face=0.6):
    return (
        _evaluation(STOP_NOW),
        _evaluation(FIXED_THERMAL, after=1.0 - thermal),
        _evaluation(FIXED_VOLTAGE, after=1.0 - voltage),
        _evaluation(FIXED_FACE_TEMPERATURE, after=1.0 - face),
    )


def _acquisition_snapshot(envelope):
    interval = MarginInterval(
        FOUR_STATE_MODEL,
        0.5 * (envelope.lower + envelope.upper),
        0.05,
        envelope.lower,
        envelope.upper,
    )
    return ProspectiveAcquisitionSnapshot(
        candidate_models=tuple(sorted((FOUR_STATE_MODEL, FIVE_STATE_MODEL))),
        admissible_candidate_models=(FOUR_STATE_MODEL,),
        excluded_candidates=(
            CandidateExclusion(FIVE_STATE_MODEL, "fit_reached_bound"),
        ),
        failed_candidate_models=(),
        model_intervals=(interval,),
        provisional_margin_envelope=envelope,
        selection_failure_reason=None,
    )


class OperatingDecisionProspectiveTests(unittest.TestCase):
    def test_action_menu_reuses_the_four_frozen_policies(self):
        policies = default_fixed_policies()
        self.assertEqual(PROSPECTIVE_ACTIONS, POLICY_NAMES)
        self.assertEqual(tuple(item.name for item in policies), PROSPECTIVE_ACTIONS)
        self.assertEqual(tuple(len(item.additional_regimes) for item in policies), (0, 3, 1, 1))
        self.assertEqual(tuple(item.extra_sensor_count for item in policies), (0, 0, 1, 1))

    def test_selector_interface_contains_only_acquisition_time_information(self):
        forbidden = {
            "truth_condition",
            "truth_family",
            "block",
            "trial_index",
            "device_token",
            "verification_run",
            "verification_score",
            "selected_policy_outcome",
            "true_margin",
            "final_response",
        }
        self.assertFalse(
            forbidden.intersection(ProspectiveAcquisitionSnapshot.__dataclass_fields__)
        )
        parameters = set(
            inspect.signature(build_prospective_acquisition_snapshot).parameters
        )
        self.assertEqual(
            parameters,
            {"fit_set", "final_regime", "physical_config"},
        )

    def test_bound_hit_and_nonconverged_candidates_are_excluded_individually(self):
        config = OperatingDecisionRealismConfig()
        final_regime = OperatingRegime(
            "untouched_final_operating_schedule",
            FINAL_EVALUATION,
            config.final_current,
            (),
        )
        interval = MarginInterval(FOUR_STATE_MODEL, 0.2, 0.05, 0.1, 0.3)
        reliable = _Fit(FOUR_STATE_MODEL, False, True)

        for unreliable in (
            _Fit(FIVE_STATE_MODEL, True, True),
            _Fit(FIVE_STATE_MODEL, False, False),
        ):
            fit_set = RealisticAcquisitionFitSet((unreliable, reliable), ())
            with patch(
                "thermotwin.studies.operating_decision_prospective."
                "forecast_realistic_margin_interval",
                return_value=interval,
            ) as forecast:
                snapshot = build_prospective_acquisition_snapshot(
                    fit_set,
                    final_regime,
                    config,
                )
            self.assertEqual(snapshot.admissible_candidate_models, (FOUR_STATE_MODEL,))
            self.assertEqual(snapshot.model_intervals, (interval,))
            self.assertEqual(
                snapshot.excluded_candidates[0].model_name,
                FIVE_STATE_MODEL,
            )
            self.assertIsNone(snapshot.selection_failure_reason)
            forecast.assert_called_once()

        all_bad = RealisticAcquisitionFitSet(
            (
                _Fit(FOUR_STATE_MODEL, True, True),
                _Fit(FIVE_STATE_MODEL, False, False),
            ),
            (),
        )
        snapshot = build_prospective_acquisition_snapshot(
            all_bad,
            final_regime,
            config,
        )
        selection = select_prospective_action(
            snapshot,
            uncomputed_action_evaluations(),
        )
        self.assertEqual(snapshot.selection_failure_reason, "no_admissible_candidate")
        self.assertIsNone(selection.selected_policy)
        self.assertFalse(selection.verification_required)

    def test_acquisition_snapshot_accounts_for_failed_candidates(self):
        config = OperatingDecisionRealismConfig()
        final_regime = OperatingRegime(
            "untouched_final_operating_schedule",
            FINAL_EVALUATION,
            config.final_current,
            (),
        )
        reliable = _Fit(FOUR_STATE_MODEL, False, True)
        fit_set = RealisticAcquisitionFitSet(
            (reliable,),
            (NumericalFailure(FIVE_STATE_MODEL, "acquisition_fit", "ValueError"),),
        )
        snapshot = build_prospective_acquisition_snapshot(
            fit_set,
            final_regime,
            config,
        )
        self.assertEqual(snapshot.failed_candidate_models, (FIVE_STATE_MODEL,))
        self.assertEqual(snapshot.selection_failure_reason, "acquisition_fit_failure")
        self.assertEqual(snapshot.model_intervals, ())

        incomplete = RealisticAcquisitionFitSet((reliable,), ())
        with self.assertRaisesRegex(ValueError, "account for every candidate"):
            build_prospective_acquisition_snapshot(
                incomplete,
                final_regime,
                config,
            )

        wrong_target = OperatingRegime(
            "wrong_final_operating_schedule",
            FINAL_EVALUATION,
            PiecewiseConstantCurrent.constant(0.0),
            (),
        )
        with self.assertRaisesRegex(ValueError, "frozen untouched final regime"):
            build_prospective_acquisition_snapshot(
                fit_set,
                wrong_target,
                config,
            )

    def test_resolved_envelope_stops_but_still_requires_verification(self):
        selection = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(0.10, 1.10)),
            _evaluations(),
        )
        self.assertEqual(selection.selected_policy, STOP_NOW)
        self.assertTrue(selection.selection_succeeded)
        self.assertFalse(selection.requires_additional_acquisition)
        self.assertTrue(selection.verification_required)
        self.assertEqual(selection.reason, "provisionally_resolved_stop_then_verify")

    def test_unresolved_envelope_selects_maximum_value_per_cost(self):
        selection = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(-0.10, 0.90)),
            _evaluations(),
        )
        self.assertEqual(selection.selected_policy, FIXED_VOLTAGE)
        self.assertTrue(selection.requires_additional_acquisition)
        self.assertEqual(selection.ranked_policies[0], FIXED_VOLTAGE)
        self.assertEqual(
            selection.reason,
            "maximum_expected_uncertainty_reduction_per_cost",
        )

    def test_exact_ties_use_the_frozen_action_order(self):
        evaluations = (
            _evaluation(STOP_NOW),
            _evaluation(FIXED_THERMAL, before=1.0, after=0.5, cost=0.5),
            _evaluation(FIXED_VOLTAGE, before=1.0, after=0.5, cost=0.5),
            _evaluation(
                FIXED_FACE_TEMPERATURE,
                before=1.0,
                after=0.5,
                cost=0.5,
            ),
        )
        selection = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(-0.10, 0.90)),
            evaluations,
        )
        self.assertEqual(selection.selected_policy, FIXED_THERMAL)
        self.assertEqual(
            selection.tied_policies,
            (FIXED_THERMAL, FIXED_VOLTAGE, FIXED_FACE_TEMPERATURE),
        )
        self.assertEqual(
            selection.reason,
            "maximum_value_per_cost_with_frozen_tie_break",
        )

    def test_secondary_merit_breaks_equal_utility_before_action_order(self):
        evaluations = (
            _evaluation(STOP_NOW),
            _evaluation(FIXED_THERMAL, after=0.5, cost=0.5),
            _evaluation(FIXED_VOLTAGE, after=0.2, cost=0.8),
            _evaluation(FIXED_FACE_TEMPERATURE, after=0.8, cost=0.2),
        )
        selection = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(-0.10, 0.90)),
            evaluations,
        )
        self.assertEqual(selection.selected_policy, FIXED_VOLTAGE)
        self.assertEqual(selection.tied_policies, (FIXED_VOLTAGE,))
        self.assertEqual(
            selection.reason,
            "maximum_expected_uncertainty_reduction_per_cost",
        )

    def test_no_valuable_action_selects_stop_and_still_verifies(self):
        evaluations = (
            _evaluation(STOP_NOW),
            _evaluation(FIXED_THERMAL, before=1.0, after=1.0),
            _evaluation(FIXED_VOLTAGE, before=1.0, after=1.0),
            _evaluation(FIXED_FACE_TEMPERATURE, before=1.0, after=1.0),
        )
        selection = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(-0.10, 0.90)),
            evaluations,
        )
        self.assertEqual(selection.selected_policy, STOP_NOW)
        self.assertTrue(selection.selection_succeeded)
        self.assertFalse(selection.requires_additional_acquisition)
        self.assertTrue(selection.verification_required)
        self.assertEqual(
            selection.reason,
            "no_valuable_acquisition_stop_then_verify",
        )

    def test_uncomputed_scorecard_is_a_selection_failure(self):
        selection = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(-0.10, 0.30)),
            uncomputed_action_evaluations(),
        )
        self.assertIsNone(selection.selected_policy)
        self.assertFalse(selection.selection_succeeded)
        self.assertFalse(selection.verification_required)
        self.assertEqual(selection.reason, "no_scored_acquisition_action")

    def test_padded_stop_gate_records_single_candidate_surcharge(self):
        offsets = ProspectiveDevelopmentOffsets(
            version="development_probe_v1",
            stop_now=0.05,
        )
        rule = ProspectiveSelectorRule(
            development_offsets=offsets,
            stopping_clearance=0.05,
            single_candidate_stopping_clearance=0.03,
        )
        snapshot = _acquisition_snapshot(MarginEnvelope(0.12, 0.30))
        padded_evaluations = (
            _evaluation(STOP_NOW),
            _evaluation(
                FIXED_THERMAL,
                before=0.28,
                after=0.24,
                raw_before=0.18,
                raw_after=0.24,
                development_offset=0.0,
            ),
            _evaluation(
                FIXED_VOLTAGE,
                before=0.28,
                after=0.20,
                raw_before=0.18,
                raw_after=0.20,
                development_offset=0.0,
            ),
            _evaluation(
                FIXED_FACE_TEMPERATURE,
                before=0.28,
                after=0.22,
                raw_before=0.18,
                raw_after=0.22,
                development_offset=0.0,
            ),
        )
        selection = select_prospective_action(snapshot, padded_evaluations, rule)
        self.assertEqual(selection.selected_policy, FIXED_VOLTAGE)
        self.assertAlmostEqual(
            selection.padded_initial_margin_envelope.lower,
            0.07,
        )
        self.assertAlmostEqual(
            selection.padded_initial_margin_envelope.upper,
            0.35,
        )
        self.assertEqual(selection.admissible_candidate_count, 1)
        self.assertEqual(
            selection.candidate_reliability_stratum,
            "one_admissible_candidate",
        )
        self.assertEqual(selection.effective_stopping_clearance, 0.08)
        self.assertEqual(
            selection.selector_protocol_digest,
            prospective_selector_protocol_digest(rule),
        )

        resolved = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(0.14, 0.30)),
            (
                _evaluation(STOP_NOW),
                _evaluation(
                    FIXED_THERMAL,
                    before=0.26,
                    after=0.22,
                    raw_before=0.16,
                    raw_after=0.22,
                    development_offset=0.0,
                ),
                _evaluation(
                    FIXED_VOLTAGE,
                    before=0.26,
                    after=0.18,
                    raw_before=0.16,
                    raw_after=0.18,
                    development_offset=0.0,
                ),
                _evaluation(
                    FIXED_FACE_TEMPERATURE,
                    before=0.26,
                    after=0.20,
                    raw_before=0.16,
                    raw_after=0.20,
                    development_offset=0.0,
                ),
            ),
            rule,
        )
        self.assertEqual(resolved.selected_policy, STOP_NOW)
        self.assertTrue(resolved.verification_required)

    def test_rule_digest_binds_offsets_and_single_candidate_clearance(self):
        first = ProspectiveSelectorRule(
            development_offsets=ProspectiveDevelopmentOffsets(
                version="development_offsets_v1",
                fixed_voltage=0.02,
            ),
            single_candidate_stopping_clearance=0.01,
        )
        second = ProspectiveSelectorRule(
            development_offsets=ProspectiveDevelopmentOffsets(
                version="development_offsets_v2",
                fixed_voltage=0.02,
            ),
            single_candidate_stopping_clearance=0.01,
        )
        third = ProspectiveSelectorRule(
            development_offsets=first.development_offsets,
            single_candidate_stopping_clearance=0.02,
        )
        self.assertNotEqual(
            prospective_selector_protocol_digest(first),
            prospective_selector_protocol_digest(second),
        )
        self.assertNotEqual(
            prospective_selector_protocol_digest(first),
            prospective_selector_protocol_digest(third),
        )
        self.assertEqual(
            prospective_selector_rule_from_payload(
                prospective_selector_rule_payload(first)
            ),
            first,
        )

    def test_reserved_unfitted_offset_version_requires_zero_values(self):
        with self.assertRaisesRegex(ValueError, "unfitted-zero"):
            ProspectiveDevelopmentOffsets(stop_now=0.01)

    def test_selector_rejects_mispaired_or_incorrectly_padded_baselines(self):
        snapshot = _acquisition_snapshot(MarginEnvelope(-0.10, 0.90))
        mispaired = list(_evaluations())
        mispaired[1] = _evaluation(
            FIXED_THERMAL,
            before=1.0,
            after=0.6,
            raw_before=0.9,
            raw_after=0.6,
            development_offset=0.0,
        )
        with self.assertRaisesRegex(ValueError, "raw uncertainty baseline"):
            select_prospective_action(snapshot, mispaired)

        offsets = ProspectiveDevelopmentOffsets(
            version="development_offsets_v1",
            stop_now=0.05,
        )
        with self.assertRaisesRegex(ValueError, "padded uncertainty baseline"):
            select_prospective_action(
                snapshot,
                _evaluations(),
                ProspectiveSelectorRule(development_offsets=offsets),
            )

    def test_thresholds_select_the_best_qualifying_action(self):
        evaluations = (
            _evaluation(STOP_NOW),
            _evaluation(FIXED_THERMAL, after=0.8, cost=0.1),
            _evaluation(FIXED_VOLTAGE, after=0.6, cost=1.0),
            _evaluation(FIXED_FACE_TEMPERATURE, after=0.9, cost=1.0),
        )
        selection = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(-0.10, 0.90)),
            evaluations,
            ProspectiveSelectorRule(
                minimum_expected_reduction=0.3,
                minimum_utility_per_cost=0.1,
            ),
        )
        self.assertEqual(selection.ranked_policies[0], FIXED_THERMAL)
        self.assertEqual(selection.selected_policy, FIXED_VOLTAGE)

    def test_all_eligible_actions_share_the_common_uncertainty_baseline(self):
        evaluations = list(_evaluations())
        evaluations[2] = _evaluation(
            FIXED_VOLTAGE,
            before=2.0,
            after=1.2,
        )
        with self.assertRaisesRegex(ValueError, "one uncertainty baseline"):
            select_prospective_action(
                _acquisition_snapshot(MarginEnvelope(-0.10, 0.30)),
                evaluations,
            )

    def test_selection_boundaries_are_frozen(self):
        snapshot = _acquisition_snapshot(MarginEnvelope(-0.10, 0.90))
        evaluations = (
            _evaluation(STOP_NOW),
            _evaluation(FIXED_THERMAL, after=0.6),
            _evaluation(FIXED_VOLTAGE, after=0.7),
            _evaluation(FIXED_FACE_TEMPERATURE, after=0.8),
        )
        at_threshold = select_prospective_action(
            snapshot,
            evaluations,
            ProspectiveSelectorRule(
                minimum_expected_reduction=0.4,
                minimum_utility_per_cost=0.4,
            ),
        )
        self.assertEqual(at_threshold.selected_policy, STOP_NOW)

        below_threshold = select_prospective_action(
            snapshot,
            evaluations,
            ProspectiveSelectorRule(
                minimum_expected_reduction=0.399999999999999,
                minimum_utility_per_cost=0.399999999999999,
                utility_decimal_places=15,
            ),
        )
        self.assertEqual(below_threshold.selected_policy, FIXED_THERMAL)

        positive_boundary = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(0.10, 1.10)),
            evaluations,
            ProspectiveSelectorRule(stopping_clearance=0.10),
        )
        self.assertEqual(positive_boundary.selected_policy, STOP_NOW)

        negative_boundary = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(-1.10, -0.10)),
            evaluations,
            ProspectiveSelectorRule(stopping_clearance=0.10),
        )
        self.assertEqual(negative_boundary.selected_policy, FIXED_THERMAL)
        below_negative_boundary = select_prospective_action(
            _acquisition_snapshot(
                MarginEnvelope(
                    math.nextafter(-0.10, -math.inf) - 1.0,
                    math.nextafter(-0.10, -math.inf),
                )
            ),
            evaluations,
            ProspectiveSelectorRule(stopping_clearance=0.10),
        )
        self.assertEqual(below_negative_boundary.selected_policy, STOP_NOW)

    def test_extreme_finite_scores_quantize_or_fail_cleanly(self):
        evaluations = (
            _evaluation(STOP_NOW),
            _evaluation(FIXED_THERMAL, before=1.0e308, after=0.0, cost=1.0e308),
            _evaluation(FIXED_VOLTAGE, before=1.0e308, after=5.0e307, cost=5.0e307),
            _evaluation(
                FIXED_FACE_TEMPERATURE,
                before=1.0e308,
                after=7.5e307,
                cost=2.5e307,
            ),
        )
        selection = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(-5.0e307, 5.0e307)),
            evaluations,
        )
        self.assertEqual(selection.selected_policy, FIXED_THERMAL)

        with self.assertRaisesRegex(ValueError, "derived utility"):
            _evaluation(FIXED_VOLTAGE, before=1.0, after=0.0, cost=5.0e-324)

    def test_selection_output_copies_mutable_scorecard_inputs(self):
        evaluations = list(_evaluations())
        selection = select_prospective_action(
            _acquisition_snapshot(MarginEnvelope(-0.10, 0.90)),
            evaluations,
        )
        evaluations.append(_evaluation(FIXED_VOLTAGE))
        self.assertEqual(len(selection.action_evaluations), 4)

    def test_action_records_and_rules_reject_malformed_values(self):
        with self.assertRaisesRegex(ValueError, "positive declared cost"):
            _evaluation(FIXED_VOLTAGE, cost=0.0)
        with self.assertRaisesRegex(ValueError, "finite number"):
            ProspectiveActionEvaluation(
                FIXED_VOLTAGE,
                True,
                uncertainty_before=1.0,
                expected_uncertainty_after=math.nan,
                declared_cost=1.0,
                prospective_draw_count=8,
            )
        with self.assertRaisesRegex(ValueError, "prospective draws"):
            ProspectiveActionEvaluation(
                FIXED_VOLTAGE,
                True,
                uncertainty_before=1.0,
                expected_uncertainty_after=0.5,
                declared_cost=1.0,
            )
        with self.assertRaisesRegex(ValueError, "action order is frozen"):
            ProspectiveSelectorRule(
                action_order=(
                    STOP_NOW,
                    FIXED_VOLTAGE,
                    FIXED_THERMAL,
                    FIXED_FACE_TEMPERATURE,
                )
            )
        with self.assertRaisesRegex(ValueError, "canonical order"):
            select_prospective_action(
                _acquisition_snapshot(MarginEnvelope(-0.1, 0.2)),
                tuple(reversed(_evaluations())),
            )

    def test_rule_payload_round_trip_is_strict_and_finite(self):
        rule = ProspectiveSelectorRule(stopping_clearance=0.01)
        payload = prospective_selector_rule_payload(rule)
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        self.assertTrue(encoded)
        self.assertEqual(prospective_selector_rule_from_payload(payload), rule)

        changed = dict(payload)
        changed["protocol_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            prospective_selector_rule_from_payload(changed)
        missing = dict(payload)
        missing.pop("action_order")
        with self.assertRaisesRegex(ValueError, "malformed"):
            prospective_selector_rule_from_payload(missing)
        boolean_schema = dict(payload)
        boolean_schema["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "schema version"):
            prospective_selector_rule_from_payload(boolean_schema)
        self.assertNotEqual(
            prospective_selector_protocol_digest(rule),
            prospective_selector_protocol_digest(
                ProspectiveSelectorRule(stopping_clearance=0.02)
            ),
        )
        canonical = prospective_selector_protocol_digest(
            ProspectiveSelectorRule(stopping_clearance=0)
        )
        self.assertEqual(
            canonical,
            prospective_selector_protocol_digest(
                ProspectiveSelectorRule(stopping_clearance=0.0)
            ),
        )
        self.assertEqual(
            canonical,
            prospective_selector_protocol_digest(
                ProspectiveSelectorRule(stopping_clearance=-0.0)
            ),
        )

    def test_completed_r2_selector_protocol_is_unchanged(self):
        self.assertEqual(
            POLICY_NAMES,
            (STOP_NOW, FIXED_THERMAL, FIXED_VOLTAGE, FIXED_FACE_TEMPERATURE),
        )
        self.assertIn("decision_directed_selector", STAGE4_PROCEDURES)
        self.assertNotIn(PROSPECTIVE_FOUR_ACTION_SELECTOR, STAGE4_PROCEDURES)
        self.assertNotIn(PROSPECTIVE_FOUR_ACTION_SELECTOR, CORRECTED_FINAL_PROCEDURES)
        self.assertEqual(SelectorRule().fallback_policy, FIXED_VOLTAGE)
        unresolved = type(
            "Signal",
            (),
            {"margin_envelope": MarginEnvelope(-0.1, 0.2)},
        )()
        self.assertEqual(select_fixed_policy(unresolved), FIXED_VOLTAGE)


if __name__ == "__main__":
    unittest.main()
