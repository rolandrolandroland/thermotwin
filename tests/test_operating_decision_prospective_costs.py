from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from thermotwin.studies.operating_decision import (
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    POLICY_NAMES,
    RUN_DURATION_SECONDS,
    STOP_NOW,
    VERIFICATION,
    MarginEnvelope,
    MarginInterval,
)
import thermotwin.studies.operating_decision_calibration as calibration
from thermotwin.studies.operating_decision_prospective import (
    CandidateExclusion,
    ProspectiveAcquisitionSnapshot,
    ProspectiveDevelopmentOffsets,
    ProspectiveSelectorRule,
    prospective_selector_protocol_digest,
)
from thermotwin.studies.operating_decision_prospective_uncertainty import (
    ProspectiveActionUncertainty,
    ProspectiveModelActionSummary,
    ProspectiveUncertaintyConfig,
    ProspectiveUncertaintyResult,
    prospective_uncertainty_protocol_digest,
)
from thermotwin.studies.operating_decision_replication import (
    corrected_physical_protocol_digest,
)
import thermotwin.studies.operating_decision_replication_guard as replication_guard
from thermotwin.studies.operating_decision_realism import (
    OperatingDecisionRealismConfig,
)
from thermotwin.studies.sensor_model_discrimination import (
    COLD_FACE,
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
    MODEL_NAMES,
    VOLTAGE,
)
import thermotwin.studies.operating_decision_prospective_costs as costs
import thermotwin.studies.operating_decision_prospective_uncertainty as prospective_uncertainty


_ENERGY_BY_REGIME = {
    "initial_0.8A_20s": 10.0,
    "thermal_0.6A_30s": 3.0,
    "thermal_0.6A_15s": 4.0,
    "thermal_0.4A_5s": 5.0,
    "repeat_0.8A_20s_voltage": 7.0,
    "repeat_0.8A_20s_face_temperature": 8.0,
}


def _fake_nominal_energy(call_log=None):
    def evaluate(regime, instrumentation, _physical_config):
        if call_log is not None:
            call_log.append((regime, instrumentation))
        if regime.name == "fixed_verification":
            energy = 25.0 if instrumentation.temporary_face_sensor else 20.0
        else:
            energy = _ENERGY_BY_REGIME[regime.name]
        return SimpleNamespace(terminal_energy=energy)

    return evaluate


def _snapshot() -> ProspectiveAcquisitionSnapshot:
    interval = MarginInterval(
        FOUR_STATE_MODEL,
        0.0,
        0.5,
        -1.0,
        1.0,
    )
    return ProspectiveAcquisitionSnapshot(
        candidate_models=tuple(sorted(MODEL_NAMES)),
        admissible_candidate_models=(FOUR_STATE_MODEL,),
        excluded_candidates=(
            CandidateExclusion(FIVE_STATE_MODEL, "fit_reached_bound"),
        ),
        failed_candidate_models=(),
        model_intervals=(interval,),
        provisional_margin_envelope=MarginEnvelope(-1.0, 1.0),
        selection_failure_reason=None,
    )


def _uncertainty(
    policy_name: str,
    *,
    before: float = 2.0,
    after: float = 1.0,
    eligible: bool = True,
) -> ProspectiveActionUncertainty:
    if policy_name == STOP_NOW:
        return ProspectiveActionUncertainty(
            policy_name=STOP_NOW,
            eligible=True,
            failure_reason=None,
            uncertainty_before=before,
            expected_uncertainty_after=before,
            prospective_draw_count=0,
            source_summaries=(),
        )
    summary = ProspectiveModelActionSummary(
        policy_name=policy_name,
        generator_model=FOUR_STATE_MODEL,
        draw_count=4,
        stable_draw_count=4 if eligible else 2,
        failed_draw_count=0 if eligible else 2,
        mean_after_width=after,
        eligible=eligible,
    )
    return ProspectiveActionUncertainty(
        policy_name=policy_name,
        eligible=eligible,
        failure_reason=(
            None if eligible else "insufficient_stable_prospective_draws"
        ),
        uncertainty_before=before,
        expected_uncertainty_after=after,
        prospective_draw_count=4,
        source_summaries=(summary,),
    )


def _uncertainties(
    *,
    thermal_after: float = 0.5,
    voltage_after: float = 2.5,
    face_after: float = 1.5,
    face_eligible: bool = True,
):
    return (
        _uncertainty(STOP_NOW),
        _uncertainty(FIXED_THERMAL, after=thermal_after),
        _uncertainty(FIXED_VOLTAGE, after=voltage_after),
        _uncertainty(
            FIXED_FACE_TEMPERATURE,
            after=face_after,
            eligible=face_eligible,
        ),
    )


def _uncertainty_result(
    physical_config: OperatingDecisionRealismConfig,
    *,
    uncertainties=None,
    result_digest: str = "b" * 64,
) -> ProspectiveUncertaintyResult:
    uncertainty_config = ProspectiveUncertaintyConfig(draw_count=4)
    evidence = SimpleNamespace(
        snapshot=_snapshot(),
        evidence_digest="a" * 64,
        physical_protocol_digest=corrected_physical_protocol_digest(
            physical_config
        ),
    )
    result = object.__new__(ProspectiveUncertaintyResult)
    values = {
        "config": uncertainty_config,
        "physical_config": physical_config,
        "acquisition_evidence": evidence,
        "protocol_digest": prospective_uncertainty_protocol_digest(
            physical_config,
            uncertainty_config,
        ),
        "action_uncertainties": (
            _uncertainties() if uncertainties is None else tuple(uncertainties)
        ),
        "draw_outcomes": (),
        "stream_uses": (),
        "stream_audit": None,
        "result_digest": result_digest,
    }
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


@contextmanager
def _cost_environment(call_log=None):
    with (
        patch.object(
            costs,
            "nominal_selection_cost_proxy",
            side_effect=_fake_nominal_energy(call_log),
        ),
        patch.object(
            costs,
            "validate_prospective_uncertainty_result_integrity",
            return_value=None,
        ),
    ):
        yield


class ProspectiveCostResourceTests(unittest.TestCase):
    def setUp(self):
        self.physical = OperatingDecisionRealismConfig()

    def test_raw_resources_cover_the_complete_plan_relative_to_stop(self):
        calls = []
        with _cost_environment(calls):
            resources = costs.prospective_action_resources(self.physical)
        self.assertEqual(tuple(item.policy_name for item in resources), POLICY_NAMES)
        by_policy = {item.policy_name: item for item in resources}

        self.assertEqual(len(by_policy[STOP_NOW].plan_runs), 2)
        self.assertEqual(len(by_policy[FIXED_THERMAL].plan_runs), 5)
        self.assertEqual(len(by_policy[FIXED_VOLTAGE].plan_runs), 3)
        self.assertEqual(len(by_policy[FIXED_FACE_TEMPERATURE].plan_runs), 3)
        for item in resources:
            self.assertEqual(item.plan_runs[0].regime_name, "initial_0.8A_20s")
            self.assertEqual(item.plan_runs[-1].phase, VERIFICATION)
            self.assertEqual(item.stop_reference_plan_energy_joules, 30.0)

        self.assertEqual(
            by_policy[STOP_NOW].nominal_total_plan_energy_joules,
            30.0,
        )
        self.assertEqual(
            by_policy[FIXED_THERMAL].nominal_total_plan_energy_joules,
            42.0,
        )
        self.assertEqual(
            by_policy[FIXED_VOLTAGE].nominal_total_plan_energy_joules,
            37.0,
        )
        self.assertEqual(
            by_policy[FIXED_FACE_TEMPERATURE].nominal_total_plan_energy_joules,
            43.0,
        )
        self.assertEqual(
            tuple(item.incremental_nominal_energy_joules for item in resources),
            (0.0, 12.0, 7.0, 13.0),
        )
    def test_frozen_nominal_proxy_matches_the_declared_resource_table(self):
        resources = costs.prospective_action_resources(self.physical)
        by_policy = {item.policy_name: item for item in resources}
        expected_incremental_energy = {
            STOP_NOW: 0.0,
            FIXED_THERMAL: 38.5717,
            FIXED_VOLTAGE: 28.8143,
            FIXED_FACE_TEMPERATURE: 28.6909,
        }
        for policy_name, expected in expected_incremental_energy.items():
            with self.subTest(policy_name=policy_name):
                self.assertAlmostEqual(
                    by_policy[policy_name].incremental_nominal_energy_joules,
                    expected,
                    places=4,
                )

    def test_reset_time_is_charged_once_per_added_run(self):
        scenario = replace(
            costs.PRIMARY_PROSPECTIVE_COST_SCENARIO,
            name="short_reset",
            assumed_reset_seconds_per_added_run=30.0,
        )
        with _cost_environment():
            resources = costs.prospective_action_resources(
                self.physical,
                scenario,
            )
        by_policy = {item.policy_name: item for item in resources}
        self.assertEqual(by_policy[STOP_NOW].incremental_bench_seconds, 0.0)
        self.assertEqual(by_policy[FIXED_THERMAL].added_schedule_seconds, 240.0)
        self.assertEqual(by_policy[FIXED_THERMAL].assumed_reset_seconds, 90.0)
        self.assertEqual(by_policy[FIXED_THERMAL].incremental_bench_seconds, 330.0)
        for policy_name in (FIXED_VOLTAGE, FIXED_FACE_TEMPERATURE):
            self.assertEqual(
                by_policy[policy_name].incremental_bench_seconds,
                RUN_DURATION_SECONDS + 30.0,
            )

    def test_face_probe_load_is_included_in_verification_energy(self):
        calls = []
        with _cost_environment(calls):
            resources = costs.prospective_action_resources(self.physical)
        by_policy = {item.policy_name: item for item in resources}
        face_verification = by_policy[FIXED_FACE_TEMPERATURE].plan_runs[-1]
        voltage_verification = by_policy[FIXED_VOLTAGE].plan_runs[-1]
        self.assertTrue(face_verification.temporary_face_sensor)
        self.assertEqual(face_verification.nominal_terminal_energy_joules, 25.0)
        self.assertFalse(voltage_verification.temporary_face_sensor)
        self.assertEqual(voltage_verification.nominal_terminal_energy_joules, 20.0)
        face_call = next(
            (regime, instrumentation)
            for regime, instrumentation in calls
            if regime.name == "fixed_verification"
            and instrumentation.temporary_face_sensor
        )
        self.assertIn(COLD_FACE, face_call[0].channels)

    def test_instruments_retain_channel_identity(self):
        with _cost_environment():
            resources = costs.prospective_action_resources(self.physical)
        instruments = {
            item.policy_name: item.added_instruments for item in resources
        }
        self.assertEqual(instruments[STOP_NOW], ())
        self.assertEqual(instruments[FIXED_THERMAL], ())
        self.assertEqual(instruments[FIXED_VOLTAGE], (VOLTAGE,))
        self.assertEqual(instruments[FIXED_FACE_TEMPERATURE], (COLD_FACE,))


class ProspectiveCostFormulaTests(unittest.TestCase):
    def setUp(self):
        self.physical = OperatingDecisionRealismConfig()

    def test_primary_normalization_uses_voltage_energy_and_one_run_time(self):
        with _cost_environment():
            resources = costs.prospective_action_resources(self.physical)
            action_costs, energy_reference, time_reference = (
                costs._build_action_costs(
                    resources,
                    costs.PRIMARY_PROSPECTIVE_COST_SCENARIO,
                )
            )
        self.assertEqual(energy_reference, 7.0)
        self.assertEqual(time_reference, 2.0 * RUN_DURATION_SECONDS)
        by_policy = {item.policy_name: item for item in action_costs}
        self.assertEqual(by_policy[STOP_NOW].declared_cost, 0.0)
        self.assertEqual(by_policy[FIXED_VOLTAGE].normalized_energy, 1.0)
        self.assertEqual(by_policy[FIXED_VOLTAGE].normalized_bench_time, 1.0)
        self.assertEqual(
            by_policy[FIXED_VOLTAGE].normalized_instrumentation,
            1.0,
        )
        self.assertEqual(by_policy[FIXED_VOLTAGE].declared_cost, 1.0)
        self.assertAlmostEqual(
            by_policy[FIXED_THERMAL].declared_cost,
            (12.0 / 7.0 + 3.0) / 3.0,
        )
        self.assertAlmostEqual(
            by_policy[FIXED_FACE_TEMPERATURE].declared_cost,
            (13.0 / 7.0 + 1.0 + 1.0) / 3.0,
        )

    def test_scenario_grid_is_predeclared_unique_and_changes_face_cost(self):
        scenarios = costs.PROSPECTIVE_COST_SENSITIVITY_SCENARIOS
        self.assertEqual(
            costs.PROSPECTIVE_RESET_SENSITIVITY_SECONDS,
            (0.0, RUN_DURATION_SECONDS, 240.0),
        )
        self.assertEqual(len(scenarios), 12)
        self.assertEqual(len({item.name for item in scenarios}), 12)
        self.assertEqual(
            sum(
                item.name == costs.PROSPECTIVE_COST_PRIMARY_SCENARIO_NAME
                for item in scenarios
            ),
            1,
        )
        self.assertEqual(
            {item.face_instrumentation_multiplier for item in scenarios},
            {0.25, 1.0, 4.0},
        )
        self.assertEqual(
            {
                (
                    item.energy_weight,
                    item.bench_time_weight,
                    item.instrumentation_weight,
                )
                for item in scenarios
            },
            {
                (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                (0.70, 0.20, 0.10),
                (0.20, 0.70, 0.10),
                (0.15, 0.15, 0.70),
            },
        )

        cheap_face = next(
            item for item in scenarios if item.name == "balanced_face_quarter"
        )
        expensive_face = next(
            item for item in scenarios if item.name == "balanced_face_quadruple"
        )
        with _cost_environment():
            resources = costs.prospective_action_resources(self.physical)
            cheap, _, _ = costs._build_action_costs(resources, cheap_face)
            expensive, _, _ = costs._build_action_costs(resources, expensive_face)
        cheap_by_policy = {item.policy_name: item for item in cheap}
        expensive_by_policy = {item.policy_name: item for item in expensive}
        self.assertLess(
            cheap_by_policy[FIXED_FACE_TEMPERATURE].declared_cost,
            expensive_by_policy[FIXED_FACE_TEMPERATURE].declared_cost,
        )
        self.assertEqual(
            cheap_by_policy[FIXED_VOLTAGE].declared_cost,
            expensive_by_policy[FIXED_VOLTAGE].declared_cost,
        )

    def test_scenario_validation_rejects_malformed_weights_and_resources(self):
        invalid = (
            {"energy_weight": 0.5},
            {
                "energy_weight": 0.0,
                "bench_time_weight": 0.0,
                "instrumentation_weight": 1.0,
            },
            {"face_instrumentation_multiplier": -1.0},
            {"assumed_reset_seconds_per_added_run": math.nan},
            {"energy_weight": True},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(costs.PRIMARY_PROSPECTIVE_COST_SCENARIO, **changes)

    def test_uncertainty_propagates_positive_negative_and_ineligible_actions(self):
        uncertainties = _uncertainties(face_eligible=False)
        with _cost_environment():
            resources = costs.prospective_action_resources(self.physical)
            action_costs, _, _ = costs._build_action_costs(
                resources,
                costs.PRIMARY_PROSPECTIVE_COST_SCENARIO,
            )
        evaluations = costs._build_action_evaluations(
            uncertainties,
            action_costs,
        )
        by_policy = {item.policy_name: item for item in evaluations}
        self.assertIsNone(by_policy[STOP_NOW].declared_cost)
        self.assertEqual(
            by_policy[FIXED_THERMAL].expected_uncertainty_reduction,
            1.5,
        )
        self.assertEqual(by_policy[FIXED_THERMAL].uncertainty_before, 2.0)
        self.assertEqual(
            by_policy[FIXED_THERMAL].expected_uncertainty_after,
            0.5,
        )
        self.assertEqual(by_policy[FIXED_THERMAL].prospective_draw_count, 4)
        thermal_cost = next(
            item.declared_cost
            for item in action_costs
            if item.policy_name == FIXED_THERMAL
        )
        self.assertEqual(by_policy[FIXED_THERMAL].declared_cost, thermal_cost)
        self.assertGreater(by_policy[FIXED_THERMAL].utility_per_cost, 0.0)
        self.assertEqual(
            by_policy[FIXED_VOLTAGE].expected_uncertainty_reduction,
            -0.5,
        )
        self.assertLess(by_policy[FIXED_VOLTAGE].utility_per_cost, 0.0)
        face = by_policy[FIXED_FACE_TEMPERATURE]
        self.assertFalse(face.eligible)
        self.assertEqual(
            face.failure_reason,
            "insufficient_stable_prospective_draws",
        )
        self.assertIsNone(face.uncertainty_before)
        self.assertIsNone(face.expected_uncertainty_after)
        self.assertIsNone(face.declared_cost)
        self.assertEqual(face.prospective_draw_count, 0)


class ProspectiveCostIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.physical = OperatingDecisionRealismConfig()
        self.uncertainty = _uncertainty_result(self.physical)

    def _scorecard(
        self,
        scenario=costs.PRIMARY_PROSPECTIVE_COST_SCENARIO,
        selector_rule=ProspectiveSelectorRule(),
    ):
        with _cost_environment():
            return costs.cost_prospective_uncertainty(
                self.uncertainty,
                scenario,
                selector_rule,
            )

    def test_cost_rescoring_does_not_refit_acquisition_evidence(self):
        with (
            patch.object(
                costs,
                "nominal_selection_cost_proxy",
                side_effect=_fake_nominal_energy(),
            ),
            patch.object(
                costs,
                "validate_prospective_uncertainty_result_integrity",
                return_value=None,
            ),
            patch.object(
                prospective_uncertainty,
                "_validate_acquisition_evidence",
                side_effect=AssertionError("cost rescoring must not refit"),
            ),
        ):
            scorecard = costs.cost_prospective_uncertainty(self.uncertainty)
        self.assertEqual(scorecard.uncertainty_result, self.uncertainty)

    def test_costed_scorecard_is_bound_to_the_selector_snapshot(self):
        scorecard = self._scorecard()
        selection = costs.select_costed_prospective_action(scorecard)
        self.assertEqual(selection.selected_policy, FIXED_THERMAL)
        self.assertTrue(selection.requires_additional_acquisition)

        sentinel = object()
        rule = ProspectiveSelectorRule(minimum_expected_reduction=0.1)
        scorecard = self._scorecard(selector_rule=rule)
        with patch.object(
            costs,
            "select_prospective_action",
            return_value=sentinel,
        ) as selector:
            selected = costs.select_costed_prospective_action(scorecard, rule)
        self.assertIs(selected, sentinel)
        selector.assert_called_once_with(
            self.uncertainty.acquisition_evidence.snapshot,
            scorecard.action_evaluations,
            rule,
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            costs.select_costed_prospective_action(
                scorecard,
                ProspectiveSelectorRule(minimum_expected_reduction=0.2),
            )

    def test_scorecard_provenance_binds_the_actual_offset_rule(self):
        rule = ProspectiveSelectorRule(
            development_offsets=ProspectiveDevelopmentOffsets(
                version="development_zero_probe_v2"
            ),
            single_candidate_stopping_clearance=0.04,
        )
        scorecard = self._scorecard(selector_rule=rule)
        payload = costs.prospective_costed_scorecard_payload(scorecard)
        self.assertEqual(scorecard.selector_rule, rule)
        self.assertEqual(
            payload["selector_protocol_digest"],
            prospective_selector_protocol_digest(rule),
        )
        self.assertEqual(
            payload["selector_rule"]["development_offsets"]["version"],
            "development_zero_probe_v2",
        )
        with _cost_environment():
            self.assertNotEqual(
                scorecard.protocol_digest,
                costs.prospective_cost_protocol_digest(
                    self.physical,
                    self.uncertainty.config,
                ),
            )

    def test_protocol_digest_binds_scenario_upstream_and_physics(self):
        with _cost_environment():
            original = costs.prospective_cost_protocol_digest(
                self.physical,
                self.uncertainty.config,
            )
            self.assertEqual(
                original,
                costs.prospective_cost_protocol_digest(
                    self.physical,
                    self.uncertainty.config,
                ),
            )
            self.assertNotEqual(
                original,
                costs.prospective_cost_protocol_digest(
                    self.physical,
                    replace(self.uncertainty.config, draw_count=5),
                ),
            )
            self.assertNotEqual(
                original,
                costs.prospective_cost_protocol_digest(
                    self.physical,
                    self.uncertainty.config,
                    replace(
                        costs.PRIMARY_PROSPECTIVE_COST_SCENARIO,
                        name="long_reset",
                        assumed_reset_seconds_per_added_run=120.0,
                    ),
                ),
            )
            self.assertNotEqual(
                original,
                costs.prospective_cost_protocol_digest(
                    replace(
                        self.physical,
                        face_sensor_prior_log_standard_deviation=0.36,
                    ),
                    self.uncertainty.config,
                ),
            )
            self.assertNotEqual(
                original,
                costs.prospective_cost_protocol_digest(
                    self.physical,
                    self.uncertainty.config,
                    replace(
                        costs.PRIMARY_PROSPECTIVE_COST_SCENARIO,
                        energy_weight=0.70,
                        bench_time_weight=0.20,
                        instrumentation_weight=0.10,
                    ),
                ),
            )
            self.assertNotEqual(
                original,
                costs.prospective_cost_protocol_digest(
                    self.physical,
                    self.uncertainty.config,
                    replace(
                        costs.PRIMARY_PROSPECTIVE_COST_SCENARIO,
                        face_instrumentation_multiplier=4.0,
                    ),
                ),
            )

    def test_result_digest_binds_uncertainty_result_and_scenario(self):
        first = self._scorecard()
        second_uncertainty = _uncertainty_result(
            self.physical,
            result_digest="c" * 64,
        )
        with _cost_environment():
            second = costs.cost_prospective_uncertainty(second_uncertainty)
            reset = costs.cost_prospective_uncertainty(
                self.uncertainty,
                replace(
                    costs.PRIMARY_PROSPECTIVE_COST_SCENARIO,
                    name="reset_120",
                    assumed_reset_seconds_per_added_run=120.0,
                ),
            )
        self.assertNotEqual(first.result_digest, second.result_digest)
        self.assertNotEqual(first.result_digest, reset.result_digest)

    def test_payload_round_trip_is_strict_finite_and_rederived(self):
        scorecard = self._scorecard()
        payload = costs.prospective_costed_scorecard_payload(scorecard)
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
        with _cost_environment():
            loaded = costs.prospective_costed_scorecard_from_payload(
                decoded,
                self.uncertainty,
            )
        self.assertEqual(
            costs.prospective_costed_scorecard_payload(loaded),
            payload,
        )

    def test_payload_rejects_top_level_identity_and_reference_tampering(self):
        payload = costs.prospective_costed_scorecard_payload(self._scorecard())
        mutations = []
        extra = deepcopy(payload)
        extra["unexpected"] = 1
        mutations.append(extra)
        missing = deepcopy(payload)
        del missing["formula"]
        mutations.append(missing)
        schema = deepcopy(payload)
        schema["schema_version"] = True
        mutations.append(schema)
        formula = deepcopy(payload)
        formula["formula"] = "changed"
        mutations.append(formula)
        upstream = deepcopy(payload)
        upstream["uncertainty_result_digest"] = "0" * 64
        mutations.append(upstream)
        protocol = deepcopy(payload)
        protocol["protocol_digest"] = "0" * 64
        mutations.append(protocol)
        reference = deepcopy(payload)
        reference["energy_reference_joules"] += 1.0
        mutations.append(reference)
        for tampered in mutations:
            with self.subTest(keys=tuple(tampered)):
                with _cost_environment(), self.assertRaises(ValueError):
                    costs.prospective_costed_scorecard_from_payload(
                        tampered,
                        self.uncertainty,
                    )

    def test_payload_rejects_nested_scores_order_and_digest_tampering(self):
        payload = costs.prospective_costed_scorecard_payload(self._scorecard())
        mutations = []
        nested_key = deepcopy(payload)
        nested_key["scenario"]["unexpected"] = 1
        mutations.append(nested_key)
        cost_value = deepcopy(payload)
        cost_value["action_costs"][1]["declared_cost"] += 0.1
        mutations.append(cost_value)
        derivative = deepcopy(payload)
        derivative["action_evaluations"][1]["utility_per_cost"] += 0.1
        mutations.append(derivative)
        order = deepcopy(payload)
        order["action_resources"][1:3] = reversed(
            order["action_resources"][1:3]
        )
        mutations.append(order)
        self_consistent_resource = deepcopy(payload)
        thermal = self_consistent_resource["action_resources"][1]
        thermal["plan_runs"][1]["nominal_terminal_energy_joules"] += 1.0
        thermal["nominal_total_plan_energy_joules"] += 1.0
        thermal["incremental_nominal_energy_joules"] += 1.0
        mutations.append(self_consistent_resource)
        digest = deepcopy(payload)
        digest["result_digest"] = "0" * 64
        mutations.append(digest)
        nonfinite = deepcopy(payload)
        nonfinite["scenario"]["energy_weight"] = math.nan
        mutations.append(nonfinite)
        for tampered in mutations:
            with self.subTest(result_digest=tampered["result_digest"]):
                with _cost_environment(), self.assertRaises(ValueError):
                    costs.prospective_costed_scorecard_from_payload(
                        tampered,
                        self.uncertainty,
                    )

    def test_interfaces_exclude_truth_reveal_and_future_observations(self):
        forbidden = {
            "truth_condition",
            "truth_family",
            "trial_index",
            "device_token",
            "verification_run",
            "verification_observation",
            "selected_policy_outcome",
            "true_margin",
            "final_response",
        }
        for function in (
            costs.prospective_action_resources,
            costs.prospective_cost_protocol_digest,
            costs.cost_prospective_uncertainty,
            costs.select_costed_prospective_action,
        ):
            with self.subTest(function=function.__name__):
                self.assertFalse(
                    forbidden.intersection(inspect.signature(function).parameters)
                )

    def test_completed_protocols_remain_isolated(self):
        before = (
            calibration.STAGE4_PROCEDURES,
            replication_guard.CORRECTED_FINAL_PROCEDURES,
            prospective_selector_protocol_digest(),
        )
        with _cost_environment():
            costs.prospective_cost_protocol_digest(
                self.physical,
                self.uncertainty.config,
            )
            self._scorecard()
        after = (
            calibration.STAGE4_PROCEDURES,
            replication_guard.CORRECTED_FINAL_PROCEDURES,
            prospective_selector_protocol_digest(),
        )
        self.assertEqual(after, before)
        self.assertNotIn(
            costs.PROSPECTIVE_COST_PROTOCOL_VERSION,
            calibration.STAGE4_PROCEDURES,
        )
        self.assertNotIn(
            costs.PROSPECTIVE_COST_PROTOCOL_VERSION,
            replication_guard.CORRECTED_FINAL_PROCEDURES,
        )


if __name__ == "__main__":
    unittest.main()
