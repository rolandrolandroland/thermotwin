from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thermotwin.studies.operating_decision import POLICY_NAMES, STOP_NOW
import thermotwin.studies.operating_decision_prospective_phase_d as phase_d
import thermotwin.studies.operating_decision_prospective_phase_d6_internal_check as d6


class PhaseD6ProtocolTests(unittest.TestCase):
    def test_frozen_rule_and_authorization_are_exact(self):
        rule = d6.phase_d6_frozen_selector_rule()
        self.assertEqual(rule.minimum_utility_per_cost, 0.025)
        self.assertEqual(rule.development_offsets.stop_now, 0.0)
        self.assertEqual(rule.development_offsets.fixed_thermal, 0.0)
        self.assertEqual(rule.development_offsets.fixed_voltage, 0.074)
        self.assertEqual(rule.development_offsets.fixed_face_temperature, 0.098)
        payload = d6.phase_d6_protocol_payload(
            source_revision="a" * 40,
            source_manifest_digest="b" * 64,
        )
        self.assertEqual(payload["partition"], "p1_development_internal_check")
        self.assertEqual(payload["case_count"], 30)
        self.assertFalse(payload["independent_calibration_opened"])
        self.assertFalse(payload["reserved_evaluation_opened"])
        self.assertEqual(
            set(payload["redesign_triggers"]),
            {
                "material_defect",
                "repeated_false_approval",
                "measurement_action_ineligibility",
            },
        )

    def test_paths_are_bounded_to_ten_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                d6.phase_d6_internal_check_block_path(directory, 9).name,
                "block-009.json",
            )
            with self.assertRaises(ValueError):
                d6.phase_d6_internal_check_block_path(directory, 10)

    def test_preflight_opens_only_internal_check(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            d6,
            "validate_executing_phase_d_runtime",
            return_value=phase_d.phase_d_runtime_identity(),
        ):
            payload = d6.phase_d6_internal_check_preflight_payload(
                repository_root=Path(__file__).resolve().parents[1],
                source_revision="a" * 40,
                output_directory=directory,
            )
        self.assertEqual(payload["partition"], "p1_development_internal_check")
        self.assertEqual(payload["block_count"], 10)
        self.assertEqual(payload["case_count"], 30)
        self.assertFalse(payload["truth_analysis_authorized"])


class PhaseD6TriggerTests(unittest.TestCase):
    @staticmethod
    def _blocks():
        return [
            {
                "block": block,
                "cases": [
                    {
                        "truth_condition": family,
                        "frozen_selection_before_reveal": {
                            "selection_succeeded": True,
                            "selected_policy": STOP_NOW,
                        },
                    }
                    for family in ("family_a", "family_b", "family_c")
                ],
            }
            for block in range(10)
        ]

    @staticmethod
    def _record(index):
        block = index // 3
        return {
            "block": block,
            "truth_family": ("family_a", "family_b", "family_c")[index % 3],
            "selected_policy": STOP_NOW,
            "candidate_reliability_stratum": "two_admissible_candidates",
            "ineligible_actions": [],
            "false_approval": False,
            "pipeline_failure": False,
        }

    def _analyze(self, records):
        overall = {
            "case_count": 30,
            "definitive_decisions": 21,
            "decision_coverage": 0.7,
            "false_approvals": sum(r["false_approval"] for r in records),
            "false_rejections": 0,
            "pipeline_failures": sum(r["pipeline_failure"] for r in records),
            "verification_failures": 0,
            "selection_failures": 0,
            "mean_realized_diagnostic_energy_joules": 70.0,
            "mean_realized_extra_sensor_count": 0.1,
        }
        with patch.object(d6, "_prefix_by_count", return_value={
            "costed_scorecard": {"action_resources": [], "action_costs": []}
        }), patch.object(
            d6, "_selected_outcome_record", side_effect=records
        ), patch.object(
            d6, "_objective_for_records", return_value={"case_results": []}
        ), patch.object(
            d6, "_case_loss_records", return_value=records
        ), patch.object(
            d6, "_group_summaries", return_value={"overall": overall, "by": {}}
        ):
            return d6.analyze_phase_d6_internal_check(self._blocks())

    def test_poor_benefit_and_low_diversity_do_not_trigger_redesign(self):
        records = [self._record(index) for index in range(30)]
        result = self._analyze(records)
        self.assertFalse(result["redesign_required"])
        self.assertTrue(result["phase_e_authorized"])

    def test_two_false_approvals_in_same_stratum_trigger(self):
        records = [self._record(index) for index in range(30)]
        records[0]["false_approval"] = True
        records[3]["false_approval"] = True
        result = self._analyze(records)
        trigger = result["redesign_triggers"][
            "repeated_false_approval_same_stratum"
        ]
        self.assertTrue(trigger["triggered"])
        self.assertTrue(result["redesign_required"])

    def test_ineligibility_trigger_is_strictly_more_than_twenty_percent(self):
        six = [self._record(index) for index in range(30)]
        for record in six[:6]:
            record["ineligible_actions"] = ["fixed_face_temperature"]
        self.assertFalse(self._analyze(six)["redesign_required"])
        seven = deepcopy(six)
        seven[6]["ineligible_actions"] = ["fixed_face_temperature"]
        self.assertTrue(self._analyze(seven)["redesign_required"])


if __name__ == "__main__":
    unittest.main()
