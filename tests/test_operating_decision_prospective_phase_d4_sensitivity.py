import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from thermotwin.studies.operating_decision import POLICY_NAMES, STOP_NOW
from thermotwin.studies.operating_decision_prospective import (
    ProspectiveDevelopmentOffsets,
    ProspectiveSelectorRule,
)
from thermotwin.studies import operating_decision_prospective_phase_d4_sensitivity as d4
from thermotwin.reports import operating_decision_prospective_phase_d4 as d4_cli


def _prefix(draw_count, selected="fixed_face_temperature", *, ineligible=None):
    utilities = {
        "fixed_thermal": 0.08,
        "fixed_voltage": 0.096,
        "fixed_face_temperature": 0.10,
    }
    eligibility = {
        policy: {
            "eligible": policy != ineligible,
            "failure_reason": "fixture" if policy == ineligible else None,
        }
        for policy in POLICY_NAMES
        if policy != STOP_NOW
    }
    return {
        "draw_count": draw_count,
        "max_unstable_draws_per_source_action": 1,
        "choice_token": selected,
        "eligibility": eligibility,
        "draw_diagnostics": {"available": True, "failed_draw_count": 0},
        "selection": {
            "selected_policy": selected,
            "action_evaluations": [
                {
                    "policy_name": policy,
                    "eligible": eligibility[policy]["eligible"],
                    "utility_per_cost": utilities[policy],
                }
                for policy in utilities
            ],
        },
    }


def _blocks(*, agreements=11, ineligible=None):
    blocks = []
    index = 0
    for block in d4.PHASE_D4_BLOCK_INDICES:
        cases = []
        for family in d4.STAGE3_TRUTH_CONDITIONS:
            same = index < agreements
            cases.append(
                {
                    "block": block,
                    "truth_condition": family,
                    "n32_complete_uncertainty_result": {"fixture": True},
                    "pipeline_failures": [],
                    "primary_prefixes": [
                        _prefix(
                            16,
                            "fixed_face_temperature" if same else "fixed_voltage",
                            ineligible=ineligible,
                        ),
                        _prefix(32, "fixed_face_temperature", ineligible=ineligible),
                    ],
                }
            )
            index += 1
        blocks.append({"block": block, "cases": cases})
    return blocks


class PhaseD4SensitivityTests(TestCase):
    def test_monitor_preflight_preserves_denied_attempt_and_pass(self):
        path = Path(
            "thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_D4_MONITOR_PREFLIGHT.json"
        )
        loaded = d4.validate_phase_d4_monitor_preflight(json.loads(path.read_bytes()))
        self.assertEqual(
            loaded["attempts"][0]["result"], "failed_before_scientific_work"
        )
        self.assertEqual(loaded["attempts"][1]["result"], "passed")
        self.assertTrue(
            loaded["authorization"][
                "synchronous_recheck_required_immediately_before_computation"
            ]
        )

    def test_gate_passes_at_exact_eleven_of_twelve_with_four_percent_regret(self):
        result = d4.evaluate_phase_d4_gate(_blocks(agreements=11))
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["summary"]["agreement_count"], 11)
        self.assertAlmostEqual(
            result["summary"][
                "maximum_changed_choice_normalized_utility_regret"
            ],
            0.04,
        )
        self.assertEqual(result["accepted_draw_count"], 16)
        self.assertTrue(result["phase_d5_authorized"])

    def test_gate_fails_below_agreement_or_with_any_ineligible_action(self):
        self.assertFalse(
            d4.evaluate_phase_d4_gate(_blocks(agreements=10))["gate_passed"]
        )
        result = d4.evaluate_phase_d4_gate(
            _blocks(agreements=12, ineligible="fixed_voltage")
        )
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["ineligible_action_count"], 24)
        self.assertEqual(
            result["next_required_action"],
            "commit_budget_for_remaining_n32_and_subset_n64_before_execution",
        )

    def test_generated_n32_block_must_reproduce_preserved_n16_prefix(self):
        parent_cases = []
        generated_cases = []
        for family in d4.STAGE3_TRUTH_CONDITIONS:
            parent_cases.append(
                {
                    "truth_condition": family,
                    "device_token": f"device-{family}",
                    "n16_complete_uncertainty_result": {"family": family},
                }
            )
            generated_cases.append(
                {
                    "block": 0,
                    "truth_condition": family,
                    "device_token": f"device-{family}",
                    "admissible_source_models": ["four_state"],
                    "admissible_source_model_count": 1,
                    "pipeline_failures": [],
                    "n32_complete_uncertainty_result": {"complete": family},
                    "primary_prefixes": [_prefix(16), _prefix(32)],
                    "timing": {},
                }
            )
        d1 = {"block_results": [{"block": 0, "cases": parent_cases}]}
        block = {
            "block": 0,
            "cases": generated_cases,
            "corrected_random_stream_manifest": [{"fixture": True}],
            "corrected_random_stream_audit": {"clean": True},
            "timing": {},
        }

        def derived(complete, **_):
            return {"family": complete["complete"]}

        patches = (
            mock.patch.object(d4, "_validate_corrected_stream_records"),
            mock.patch.object(
                d4,
                "_validate_complete_uncertainty_payload",
                side_effect=lambda payload, **_: payload,
            ),
            mock.patch.object(
                d4, "_derive_uncertainty_prefix_payload", side_effect=derived
            ),
            mock.patch.object(d4, "_validate_authenticated_prefix"),
        )
        with patches[0], patches[1], patches[2], patches[3] as validate_prefix:
            d4._validate_generated_block(
                block,
                d1=d1,
                selector_rule=ProspectiveSelectorRule(
                    development_offsets=ProspectiveDevelopmentOffsets(
                        version="phase_d2_test_offsets_v1",
                        fixed_voltage=0.074,
                        fixed_face_temperature=0.098,
                    )
                ),
            )
        self.assertEqual(validate_prefix.call_count, 6)
        self.assertTrue(
            all(
                call.kwargs["require_zero_development_offsets"] is False
                for call in validate_prefix.call_args_list
            )
        )

        with mock.patch.object(d4, "_validate_corrected_stream_records"), mock.patch.object(
            d4,
            "_validate_complete_uncertainty_payload",
            side_effect=lambda payload, **_: payload,
        ), mock.patch.object(
            d4,
            "_derive_uncertainty_prefix_payload",
            return_value={"family": "wrong"},
        ):
            with self.assertRaisesRegex(ValueError, "differs from preserved Phase D1"):
                d4._validate_generated_block(
                    block, d1=d1, selector_rule=ProspectiveSelectorRule()
                )

    def test_protocol_freezes_subset_rule_and_failure_path(self):
        rule = ProspectiveSelectorRule(minimum_utility_per_cost=0.025)
        payload = d4.phase_d4_protocol_payload(
            source_revision="1" * 40,
            source_manifest_digest="2" * 64,
            input_hashes={"input": "3" * 64},
            selector_rule=rule,
        )
        self.assertEqual(payload["block_indices"], [0, 5, 10, 15])
        self.assertEqual(payload["minimum_action_agreement_count"], 11)
        self.assertEqual(payload["reference_draw_count"], 32)
        self.assertIn("remaining_n32", payload["failure_path"])

    def test_performance_fields_do_not_change_scientific_block_digest(self):
        first = {"block": 0, "value": 1, "timing": {"wall_seconds": 1.0}}
        second = {"block": 0, "value": 1, "timing": {"wall_seconds": 9.0}}
        self.assertEqual(
            d4._block_scientific_digest(first), d4._block_scientific_digest(second)
        )

    def test_block_path_rejects_outcome_directed_subset_change(self):
        with self.assertRaisesRegex(ValueError, "predeclared subset"):
            d4.phase_d4_block_path("/tmp/d4", 1)

    def test_save_seals_canonical_archive_and_detached_hashes(self):
        with TemporaryDirectory() as root:
            saved = d4.save_phase_d4_artifacts(
                {
                    "analysis_source_revision": "1" * 40,
                    "scientific_result_digest": "2" * 64,
                    "acceptance": {
                        "summary": {
                            "agreement_count": 12,
                            "maximum_changed_choice_normalized_utility_regret": 0.0,
                        },
                        "pipeline_failure_count": 0,
                        "selection_failure_count": 0,
                        "unavailable_draw_diagnostic_count": 0,
                        "ineligible_action_count": 0,
                        "whole_draw_failure_count": 0,
                        "gate_passed": True,
                        "next_required_action": "execute_phase_d5_maps",
                    },
                },
                root,
            )
            loaded = json.loads(saved.json_path.read_bytes())
            self.assertEqual(loaded["archive_size_bytes"], saved.archive_size_bytes)
            self.assertEqual(saved.archive_size_bytes, len(saved.json_path.read_bytes()))
            manifest = saved.hash_path.read_text()
            self.assertIn(saved.json_sha256, manifest)
            self.assertIn(saved.report_sha256, manifest)

    def test_cli_samples_process_tree_synchronously_before_science(self):
        with TemporaryDirectory() as root, mock.patch.object(
            d4_cli, "_clean_revision", return_value="1" * 40
        ), mock.patch.object(
            d4_cli,
            "_process_tree_rss_bytes",
            side_effect=PermissionError("process table denied"),
        ), mock.patch.object(d4_cli, "run_phase_d4_sensitivity") as run:
            with self.assertRaisesRegex(PermissionError, "process table denied"):
                d4_cli.main(
                    [
                        "--execute-sensitivity",
                        "--phase-d1-archive", "d1.json",
                        "--phase-d2-result", "d2.json",
                        "--phase-d3-result", "d3.json",
                        "--rule-freeze", "rule.json",
                        "--monitor-preflight", "monitor.json",
                        "--output-directory", root,
                    ]
                )
            run.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
