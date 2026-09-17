from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from thermotwin.studies.operating_decision import (
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    POLICY_NAMES,
    default_fixed_policies,
)
from thermotwin.studies.operating_decision_prospective import ProspectiveSelectorRule
from thermotwin.studies.operating_decision_prospective_costs import (
    PRIMARY_PROSPECTIVE_COST_SCENARIO,
)
from thermotwin.studies.operating_decision_random_streams import RandomStreamRegistry
from thermotwin.studies.operating_decision_realism import (
    OperatingDecisionRealismConfig,
)
from thermotwin.studies.operating_decision_replication import CorrectedPartition
import thermotwin.studies.operating_decision_prospective_pilot as pilot
import thermotwin.reports.operating_decision_prospective_pilot as pilot_cli
from thermotwin.reports.operating_decision_prospective_pilot import (
    _committed_source_revision,
    _require_clean_head,
    _require_project_root,
)


def _prefix(draw_count, selected, *, voltage_utility=0.96, eligibility=None):
    if eligibility is None:
        eligibility = {
            name: {"eligible": True, "failure_reason": None}
            for name in POLICY_NAMES
        }
    evaluations = [
        {
            "policy_name": name,
            "eligible": eligibility[name]["eligible"],
            "utility_per_cost": (
                1.0
                if name == FIXED_THERMAL
                else voltage_utility
                if name == FIXED_VOLTAGE
                else 0.2
            ),
        }
        for name in POLICY_NAMES
    ]
    return {
        "draw_count": draw_count,
        "max_unstable_draws_per_source_action": 0,
        "choice_token": f"action:{selected}",
        "eligibility": eligibility,
        "selection": {
            "selected_policy": selected,
            "action_evaluations": evaluations,
        },
    }


def _blocks(*, changed_regret=0.04, pipeline_failure=False):
    voltage_utility = 1.0 - changed_regret
    blocks = []
    for block in range(4):
        cases = []
        for family_index, family in enumerate(pilot.STAGE3_TRUTH_CONDITIONS):
            changed = block == 3 and family_index == 2
            reference = _prefix(16, FIXED_THERMAL, voltage_utility=voltage_utility)
            n4 = _prefix(
                4,
                FIXED_VOLTAGE if changed else FIXED_THERMAL,
                voltage_utility=voltage_utility,
            )
            n8 = _prefix(8, FIXED_THERMAL, voltage_utility=voltage_utility)
            sensitivity = deepcopy(reference)
            sensitivity["max_unstable_draws_per_source_action"] = 1
            cases.append(
                {
                    "block": block,
                    "truth_condition": family,
                    "strict_prefixes": [n4, n8, reference],
                    "n16_max_unstable_1_sensitivity": sensitivity,
                    "pipeline_failures": (
                        [{"stage": "test"}]
                        if pipeline_failure and block == 0 and family_index == 0
                        else []
                    ),
                    "timing": {"case_wall_seconds": 999.0},
                }
            )
        blocks.append(
            {
                "block": block,
                "cases": cases,
                "timing": {
                    "block_wall_seconds": float(block + 1),
                    "block_cpu_seconds": float(block + 2),
                    "peak_rss_bytes": 1000 + block,
                },
            }
        )
    return blocks


class ProspectivePilotProtocolTests(unittest.TestCase):
    def test_partition_names_and_sizes_are_exact_and_disjoint(self):
        self.assertEqual(
            pilot.PROSPECTIVE_CAMPAIGN,
            "operating_decision_prospective_v1_2026_09",
        )
        self.assertEqual(
            tuple((item.name, item.block_count) for item in pilot.PROSPECTIVE_PARTITION_PLAN),
            (
                ("p1_disposable_draw_count_pilot", 4),
                ("p1_development_tuning", 20),
                ("p1_development_internal_check", 10),
                ("p1_independent_calibration", 100),
                ("p1_reserved_evaluation", 100),
            ),
        )
        self.assertEqual(
            len({item.name for item in pilot.PROSPECTIVE_PARTITION_PLAN}),
            5,
        )
        self.assertEqual(pilot.PILOT_DRAW_COUNTS, (4, 8, 16))
        self.assertEqual(pilot.PILOT_STRICT_MAX_UNSTABLE, 0)
        self.assertEqual(pilot.PILOT_SENSITIVITY_MAX_UNSTABLE, 1)

    def test_acceptance_uses_all_cases_and_declared_agreement_and_regret(self):
        result = pilot.evaluate_pilot_acceptance(_blocks(changed_regret=0.04))
        n4, n8, n16 = result["strict_summaries"]
        self.assertEqual(n4["agreement_count"], 11)
        self.assertAlmostEqual(n4["agreement_rate"], 11 / 12)
        self.assertAlmostEqual(
            n4["maximum_changed_choice_normalized_utility_regret"],
            0.04,
        )
        self.assertTrue(n4["meets_acceptance_rule"])
        self.assertEqual(n8["agreement_count"], 12)
        self.assertEqual(n16["agreement_count"], 12)
        self.assertEqual(result["recommended_draw_count"], 4)
        self.assertTrue(result["accepted"])

    def test_excess_regret_moves_recommendation_to_next_draw_count(self):
        result = pilot.evaluate_pilot_acceptance(_blocks(changed_regret=0.06))
        self.assertFalse(result["strict_summaries"][0]["meets_regret_rule"])
        self.assertEqual(result["recommended_draw_count"], 8)

    def test_pipeline_failure_blocks_acceptance_even_when_choices_agree(self):
        result = pilot.evaluate_pilot_acceptance(
            _blocks(pipeline_failure=True)
        )
        self.assertEqual(result["pipeline_failure_count"], 1)
        self.assertIsNone(result["recommended_draw_count"])
        self.assertFalse(result["accepted"])

    def test_selected_action_eligibility_differences_are_reported_separately(self):
        candidate = _prefix(4, FIXED_VOLTAGE)
        reference = _prefix(16, FIXED_THERMAL)
        candidate["eligibility"][FIXED_THERMAL] = {
            "eligible": False,
            "failure_reason": "insufficient_stable_prospective_draws",
        }
        comparison = pilot.compare_pilot_choices(candidate, reference)
        self.assertFalse(comparison["agreement"])
        self.assertTrue(comparison["eligibility_changed"])
        self.assertTrue(
            comparison[
                "choice_change_with_selected_action_eligibility_difference"
            ]
        )

    def test_changed_choice_without_reference_utility_fails_regret_gate(self):
        candidate = _prefix(4, FIXED_VOLTAGE)
        reference = _prefix(16, FIXED_THERMAL)
        reference["eligibility"][FIXED_VOLTAGE] = {
            "eligible": False,
            "failure_reason": "insufficient_stable_prospective_draws",
        }
        voltage = next(
            item
            for item in reference["selection"]["action_evaluations"]
            if item["policy_name"] == FIXED_VOLTAGE
        )
        voltage["eligible"] = False
        voltage["utility_per_cost"] = None
        comparison = pilot.compare_pilot_choices(candidate, reference)
        self.assertFalse(comparison["regret_evaluable"])
        self.assertIsNone(comparison["normalized_utility_regret"])
        summary = pilot._summarize_comparisons(
            (comparison,),
            pilot.PilotAcceptanceRule(),
        )
        self.assertFalse(summary["meets_regret_rule"])
        self.assertEqual(summary["changed_choice_count"], 1)
        self.assertEqual(summary["unevaluable_changed_choice_count"], 1)

    def test_reference_selection_failures_block_overall_pilot_gate(self):
        blocks = _blocks()
        for block in blocks:
            for case in block["cases"]:
                for prefix in case["strict_prefixes"]:
                    prefix["choice_token"] = (
                        "selection_failure:no_scored_acquisition_action"
                    )
                    prefix["selection"] = None
                    prefix["eligibility"] = {}
        result = pilot.evaluate_pilot_acceptance(blocks)
        self.assertEqual(result["stability_recommended_draw_count"], 4)
        self.assertEqual(result["n16_selection_failure_count"], 12)
        self.assertFalse(result["feasibility_gate_passed"])
        self.assertIsNone(result["recommended_draw_count"])
        self.assertFalse(result["accepted"])

    def test_scientific_payload_is_independent_of_worker_order_and_timings(self):
        blocks = _blocks()
        acceptance = pilot.evaluate_pilot_acceptance(blocks)
        first = pilot.prospective_pilot_scientific_payload(
            source_revision="a" * 40,
            protocol_digest="b" * 64,
            block_results=blocks,
            acceptance=acceptance,
        )
        changed = deepcopy(list(reversed(blocks)))
        for block in changed:
            block["timing"]["block_wall_seconds"] += 10_000.0
            for case in block["cases"]:
                case["timing"]["case_wall_seconds"] += 10_000.0
        second = pilot.prospective_pilot_scientific_payload(
            source_revision="a" * 40,
            protocol_digest="b" * 64,
            block_results=changed,
            acceptance=acceptance,
        )
        self.assertEqual(first, second)
        self.assertNotIn("timing", json.dumps(first))

    def test_family_runner_generates_once_prefixes_authentically_and_saves_before_reveal(self):
        events = []
        prefix_calls = []
        policies = default_fixed_policies()

        def build_case(_family, _block, policy, *_args):
            return SimpleNamespace(
                case_id=SimpleNamespace(device_token="device-token"),
                acquisition_runs=(f"initial-{policy.name}",),
                final_regime="final-regime",
                policy=policy,
            )

        evidence = SimpleNamespace(
            evidence_digest="e" * 64,
            snapshot=SimpleNamespace(
                admissible_candidate_models=("four_state",)
            ),
        )
        n16 = SimpleNamespace(name="n16")

        def prefix_result(_result, *, draw_count, max_unstable_draws_per_source_action):
            prefix_calls.append((draw_count, max_unstable_draws_per_source_action))
            return SimpleNamespace(
                config=SimpleNamespace(
                    draw_count=draw_count,
                    max_unstable_draws_per_source_action=(
                        max_unstable_draws_per_source_action
                    ),
                )
            )

        def prefix_record(result, **_kwargs):
            selected = FIXED_THERMAL
            return {
                "draw_count": result.config.draw_count,
                "max_unstable_draws_per_source_action": (
                    result.config.max_unstable_draws_per_source_action
                ),
                "choice_token": f"action:{selected}",
                "eligibility": {},
                "selection": {"selected_policy": selected},
                "uncertainty_summary": {"authenticated": True},
                "costed_scorecard": {"authenticated": True},
            }

        def save_decision(case, **_kwargs):
            events.append(("save", case.policy.name))
            return {"saved": case.policy.name}

        def reveal(saved, **_kwargs):
            events.append(("reveal", saved["saved"]))
            return SimpleNamespace(
                revealed={"true_margin": 1.0},
                scored={"decision": "approve"},
                nominal_selection_energy=1.0,
                realized_energy={"total": 1.0},
            )

        complete_n16 = {
            "acquisition_evidence": {
                "snapshot": {"admissible_candidate_models": ["four_state"]}
            },
            "draw_outcomes": [{"draw_index": index} for index in range(16)],
        }
        partition = CorrectedPartition(
            pilot.PROSPECTIVE_PILOT_PARTITION,
            pilot.PILOT_BLOCK_COUNT,
            pilot.PROSPECTIVE_CAMPAIGN,
        )
        with (
            patch.object(pilot, "build_corrected_blinded_case", side_effect=build_case),
            patch.object(
                pilot,
                "prepare_prospective_acquisition_evidence",
                return_value=evidence,
            ),
            patch.object(
                pilot,
                "estimate_prospective_action_uncertainty",
                return_value=n16,
            ) as estimate,
            patch.object(
                pilot,
                "prospective_uncertainty_result_payload",
                return_value=complete_n16,
            ),
            patch.object(
                pilot,
                "prefix_prospective_uncertainty_result",
                side_effect=prefix_result,
            ),
            patch.object(pilot, "_prefix_record", side_effect=prefix_record),
            patch.object(
                pilot,
                "decide_corrected_blinded_case",
                side_effect=save_decision,
            ),
            patch.object(
                pilot,
                "score_corrected_saved_decision",
                side_effect=reveal,
            ),
            patch.object(pilot, "_case_payload", return_value={"case": True}),
            patch.object(pilot, "_saved_payload", side_effect=lambda value: value),
        ):
            result = pilot._run_pilot_family(
                truth_condition=pilot.STAGE3_TRUTH_CONDITIONS[0],
                block=0,
                truth={"hidden": "truth"},
                partition=partition,
                physical_config=OperatingDecisionRealismConfig(),
                registry=RandomStreamRegistry(),
                selector_rule=ProspectiveSelectorRule(),
                cost_scenario=PRIMARY_PROSPECTIVE_COST_SCENARIO,
            )

        estimate.assert_called_once()
        config = estimate.call_args.args[3]
        self.assertEqual(config.draw_count, 16)
        self.assertEqual(config.max_unstable_draws_per_source_action, 0)
        self.assertEqual(prefix_calls, [(4, 0), (8, 0), (16, 0), (16, 1)])
        first_reveal = next(index for index, item in enumerate(events) if item[0] == "reveal")
        self.assertEqual(first_reveal, len(policies))
        self.assertTrue(all(item[0] == "save" for item in events[:first_reveal]))
        self.assertIs(result["n16_complete_uncertainty_result"], complete_n16)
        self.assertTrue(
            all("uncertainty_result" not in item for item in result["strict_prefixes"])
        )

    def test_result_rejects_a_forged_scientific_digest(self):
        blocks = _blocks()
        acceptance = pilot.evaluate_pilot_acceptance(blocks)
        with self.assertRaisesRegex(ValueError, "scientific result digest"):
            pilot.ProspectivePilotResult(
                source_revision="a" * 40,
                worker_count=4,
                protocol_digest="b" * 64,
                scientific_result_digest="c" * 64,
                block_results=tuple(blocks),
                acceptance=acceptance,
                compute_budget_inputs={},
                wall_seconds=1.0,
                cpu_seconds=2.0,
                peak_rss_bytes=3,
            )

    def test_cli_source_preflight_rejects_dirty_head(self):
        dirty = Mock(stdout=" M scientific.py\n")
        with patch("subprocess.run", return_value=dirty):
            with self.assertRaisesRegex(ValueError, "clean committed"):
                _committed_source_revision(Path("."))

    def test_cli_source_preflight_rejects_head_revision_mismatch(self):
        responses = (Mock(stdout=""), Mock(stdout="b" * 40 + "\n"))
        with patch("subprocess.run", side_effect=responses):
            with self.assertRaisesRegex(ValueError, "does not equal clean HEAD"):
                _require_clean_head(Path("."), "a" * 40)

    def test_cli_rejects_an_unrelated_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "imported source tree"):
                _require_project_root(Path(directory))

    def test_cli_rechecks_clean_head_after_run_before_writing(self):
        revision = "a" * 40
        saved = SimpleNamespace(
            json_path=Path("pilot.json"),
            report_path=Path("pilot.txt"),
            hash_path=Path("pilot.sha256"),
            archive_size_bytes=1,
            json_sha256="b" * 64,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                pilot_cli,
                "_committed_source_revision",
                return_value=revision,
            ),
            patch.object(pilot_cli, "_require_clean_head") as clean,
            patch.object(
                pilot_cli,
                "run_prospective_draw_count_pilot",
                return_value=object(),
            ),
            patch.object(
                pilot_cli,
                "save_prospective_pilot_artifacts",
                return_value=saved,
            ) as save,
        ):
            pilot_cli.main(
                (
                    "--execute-disposable-pilot",
                    "--json",
                    str(Path(directory) / "pilot.json"),
                )
            )
        self.assertEqual(clean.call_count, 2)
        self.assertEqual(clean.call_args_list[-1].args[1], revision)
        save.assert_called_once()


class ProspectivePilotArchiveTests(unittest.TestCase):
    def _result(self):
        blocks = _blocks()
        acceptance = pilot.evaluate_pilot_acceptance(blocks)
        scientific = pilot.prospective_pilot_scientific_payload(
            source_revision="a" * 40,
            protocol_digest="b" * 64,
            block_results=blocks,
            acceptance=acceptance,
        )
        digest = pilot._digest(
            "thermotwin.prospective_pilot.scientific_result",
            scientific,
        )
        budget = {
            "one_source_case_count": 6,
            "two_source_case_count": 6,
            "zero_source_case_count": 0,
            "unknown_source_case_count": 0,
            "measured_worker_count": 4,
            "runtime_host_manifest": {
                "python_version": "3.10.12",
                "operating_system": "Darwin",
                "machine": "arm64",
            },
            "conservative_measured_throughput_phase_estimates": [
                {
                    "partition": spec.name,
                    "estimated_wall_seconds_at_measured_worker_count": 10.0,
                    "estimated_cpu_seconds": 20.0,
                }
                for spec in pilot.PROSPECTIVE_PARTITION_PLAN[1:]
            ],
            "draw_count_linear_projections": [
                {
                    "draw_count": count,
                    "estimated_full_campaign_wall_seconds_at_measured_concurrency": 100.0,
                    "estimated_full_campaign_cpu_seconds": 200.0,
                }
                for count in (4, 8, 16)
            ],
        }
        return pilot.ProspectivePilotResult(
            source_revision="a" * 40,
            worker_count=4,
            protocol_digest="b" * 64,
            scientific_result_digest=digest,
            block_results=tuple(blocks),
            acceptance=acceptance,
            compute_budget_inputs=budget,
            wall_seconds=1.0,
            cpu_seconds=2.0,
            peak_rss_bytes=3,
        )

    def test_complete_archive_records_exact_size_and_detached_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = pilot.save_prospective_pilot_artifacts(
                self._result(),
                json_path=root / "pilot.json",
                report_path=root / "pilot.txt",
                hash_path=root / "pilot.sha256",
            )
            payload = json.loads(saved.json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["archive_size_bytes"], saved.archive_size_bytes)
            self.assertEqual(saved.archive_size_bytes, saved.json_path.stat().st_size)
            self.assertEqual(
                saved.json_sha256,
                hashlib.sha256(saved.json_path.read_bytes()).hexdigest(),
            )
            manifest = saved.hash_path.read_text(encoding="utf-8")
            self.assertIn(saved.json_sha256, manifest)
            self.assertIn(saved.report_sha256, manifest)

    def test_archive_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "pilot.json"
            json_path.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                pilot.save_prospective_pilot_artifacts(
                    self._result(),
                    json_path=json_path,
                    report_path=root / "pilot.txt",
                    hash_path=root / "pilot.sha256",
                )


if __name__ == "__main__":
    unittest.main()
