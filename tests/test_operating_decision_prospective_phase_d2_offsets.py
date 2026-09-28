from copy import deepcopy
import math
from pathlib import Path
import unittest

from thermotwin.studies.operating_decision import POLICY_NAMES
import thermotwin.studies.operating_decision_prospective_phase_d2_offsets as d2
from thermotwin.studies.operating_decision_realism import STAGE3_TRUTH_CONDITIONS


class PhaseD2OffsetTests(unittest.TestCase):
    def test_case_nonconformity_formula_and_invalid_interval_policy(self):
        self.assertEqual(
            d2.phase_d2_case_nonconformity({"lower": 1.0, "upper": 2.0}, 1.5),
            (0.0, "valid"),
        )
        self.assertEqual(
            d2.phase_d2_case_nonconformity({"lower": 1.0, "upper": 2.0}, 0.75),
            (0.25, "valid"),
        )
        score, status = d2.phase_d2_case_nonconformity(
            {"lower": 1.0, "upper": 2.0}, 2.4
        )
        self.assertAlmostEqual(score, 0.4)
        self.assertEqual(status, "valid")
        for interval in (
            None,
            {},
            {"lower": 2.0, "upper": 1.0},
            {"lower": 0.0, "upper": math.inf},
        ):
            score, _ = d2.phase_d2_case_nonconformity(interval, 0.0)
            self.assertEqual(score, math.inf)

    @staticmethod
    def _archive(*, failing_policy=None):
        blocks = []
        for block in range(20):
            cases = []
            for family_index, family in enumerate(STAGE3_TRUTH_CONDITIONS):
                margin = float(block + family_index) / 100.0
                fixed = {}
                for policy_index, policy in enumerate(POLICY_NAMES):
                    width = float(policy_index + 1) / 1000.0
                    fixed[policy] = {
                        "saved_before_reveal": {
                            "margin_envelope": {
                                "lower": margin - width,
                                "upper": margin + width,
                            }
                        },
                        "post_reveal_score": {"revealed": {"true_margin": margin}},
                    }
                if failing_policy is not None and block >= 17:
                    fixed[failing_policy]["failure"] = {"error_type": "x"}
                cases.append(
                    {
                        "truth_condition": family,
                        "device_token": f"device-{block}-{family_index}",
                        "fixed_policy_results": fixed,
                    }
                )
            blocks.append({"block": block, "cases": cases})
        return {
            "partition": "p1_development_tuning",
            "block_results": blocks,
        }

    def test_analysis_retains_all_cases_and_authorizes_d3_when_finite(self):
        result = d2.analyze_validated_phase_d1_archive(self._archive())
        self.assertEqual(len(result["block_records"]), 20)
        self.assertTrue(
            all(len(item["case_scores"]) == 3 for item in result["block_records"])
        )
        self.assertEqual(
            result["offset_estimation"]["status"], "finite_offsets_ready"
        )
        self.assertTrue(
            result["offset_estimation"]["tuning_winner_selection_authorized"]
        )

    def test_three_failed_blocks_make_required_offset_infinite(self):
        policy = POLICY_NAMES[2]
        result = d2.analyze_validated_phase_d1_archive(
            self._archive(failing_policy=policy)
        )
        estimate = result["offset_estimation"]
        self.assertEqual(estimate["status"], "phase_d_infeasible")
        self.assertEqual(estimate["infinite_actions"], [policy])
        self.assertFalse(estimate["tuning_winner_selection_authorized"])

    def test_report_states_the_gate_and_no_refitting(self):
        analysis = self._archive()
        body = d2.analyze_validated_phase_d1_archive(analysis)
        payload = {
            "analysis_source_revision": "a" * 40,
            "input_phase_d1": {"json_sha256": "b" * 64, "scientific_result_digest": "c" * 64},
            "scientific_result_digest": "d" * 64,
            **body,
        }
        report = d2.format_phase_d2_offset_report(payload)
        self.assertIn("Refitting performed: no", report)
        self.assertIn(
            "PASS: all four required development offsets are finite", report
        )
        self.assertIn("Phase D3", report)

    def test_output_paths_refuse_overwrite(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / d2.PHASE_D2_JSON_NAME).write_text("occupied")
            with self.assertRaisesRegex(FileExistsError, "refuses to overwrite"):
                d2.save_phase_d2_offset_artifacts({}, output_directory=root)


if __name__ == "__main__":
    unittest.main()
