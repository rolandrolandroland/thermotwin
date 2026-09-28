from pathlib import Path
import tempfile
import unittest

from thermotwin.studies.operating_decision import POLICY_NAMES
from thermotwin.studies.operating_decision_prospective import (
    ProspectiveDevelopmentOffsets,
)
from thermotwin.studies.operating_decision_prospective_phase_d import (
    PHASE_D_FINITE_OFFSETS_VERSION,
)
import thermotwin.studies.operating_decision_prospective_phase_d3_grid as d3
from thermotwin.studies.operating_decision_realism import STAGE3_TRUTH_CONDITIONS


class PhaseD3GridTests(unittest.TestCase):
    @staticmethod
    def _offsets():
        return ProspectiveDevelopmentOffsets(
            version=PHASE_D_FINITE_OFFSETS_VERSION,
            stop_now=0.0,
            fixed_thermal=0.0,
            fixed_voltage=0.074,
            fixed_face_temperature=0.098,
        )

    @staticmethod
    def _complete():
        baseline = 0.4
        widths = {
            "fixed_thermal": 0.39,
            # The D2 offsets make these padded widths 0.35 and 0.38 K.
            "fixed_voltage": 0.202,
            "fixed_face_temperature": 0.184,
        }
        actions = [
            {
                "policy_name": "stop_now",
                "eligible": True,
                "failure_reason": None,
                "uncertainty_before": baseline,
                "expected_uncertainty_after": baseline,
                "prospective_draw_count": 0,
                "source_summaries": [],
            }
        ]
        draws = []
        for policy_name in POLICY_NAMES[1:]:
            actions.append(
                {
                    "policy_name": policy_name,
                    "eligible": True,
                    "failure_reason": None,
                    "uncertainty_before": baseline,
                    "expected_uncertainty_after": widths[policy_name],
                    "prospective_draw_count": 1,
                    "source_summaries": [
                        {
                            "policy_name": policy_name,
                            "generator_model": "four_state",
                            "draw_count": 1,
                            "stable_draw_count": 1,
                            "failed_draw_count": 0,
                            "mean_after_width": widths[policy_name],
                            "eligible": True,
                            "max_unstable_draws": 0,
                        }
                    ],
                }
            )
            draws.append(
                {
                    "policy_name": policy_name,
                    "generator_model": "four_state",
                    "failed": False,
                    "raw_after_width": widths[policy_name],
                    "initially_admissible_became_inadmissible": [],
                }
            )
        return {
            "acquisition_evidence": {
                "snapshot": {
                    "candidate_models": ["five_state", "four_state"],
                    "admissible_candidate_models": ["four_state"],
                    "excluded_candidates": [
                        {
                            "model_name": "five_state",
                            "reason": "fit_reached_bound",
                        }
                    ],
                    "failed_candidate_models": [],
                    "model_intervals": [
                        {
                            "model_name": "four_state",
                            "estimate": 0.0,
                            "local_standard_error": 0.1,
                            "lower": -0.2,
                            "upper": 0.2,
                        }
                    ],
                    "provisional_margin_envelope": {
                        "lower": -0.2,
                        "upper": 0.2,
                    },
                    "selection_failure_reason": None,
                }
            },
            "action_uncertainties": actions,
            "draw_outcomes": draws,
        }

    @staticmethod
    def _scorecard():
        costs = {
            "stop_now": (0.0, 0.0),
            "fixed_thermal": (2.0, 1.3),
            "fixed_voltage": (1.0, 1.0),
            "fixed_face_temperature": (0.1, 1.0),
        }
        return {
            "action_costs": [
                {
                    "policy_name": policy,
                    "declared_cost": values[0],
                    "normalized_energy": values[1],
                }
                for policy, values in costs.items()
            ],
            "action_resources": [
                {
                    "policy_name": policy,
                    "added_run_count": 0 if policy == "stop_now" else 1,
                    "added_instruments": (
                        []
                        if policy in ("stop_now", "fixed_thermal")
                        else ["sensor"]
                    ),
                }
                for policy in POLICY_NAMES
            ],
        }

    @staticmethod
    def _fixed_policy_results(true_margin=0.1):
        intervals = {
            "stop_now": {"lower": -0.1, "upper": 0.2},
            "fixed_thermal": {"lower": -0.05, "upper": 0.15},
            "fixed_voltage": {"lower": 0.08, "upper": 0.2},
            "fixed_face_temperature": {"lower": 0.01, "upper": 0.2},
        }
        result = {}
        for index, policy in enumerate(POLICY_NAMES):
            result[policy] = {
                "saved_before_reveal": {
                    "margin_envelope": intervals[policy],
                    "verifications": [{"passed": True}],
                },
                "post_reveal_score": {
                    "revealed": {"true_margin": true_margin},
                    "scored": {
                        "diagnostic_run_count": 2 + index,
                        "energized_schedule_time_seconds": 160.0 + 80.0 * index,
                        "extra_sensor_count": int(index >= 2),
                        "total_diagnostic_energy": 60.0 + 10.0 * index,
                    },
                },
            }
        return result

    @classmethod
    def _archive(cls):
        scorecard = cls._scorecard()
        blocks = []
        for block in range(20):
            cases = []
            for family_index, family in enumerate(STAGE3_TRUTH_CONDITIONS):
                cases.append(
                    {
                        "block": block,
                        "truth_condition": family,
                        "device_token": f"device-{block}-{family_index}",
                        "n16_complete_uncertainty_result": cls._complete(),
                        "primary_prefixes": [
                            {
                                "draw_count": 16,
                                "max_unstable_draws_per_source_action": 1,
                                "costed_scorecard": scorecard,
                            }
                        ],
                        "fixed_policy_results": cls._fixed_policy_results(),
                    }
                )
            blocks.append({"block": block, "cases": cases})
        return {
            "partition": "p1_development_tuning",
            "block_results": blocks,
        }

    def test_thresholds_change_actions_decisions_losses_and_winner(self):
        result = d3.analyze_phase_d3_evidence(self._archive(), self._offsets())
        self.assertEqual(result["grid_row_count"], 81)
        actions = {
            case["selected_policy"]
            for row in result["grid_rows"]
            for case in row["case_results"]
        }
        self.assertEqual(actions, {"fixed_voltage", "fixed_face_temperature"})

        face_row = next(
            row for row in result["grid_rows"] if row["grid_values"] == [0.0, 0.0, 0.0, 0.0]
        )
        face_case = face_row["case_results"][0]
        self.assertEqual(face_case["selected_policy"], "fixed_face_temperature")
        self.assertEqual(face_case["adjusted_decision"]["raw_decision"], "approve")
        self.assertEqual(
            face_case["adjusted_decision"]["development_adjusted_decision"],
            "insufficient_evidence",
        )

        voltage_row = next(
            row
            for row in result["grid_rows"]
            if row["grid_values"] == [0.0, 0.0, 0.025, 0.0]
        )
        voltage_case = voltage_row["case_results"][0]
        self.assertEqual(voltage_case["selected_policy"], "fixed_voltage")
        self.assertEqual(
            voltage_case["adjusted_decision"]["development_adjusted_decision"],
            "approve",
        )
        self.assertLess(voltage_case["loss"], face_case["loss"])
        self.assertEqual(result["winning_grid_values"], [0.0, 0.0, 0.025, 0.0])
        self.assertGreaterEqual(result["winning_pre_lexicographic_tie_count"], 1)

    def test_primary_n16_allowance_is_required(self):
        archive = self._archive()
        archive["block_results"][0]["cases"][0]["primary_prefixes"][0][
            "max_unstable_draws_per_source_action"
        ] = 0
        with self.assertRaisesRegex(ValueError, "frozen allowance"):
            d3.analyze_phase_d3_evidence(archive, self._offsets())

    def test_source_allowlist_contains_d3_entry_points(self):
        self.assertIn(
            "thermotwin/studies/operating_decision_prospective_phase_d3_grid.py",
            d3.PHASE_D3_ANALYSIS_SOURCE_PATHS,
        )
        self.assertIn(
            "thermotwin/reports/operating_decision_prospective_phase_d3.py",
            d3.PHASE_D3_ANALYSIS_SOURCE_PATHS,
        )

    def test_artifact_save_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / d3.PHASE_D3_JSON_NAME).write_text("occupied")
            with self.assertRaisesRegex(FileExistsError, "refuses to overwrite"):
                d3.save_phase_d3_grid_artifacts({}, output_directory=root)


if __name__ == "__main__":
    unittest.main()
