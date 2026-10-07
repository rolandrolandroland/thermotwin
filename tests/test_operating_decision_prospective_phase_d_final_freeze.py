from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from thermotwin.studies.operating_decision_prospective_phase_d_final_freeze import (
    load_prospective_phase_d_final_freeze,
    validate_phase_d_final_freeze_evidence,
    validate_prospective_phase_d_final_freeze,
)


class PhaseDFinalFreezeTests(unittest.TestCase):
    def test_committed_closeout_authenticates_and_keeps_future_partitions_closed(self):
        record = load_prospective_phase_d_final_freeze()
        evidence = record["evidence"]["phase_d6"]
        self.assertEqual(evidence["block_count"], 10)
        self.assertEqual(evidence["case_count"], 30)
        authorization = record["authorization"]
        self.assertTrue(authorization["phase_e_protocol_work_authorized_now"])
        self.assertTrue(
            authorization["development_internal_check_closed_against_regeneration"]
        )
        self.assertFalse(authorization["independent_calibration_authorized_now"])
        self.assertFalse(authorization["reserved_evaluation_authorized_now"])

    def test_decision_and_interval_coverage_remain_distinct(self):
        record = load_prospective_phase_d_final_freeze()
        result = record["internal_check_result"]
        overall = result["metrics"]["overall"]
        self.assertEqual(overall["definitive_decisions"], 24)
        self.assertEqual(overall["decision_coverage"], 24 / 30)
        self.assertEqual(overall["adjusted_interval_covered"], 27)
        self.assertEqual(result["one_candidate_stopping"]["case_count"], 6)
        self.assertEqual(
            result["one_candidate_stopping"]["adjusted_interval_covered"], 4
        )
        self.assertEqual(
            result["simultaneous_development_adjusted_interval_block_coverage"],
            {"count": 7, "denominator": 10},
        )

    def test_full_fixed_comparator_frontier_is_preserved(self):
        record = load_prospective_phase_d_final_freeze()
        comparators = record["fixed_policy_comparators"]
        self.assertEqual(set(comparators), {
            "stop_now", "fixed_thermal", "fixed_voltage", "fixed_face_temperature"
        })
        for comparator in comparators.values():
            self.assertEqual(comparator["overall"]["case_count"], 30)
            self.assertIn("temperature_dependent_contact", comparator["by"]["truth_family"])
        self.assertFalse(record["comparator_analysis"]["refitting_performed"])

    def test_changes_to_rule_evidence_or_authorization_are_rejected(self):
        original = load_prospective_phase_d_final_freeze()
        modifications = [
            ("selector", "minimum_utility_per_cost", 0.0),
            ("authorization", "independent_calibration_authorized_now", True),
            ("authorization", "reserved_evaluation_authorized_now", True),
            ("evidence", "phase_d6", {}),
        ]
        for parent, key, value in modifications:
            with self.subTest(parent=parent, key=key):
                changed = deepcopy(original)
                changed[parent][key] = value
                with self.assertRaises(ValueError):
                    validate_prospective_phase_d_final_freeze(changed)

    def test_descriptive_analysis_rejects_unauthenticated_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "archive bytes do not match"):
                validate_phase_d_final_freeze_evidence(path)


if __name__ == "__main__":
    unittest.main()
