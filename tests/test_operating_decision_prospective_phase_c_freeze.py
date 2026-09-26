import unittest
from copy import deepcopy

from thermotwin.studies.operating_decision_prospective_phase_c_freeze import (
    FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION,
    FROZEN_PROSPECTIVE_DRAW_COUNT,
    PHASE_C_EVIDENCE_SOURCE_REVISION,
    PHASE_C_FREEZE_PAYLOAD_DIGEST,
    prospective_phase_c_freeze_digest,
    prospective_phase_c_freeze_payload,
    validate_prospective_phase_c_freeze,
)


class ProspectivePhaseCFreezeTests(unittest.TestCase):
    def test_freeze_binds_p4_evidence_and_selected_rule(self):
        payload = prospective_phase_c_freeze_payload()
        self.assertEqual(
            PHASE_C_EVIDENCE_SOURCE_REVISION,
            "c0518f5885f8421a1f6f95d4f60c5dc5744cfb4a",
        )
        self.assertEqual(FROZEN_PROSPECTIVE_DRAW_COUNT, 16)
        self.assertEqual(FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION, 1)
        self.assertEqual(payload["selection"]["draw_count"], 16)
        self.assertEqual(
            payload["selection"]["n16_to_n32_action_agreement"],
            {"count": 12, "denominator": 12},
        )
        self.assertEqual(
            payload["evidence"]["parent"]["json_sha256"],
            "91f7d21c9e3a72bb8f341efc78ff7ce0a1a570fef0ff8540ed998e1655770bad",
        )
        self.assertEqual(
            payload["evidence"]["n32_followup"]["json_sha256"],
            "749e6896bbd90503faf75c6deaffd24a5ede63baf5e6e15fa84e2d08c12f66cf",
        )

    def test_compute_plan_is_complete_and_additive(self):
        plan = prospective_phase_c_freeze_payload()["compute_plan"]
        partitions = plan["planned_partitions"]
        self.assertEqual(sum(item["block_count"] for item in partitions), 230)
        self.assertEqual(sum(item["case_count"] for item in partitions), 690)
        self.assertAlmostEqual(
            sum(item["estimated_wall_seconds"] for item in partitions),
            plan["estimated_total_wall_seconds"],
        )
        self.assertAlmostEqual(
            sum(item["estimated_cpu_seconds"] for item in partitions),
            plan["estimated_total_cpu_seconds"],
        )
        self.assertEqual(
            sum(item["estimated_archive_bytes"] for item in partitions),
            plan["estimated_total_archive_bytes"],
        )
        self.assertFalse(plan["additional_sensor_quality_simulations_included"])

    def test_freeze_authorizes_only_development_tuning(self):
        authorization = prospective_phase_c_freeze_payload()["authorization"]
        self.assertTrue(authorization["phase_d_entry_authorized"])
        self.assertEqual(
            authorization["next_partition"],
            "p1_development_tuning",
        )
        self.assertFalse(authorization["development_internal_check_authorized_now"])
        self.assertFalse(authorization["independent_calibration_authorized_now"])
        self.assertFalse(authorization["reserved_evaluation_authorized_now"])

    def test_freeze_digest_is_stable_and_tampering_is_rejected(self):
        self.assertEqual(
            prospective_phase_c_freeze_digest(),
            PHASE_C_FREEZE_PAYLOAD_DIGEST,
        )
        payload = prospective_phase_c_freeze_payload()
        self.assertEqual(validate_prospective_phase_c_freeze(payload), payload)
        tampered = deepcopy(payload)
        tampered["selection"]["draw_count"] = 8
        with self.assertRaisesRegex(ValueError, "committed record"):
            validate_prospective_phase_c_freeze(tampered)


if __name__ == "__main__":
    unittest.main()
