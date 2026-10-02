import json
from copy import deepcopy
from pathlib import Path
import unittest

from thermotwin.studies.operating_decision_prospective_phase_d_freeze import (
    PHASE_D_FREEZE_ARTIFACT_DIGEST,
    prospective_phase_d_freeze_artifact,
    prospective_phase_d_freeze_digest,
    validate_prospective_phase_d_freeze,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = (
    PROJECT_ROOT
    / "thermotwin"
    / "OPERATING_DECISION_PROSPECTIVE_PHASE_D_PROVISIONAL_FREEZE.json"
)


class ProspectivePhaseDFreezeTests(unittest.TestCase):
    def test_tracked_artifact_matches_code_and_digest(self):
        artifact = json.loads(FREEZE_PATH.read_bytes())
        self.assertEqual(artifact, prospective_phase_d_freeze_artifact())
        self.assertEqual(
            artifact["artifact_digest"], PHASE_D_FREEZE_ARTIFACT_DIGEST
        )
        self.assertEqual(
            prospective_phase_d_freeze_digest(), PHASE_D_FREEZE_ARTIFACT_DIGEST
        )
        self.assertEqual(validate_prospective_phase_d_freeze(artifact), artifact)

    def test_freeze_binds_selector_maps_and_only_internal_check(self):
        artifact = prospective_phase_d_freeze_artifact()
        self.assertEqual(artifact["selector"]["draw_count"], 16)
        self.assertEqual(
            artifact["primary_design"]["selected_action_counts"],
            {
                "stop_now": 34,
                "fixed_thermal": 16,
                "fixed_voltage": 5,
                "fixed_face_temperature": 5,
                "selection_failure": 0,
            },
        )
        self.assertEqual(artifact["map_sensitivity"]["map_cell_count"], 48)
        authorization = artifact["authorization"]
        self.assertTrue(authorization["development_internal_check_authorized_now"])
        self.assertFalse(authorization["independent_calibration_authorized_now"])
        self.assertFalse(authorization["reserved_evaluation_authorized_now"])

    def test_tampering_is_rejected(self):
        artifact = prospective_phase_d_freeze_artifact()
        tampered = deepcopy(artifact)
        tampered["selector"]["minimum_utility_per_cost"] = 0.0
        with self.assertRaisesRegex(ValueError, "committed record"):
            validate_prospective_phase_d_freeze(tampered)


if __name__ == "__main__":
    unittest.main()
