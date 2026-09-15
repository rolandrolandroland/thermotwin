import json
import math
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from thermotwin.studies.operating_decision import STOP_NOW, default_fixed_policies
from thermotwin.studies.operating_decision_diagnostics import (
    corrected_trial_payload,
    save_diagnostic_payload,
)
from thermotwin.studies.operating_decision_random_streams import RandomStreamRegistry
from thermotwin.studies.operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
)
from thermotwin.studies.operating_decision_replication import (
    CorrectedPartition,
    build_corrected_blinded_case,
    corrected_partition_config,
    corrected_truth_for_block,
    decide_corrected_blinded_case,
    score_corrected_saved_decision,
)


class OperatingDecisionDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.partition = CorrectedPartition("diagnostic_unit", 1, "unit_campaign")
        base = OperatingDecisionRealismConfig()
        cls.config = corrected_partition_config(
            cls.partition,
            replace(base, sensor=replace(base.sensor, fit_iterations=1)),
        )
        registry = RandomStreamRegistry()
        truth = corrected_truth_for_block(cls.partition, 0, cls.config, registry)
        policy = next(item for item in default_fixed_policies() if item.name == STOP_NOW)
        case = build_corrected_blinded_case(
            STAGE3_TRUTH_CONDITIONS[0],
            0,
            policy,
            truth,
            cls.partition,
            cls.config,
            registry,
        )
        saved = decide_corrected_blinded_case(case, config=cls.config)
        cls.record = score_corrected_saved_decision(
            saved,
            truth_condition=STAGE3_TRUTH_CONDITIONS[0],
            block=0,
            case=case,
            truth=truth,
            partition=cls.partition,
            config=cls.config,
        )

    def test_trial_payload_retains_observations_fits_covariance_truth_and_resources(self):
        payload = corrected_trial_payload(self.record)
        self.assertEqual(payload["identity"]["policy"], STOP_NOW)
        self.assertTrue(payload["acquisition_runs"][0]["observations"])
        verification = payload["saved_decision"]["candidate_verifications"]
        self.assertEqual(len(verification), 2)
        self.assertTrue(verification[0]["fit"]["covariance"])
        optimizer = verification[0]["fit"]["optimizer"]
        self.assertIn(
            optimizer["termination_reason"],
            {
                "scaled_projected_gradient_tolerance",
                "step_and_objective_stagnation",
                "fixed_iteration_limit",
            },
        )
        self.assertIsInstance(optimizer["converged"], bool)
        self.assertIn("scaled_gradient_infinity_norm", optimizer)
        self.assertIn("scaled_projected_gradient_infinity_norm", optimizer)
        self.assertTrue(payload["truth_after_reveal"]["final_cold_face_temperature"])
        resources = payload["resources"]
        self.assertIn("nominal_selection_energy_proxy_joules", resources)
        self.assertIn("realized_total_terminal_energy_joules", resources)
        json.dumps(payload, allow_nan=False)

    def test_expected_infinite_failure_score_is_encoded_as_standard_json(self):
        saved = self.record.scored.saved
        verification = saved.verifications[0]._replace(normalized_score=math.inf)
        changed_saved = saved._replace(
            verifications=(verification, *saved.verifications[1:])
        )
        changed_record = replace(
            self.record,
            scored=self.record.scored._replace(saved=changed_saved),
        )
        payload = corrected_trial_payload(changed_record)
        encoded = payload["saved_decision"]["candidate_verifications"][0][
            "normalized_score"
        ]
        self.assertEqual(encoded, {"nonfinite_float": "positive_infinity"})
        json.dumps(payload, allow_nan=False)

    def test_writer_refuses_overwrite_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.json"
            save_diagnostic_payload({"ok": True}, path)
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            with self.assertRaises(FileExistsError):
                save_diagnostic_payload({"ok": False}, path)
            with self.assertRaises(ValueError):
                save_diagnostic_payload({"bad": float("nan")}, path.with_name("bad.json"))


if __name__ == "__main__":
    unittest.main()
