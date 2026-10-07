from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thermotwin.studies.operating_decision import POLICY_NAMES
from thermotwin.studies.operating_decision_prospective_phase_e import PHASE_E_REHEARSAL_PARTITION
from thermotwin.studies.operating_decision_prospective_phase_e_calibration import SELECTOR_PROCEDURE, case_nonconformity
from thermotwin.studies.operating_decision_prospective_pilot import (
    PROSPECTIVE_CALIBRATION_PARTITION, PROSPECTIVE_RESERVED_PARTITION,
)
import thermotwin.studies.operating_decision_prospective_phase_e_execution as execution


def case_fixture():
    fixed = {
        p: {"saved_before_reveal": {"margin_envelope": {"lower": -.5, "upper": -.1},
                                    "verifications": [{"passed": True}]},
            "post_reveal_score": {"revealed": {"true_margin": -.25},
                                  "scored": {"total_diagnostic_energy": 60.,
                                             "diagnostic_run_count": 2,
                                             "extra_sensor_count": 0,
                                             "energized_schedule_time_seconds": 160.}}}
        for p in POLICY_NAMES
    }
    return {"block": 0, "truth_condition": "matched_four_state", "device_token": "device",
            "selected_policy_at_n16_primary": "stop_now", "fixed_policy_results": fixed}


class PhaseEExecutionTests(unittest.TestCase):
    def test_partition_entry_point_refuses_reserved_and_development(self):
        self.assertEqual(execution.phase_e_partition(PROSPECTIVE_CALIBRATION_PARTITION).block_count, 100)
        self.assertEqual(execution.phase_e_partition(PHASE_E_REHEARSAL_PARTITION).block_count, 1)
        for name in (PROSPECTIVE_RESERVED_PARTITION, "p1_development_tuning", "custom"):
            with self.assertRaises(ValueError):
                execution.phase_e_partition(name)

    def test_case_records_preserve_all_procedures_and_offsets(self):
        rows = execution.calibration_case_records(case_fixture())
        self.assertEqual(len(rows), 5)
        self.assertEqual({r["procedure"] for r in rows}, {SELECTOR_PROCEDURE, *POLICY_NAMES})
        self.assertEqual(next(r for r in rows if r["procedure"] == "fixed_voltage")["development_offset_kelvin"], .074)
        self.assertEqual(next(r for r in rows if r["procedure"] == SELECTOR_PROCEDURE)["selected_action"], "stop_now")

    def test_selection_and_fixed_save_failures_stay_in_denominator(self):
        case = case_fixture()
        case["selected_policy_at_n16_primary"] = None
        case["fixed_policy_results"]["fixed_voltage"] = {
            "saved_before_reveal": None, "post_reveal_score": None, "failure": "decision_not_saved"}
        rows = execution.calibration_case_records(case)
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(case_nonconformity(r) == float("inf") for r in rows), 2)
        self.assertIsNone(next(r for r in rows if r["procedure"] == SELECTOR_PROCEDURE)["realized_resources"]["total_diagnostic_energy"])

    def test_disagreeing_or_nonfinite_shared_truth_is_material_incident(self):
        for value in (None, -.3):
            case = case_fixture()
            case["fixed_policy_results"]["fixed_voltage"]["post_reveal_score"]["revealed"]["true_margin"] = value
            with self.assertRaises(ValueError):
                execution.calibration_case_records(case)

    def test_block_seal_detects_partition_context_and_byte_tampering(self):
        context = {"source_revision": "a"*40, "source_manifest_digest": "b"*64,
                   "protocol_digest": "c"*64, "runtime_identity": {"runtime": "fixture"}}
        block = {"block": 0, "cases": [], "timing": {"wall_seconds": 1.}}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            execution, "validate_phase_e_block", side_effect=lambda x, **kw: deepcopy(x)
        ):
            path = Path(directory) / "block-000.json"
            execution._save_block(path, block, partition_name=PHASE_E_REHEARSAL_PARTITION, context=context)
            execution.load_phase_e_block(path, partition_name=PHASE_E_REHEARSAL_PARTITION, context=context)
            for changed_context in (dict(context, source_revision="d"*40), dict(context, protocol_digest="e"*64)):
                with self.assertRaises(ValueError):
                    execution.load_phase_e_block(path, partition_name=PHASE_E_REHEARSAL_PARTITION, context=changed_context)
            with self.assertRaises(ValueError):
                execution.load_phase_e_block(path, partition_name=PROSPECTIVE_CALIBRATION_PARTITION, context=context)
            with self.assertRaises(FileExistsError):
                execution._save_block(path, block, partition_name=PHASE_E_REHEARSAL_PARTITION, context=context)
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                execution.load_phase_e_block(path, partition_name=PHASE_E_REHEARSAL_PARTITION, context=context)

    def test_scientific_digest_ignores_timing_but_not_numerical_data(self):
        context = {"source_revision": "a"*40, "source_manifest_digest": "b"*64,
                   "protocol_digest": "c"*64, "runtime_identity": {}}
        block = {"block": 0, "cases": [], "observation": [1., 2.],
                 "timing": {"wall_seconds": 1.}}
        with patch.object(execution, "validate_phase_e_block", side_effect=lambda x, **kw: deepcopy(x)):
            a, _ = execution.seal_phase_e_block(block, partition_name=PHASE_E_REHEARSAL_PARTITION, context=context)
            changed = deepcopy(block)
            changed["timing"]["wall_seconds"] = 100.
            b, _ = execution.seal_phase_e_block(changed, partition_name=PHASE_E_REHEARSAL_PARTITION, context=context)
            self.assertEqual(a["scientific_block_digest"], b["scientific_block_digest"])
            self.assertNotEqual(a["archive_content_digest"], b["archive_content_digest"])
            changed["observation"][0] = 3.
            c, _ = execution.seal_phase_e_block(changed, partition_name=PHASE_E_REHEARSAL_PARTITION, context=context)
            self.assertNotEqual(a["scientific_block_digest"], c["scientific_block_digest"])

    def test_constructed_probe_is_not_scientific_calibration(self):
        seed = {"block_result": {"block": 0, "observations": [1., 2.]}}
        with tempfile.TemporaryDirectory() as directory:
            record = execution.constructed_phase_e_archive_probe(seed, output_directory=directory)
            archive = json.loads((Path(directory) / "constructed-100-block-archive.json").read_bytes())
        self.assertTrue(record["roundtrip_passed"])
        self.assertEqual(len(archive["constructed_blocks"]), 100)
        self.assertEqual(record["constructed_calibration_record_count"], 1500)
        self.assertEqual(record["calibration_order_statistic_rank"], 96)
        self.assertEqual(archive["scientific_use"], "prohibited_constructed_resource_evidence_only")

    def test_uncommitted_gate_is_rejected_before_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text('{"calibration_generation_authorized":true}')
            with patch.object(execution.subprocess, "run") as run:
                run.return_value.stdout = b"different committed bytes"
                with self.assertRaises(ValueError):
                    execution.load_committed_generation_gate(path, gate_repository_root=directory, context={})

    def test_incomplete_wrong_source_or_reserved_authorizing_gate_rejected(self):
        context = {"source_revision": "a"*40, "source_manifest_digest": "b"*64,
                   "protocol_digest": "c"*64, "runtime_identity": {}}
        gate = {
            "protocol_version": "phase_e_generation_gate_v1", **context,
            "calibration_generation_authorized": True, "reserved_generation_authorized": False,
            "checks": {k: True for k in ("protocol_tests", "exact_source_ci", "disposable_roundtrip",
                                        "exact_scientific_replay", "constructed_100_block_roundtrip",
                                        "process_tree_monitor_preflight")},
            "ci": {"head_sha": "a"*40, "conclusion": "success", "status": "completed"},
        }
        self.assertEqual(execution.validate_generation_gate(gate, context=context), gate)
        for invalid in (dict(gate, source_revision="d"*40), dict(gate, reserved_generation_authorized=True),
                        dict(gate, checks={}), dict(gate, ci={"conclusion": "success"})):
            with self.assertRaises(ValueError):
                execution.validate_generation_gate(invalid, context=context)

    def test_lock_recovery_requires_dead_process_and_preserves_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.lock"
            path.write_text(json.dumps({"pid": 123, "source_revision": "a"*40}))
            with patch.object(execution.os, "kill", return_value=None):
                with self.assertRaises(ValueError):
                    execution.recover_interrupted_execution_lock(directory, source_revision="a"*40)
            with patch.object(execution.os, "kill", side_effect=ProcessLookupError):
                with self.assertRaises(ValueError):
                    execution.recover_interrupted_execution_lock(directory, source_revision="b"*40)
                preserved = execution.recover_interrupted_execution_lock(directory, source_revision="a"*40)
            self.assertEqual(len(preserved), 1)
            self.assertFalse(path.exists())
            self.assertEqual(json.loads(Path(preserved[0]).read_bytes())["pid"], 123)


if __name__ == "__main__":
    unittest.main()
