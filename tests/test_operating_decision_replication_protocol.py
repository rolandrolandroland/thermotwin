import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from thermotwin.studies.operating_decision_realism import OperatingDecisionRealismConfig
from thermotwin.studies.operating_decision_replication_protocol import (
    CORRECTED_NUMERICAL_SOURCE_PATHS,
    CorrectedReplicationPlan,
    build_corrected_generator_freeze,
    corrected_generator_freeze_from_payload,
    corrected_generator_freeze_payload,
    load_corrected_generator_freeze,
    save_corrected_generator_freeze,
    verify_corrected_generator_freeze,
)


class OperatingDecisionReplicationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "generator.py").write_text("VERSION = 2\n", encoding="utf-8")
        (self.root / "README.md").write_text("docs\n", encoding="utf-8")
        self.paths = ("generator.py",)
        self.plan = CorrectedReplicationPlan()
        self.config = OperatingDecisionRealismConfig()

    def tearDown(self):
        self.temporary.cleanup()

    def artifact(self):
        return build_corrected_generator_freeze(
            self.root,
            self.config,
            self.plan,
            source_paths=self.paths,
        )

    def test_default_plan_has_fresh_semantic_partitions_and_reserved_size(self):
        names = tuple(item.name for item in self.plan.partitions)
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("r2_reserved_evaluation", names)
        self.assertEqual(
            self.plan.partition("r2_reserved_evaluation").block_count,
            50,
        )
        self.assertEqual(self.plan.bootstrap_draws, 20_000)

    def test_freeze_is_content_addressed_and_round_trips(self):
        artifact = self.artifact()
        payload = corrected_generator_freeze_payload(artifact)
        self.assertNotIn("source_revision", payload)
        self.assertEqual(
            corrected_generator_freeze_from_payload(
                json.loads(json.dumps(payload))
            ),
            artifact,
        )
        verification = verify_corrected_generator_freeze(
            artifact,
            self.root,
            self.config,
            self.plan,
            expected_source_paths=self.paths,
        )
        self.assertTrue(verification.ok)

    def test_source_physics_and_partition_drift_fail_before_generation(self):
        artifact = self.artifact()
        (self.root / "generator.py").write_text("VERSION = 3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed files"):
            verify_corrected_generator_freeze(
                artifact,
                self.root,
                self.config,
                self.plan,
                expected_source_paths=self.paths,
            )
        with self.assertRaisesRegex(ValueError, "physical protocol"):
            verify_corrected_generator_freeze(
                artifact,
                self.root,
                replace(self.config, local_interval_multiplier=3.0),
                self.plan,
                expected_source_paths=self.paths,
            )
        with self.assertRaisesRegex(ValueError, "plan differs"):
            verify_corrected_generator_freeze(
                artifact,
                self.root,
                self.config,
                replace(self.plan, parent_rehearsal_blocks=21),
                expected_source_paths=self.paths,
            )

    def test_documentation_change_does_not_invalidate_freeze(self):
        artifact = self.artifact()
        (self.root / "README.md").write_text("changed docs\n", encoding="utf-8")
        self.assertTrue(
            verify_corrected_generator_freeze(
                artifact,
                self.root,
                self.config,
                self.plan,
                expected_source_paths=self.paths,
            ).ok
        )

    def test_save_refuses_overwrite_and_loader_rejects_revision_label(self):
        artifact = self.artifact()
        destination = self.root / "freeze.json"
        save_corrected_generator_freeze(artifact, destination)
        self.assertEqual(load_corrected_generator_freeze(destination), artifact)
        with self.assertRaises(FileExistsError):
            save_corrected_generator_freeze(artifact, destination)
        payload = corrected_generator_freeze_payload(artifact)
        payload["source_revision"] = "NOT_A_REAL_REVISION"
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            corrected_generator_freeze_from_payload(payload)

    def test_plan_rejects_small_calibration_and_bootstrap_counts(self):
        with self.assertRaisesRegex(ValueError, "conformal calibration"):
            CorrectedReplicationPlan(parent_calibration_blocks=9)
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            CorrectedReplicationPlan(bootstrap_draws=999)

    def test_source_allowlist_covers_the_loaded_local_runtime(self):
        repository = Path(__file__).resolve().parents[1]
        probe = """
import json
from pathlib import Path
import sys
import thermotwin.reports.operating_decision_replication
import thermotwin.studies.operating_decision_replication_guard

repository = Path(sys.argv[1]).resolve()
loaded = set()
for module in tuple(sys.modules.values()):
    source = getattr(module, "__file__", None)
    if source is None:
        continue
    try:
        relative = Path(source).resolve().relative_to(repository)
    except ValueError:
        continue
    if relative.parts[0] == "thermotwin" and relative.suffix == ".py":
        loaded.add(relative.as_posix())
print(json.dumps(sorted(loaded)))
"""
        result = subprocess.run(
            (sys.executable, "-c", probe, str(repository)),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        loaded_sources = set(json.loads(result.stdout))
        self.assertFalse(loaded_sources - set(CORRECTED_NUMERICAL_SOURCE_PATHS))


if __name__ == "__main__":
    unittest.main()
