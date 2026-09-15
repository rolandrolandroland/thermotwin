from pathlib import Path
import tempfile
import unittest

from thermotwin.reports.operating_decision_replication import main
from thermotwin.studies.operating_decision_replication_protocol import (
    load_corrected_generator_freeze,
    verify_corrected_generator_freeze,
)


class OperatingDecisionReplicationReportTests(unittest.TestCase):
    def test_freeze_generator_command_writes_and_verifies_content_artifact(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generator.json"
            self.assertEqual(
                main(
                    (
                        "freeze-generator",
                        "--repository-root",
                        str(repository),
                        "--output",
                        str(output),
                    )
                ),
                0,
            )
            artifact = load_corrected_generator_freeze(output)
            self.assertTrue(
                verify_corrected_generator_freeze(
                    artifact,
                    repository,
                ).ok
            )
            with self.assertRaises(FileExistsError):
                main(
                    (
                        "freeze-generator",
                        "--repository-root",
                        str(repository),
                        "--output",
                        str(output),
                    )
                )

    def test_freeze_generator_command_accepts_a_reduced_disposable_plan(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generator.json"
            self.assertEqual(
                main(
                    (
                        "freeze-generator",
                        "--repository-root",
                        str(repository),
                        "--output",
                        str(output),
                        "--campaign",
                        "disposable_cli_dry_run",
                        "--gate-development-blocks",
                        "10",
                        "--parent-calibration-blocks",
                        "10",
                        "--parent-rehearsal-blocks",
                        "10",
                        "--guard-development-blocks",
                        "10",
                        "--guard-calibration-blocks",
                        "10",
                        "--reserved-evaluation-blocks",
                        "10",
                        "--bootstrap-draws",
                        "1000",
                        "--bootstrap-partition",
                        "disposable_bootstrap",
                    )
                ),
                0,
            )

            artifact = load_corrected_generator_freeze(output)
            self.assertEqual(artifact.plan.campaign, "disposable_cli_dry_run")
            self.assertEqual(
                tuple(partition.block_count for partition in artifact.plan.partitions),
                (10, 10, 10, 10, 10, 10),
            )
            self.assertEqual(artifact.plan.bootstrap_draws, 1000)
            self.assertEqual(
                artifact.plan.bootstrap_partition_name,
                "disposable_bootstrap",
            )
            self.assertTrue(
                verify_corrected_generator_freeze(
                    artifact,
                    repository,
                    plan=artifact.plan,
                ).ok
            )


if __name__ == "__main__":
    unittest.main()
