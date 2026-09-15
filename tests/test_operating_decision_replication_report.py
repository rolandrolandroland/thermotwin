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


if __name__ == "__main__":
    unittest.main()
