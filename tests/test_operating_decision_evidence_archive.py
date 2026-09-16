import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.archive_operating_decision_evidence import (
    archive_diagnostics,
    recompute_diagnostic_evidence_digest,
)


class OperatingDecisionEvidenceArchiveTests(unittest.TestCase):
    @staticmethod
    def diagnostic(partition="r2_parent_rehearsal", blocks=1):
        records = [
            {
                "case": index,
                "saved_decision": {
                    "decision": "insufficient_evidence",
                    "decision_computation_seconds": index / 10.0,
                },
            }
            for index in range(blocks * 12)
        ]
        payload = {
            "schema_version": 1,
            "generator_version": "operating_decision_generator_v4",
            "random_stream_protocol_version": "thermotwin-operating-decision-rng-v2",
            "partition": {
                "campaign": "unit_external_evidence",
                "name": partition,
                "paired_blocks_per_family": blocks,
            },
            "stream_manifest": [],
            "stream_audit": {
                "ok": True,
                "use_count": 24,
                "unique_key_count": 12,
                "unique_seed_count": 12,
                "declared_pairings": [],
                "unintended_reuses": [],
            },
            "records": records,
            "measurement_boundaries": {
                "energized_schedule_duration_excludes_resets": True,
            },
        }
        payload["deterministic_evidence_digest"] = (
            recompute_diagnostic_evidence_digest(payload)
        )
        return payload

    @staticmethod
    def head(root):
        return subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def write_diagnostic(self, path, payload=None):
        value = self.diagnostic() if payload is None else payload
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path

    def archive(self, root, diagnostic, *, archive_name="archives", manifest_name="manifest.json"):
        repository = Path(__file__).resolve().parents[1]
        return archive_diagnostics(
            (diagnostic,),
            archive_directory=root / archive_name,
            manifest_path=repository / manifest_name,
            repository_root=repository,
            command="rehearse-parent",
            source_commit=self.head(repository),
            canonical_arguments=("--artifact", "thermotwin/PARENT.json", "--workers", "4"),
            release_tag="unit-evidence-v1",
            retrieval_base_url=(
                "https://github.com/example/project/releases/download/unit-evidence-v1"
            ),
        )

    def test_deterministic_archive_manifest_and_round_trip(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostic = self.write_diagnostic(root / "rehearsal.json")
            manifests = []
            archive_bytes = []
            for index in range(2):
                manifest_path = repository / f"unit-evidence-manifest-{index}.json"
                self.addCleanup(manifest_path.unlink, missing_ok=True)
                manifest = self.archive(
                    root,
                    diagnostic,
                    archive_name=f"archives-{index}",
                    manifest_name=manifest_path.name,
                )
                manifests.append(copy.deepcopy(manifest))
                archive = root / f"archives-{index}" / manifest["evidence"][0]["archive"]["asset_name"]
                archive_bytes.append(archive.read_bytes())
                self.assertEqual(gzip.decompress(archive.read_bytes()), diagnostic.read_bytes())
                self.assertEqual(archive.read_bytes()[4:8], b"\x00\x00\x00\x00")
            self.assertEqual(archive_bytes[0], archive_bytes[1])
            self.assertEqual(manifests[0], manifests[1])
            manifest = manifests[0]
            claimed = manifest.pop("manifest_digest")
            canonical = json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            self.assertEqual(claimed, hashlib.sha256(canonical).hexdigest())

    def test_tampered_rows_are_rejected(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.diagnostic()
            payload["records"][0]["case"] = 999
            diagnostic = self.write_diagnostic(root / "tampered.json", payload)
            manifest = repository / "unit-tampered-manifest.json"
            self.addCleanup(manifest.unlink, missing_ok=True)
            with self.assertRaisesRegex(ValueError, "digest does not match"):
                self.archive(root, diagnostic, manifest_name=manifest.name)

    def test_unclean_stream_audit_is_rejected(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.diagnostic()
            payload["stream_audit"]["ok"] = False
            payload["deterministic_evidence_digest"] = recompute_diagnostic_evidence_digest(payload)
            diagnostic = self.write_diagnostic(root / "unclean.json", payload)
            manifest = repository / "unit-unclean-manifest.json"
            self.addCleanup(manifest.unlink, missing_ok=True)
            with self.assertRaisesRegex(ValueError, "audit is not clean"):
                self.archive(root, diagnostic, manifest_name=manifest.name)

    def test_source_commit_and_https_are_enforced(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostic = self.write_diagnostic(root / "rehearsal.json")
            with self.assertRaisesRegex(ValueError, "repository HEAD"):
                archive_diagnostics(
                    (diagnostic,),
                    archive_directory=root / "archives",
                    manifest_path=repository / "unit-bad-commit.json",
                    repository_root=repository,
                    command="rehearse-parent",
                    source_commit="0" * 40,
                    canonical_arguments=("--workers", "4"),
                    release_tag="unit-v1",
                    retrieval_base_url="https://example.com/unit-v1",
                )
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                archive_diagnostics(
                    (diagnostic,),
                    archive_directory=root / "archives",
                    manifest_path=repository / "unit-bad-url.json",
                    repository_root=repository,
                    command="rehearse-parent",
                    source_commit=self.head(repository),
                    canonical_arguments=("--workers", "4"),
                    release_tag="unit-v1",
                    retrieval_base_url="http://example.com/unit-v1",
                )

    def test_duplicate_partition_and_overwrite_are_rejected(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_diagnostic(root / "first.json")
            second = self.write_diagnostic(root / "second.json")
            manifest = repository / "unit-duplicate-manifest.json"
            self.addCleanup(manifest.unlink, missing_ok=True)
            with self.assertRaisesRegex(ValueError, "partitions must be unique"):
                archive_diagnostics(
                    (first, second),
                    archive_directory=root / "archives",
                    manifest_path=manifest,
                    repository_root=repository,
                    command="rehearse-parent",
                    source_commit=self.head(repository),
                    canonical_arguments=("--workers", "4"),
                    release_tag="unit-v1",
                    retrieval_base_url="https://example.com/unit-v1",
                )
            existing = repository / "unit-existing-manifest.json"
            existing.write_text("{}\n", encoding="utf-8")
            self.addCleanup(existing.unlink, missing_ok=True)
            with self.assertRaises(FileExistsError):
                self.archive(root, first, manifest_name=existing.name)


if __name__ == "__main__":
    unittest.main()
