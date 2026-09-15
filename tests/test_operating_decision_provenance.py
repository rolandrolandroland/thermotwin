import json
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import unittest

from thermotwin.studies.operating_decision_provenance import (
    CommittedPathVerification,
    create_source_manifest,
    verify_committed_artifact_chain,
    source_manifest_from_payload,
    source_manifest_payload,
    verify_committed_paths,
    verify_source_manifest,
)


class OperatingDecisionProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
        (self.root / "solver.py").write_text("def solve(): return 2\n", encoding="utf-8")
        (self.root / "README.md").write_text("documentation\n", encoding="utf-8")
        self.paths = ("solver.py", "model.py")

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self):
        return create_source_manifest(self.root, self.paths)

    def test_manifest_is_sorted_deterministic_and_round_trips(self):
        first = self.manifest()
        second = create_source_manifest(self.root, reversed(self.paths))
        self.assertEqual(first, second)
        self.assertEqual(tuple(item.path for item in first.files), ("model.py", "solver.py"))
        payload = source_manifest_payload(first)
        self.assertNotIn("source_revision", payload)
        encoded = json.loads(json.dumps(payload))
        self.assertEqual(source_manifest_from_payload(encoded), first)
        self.assertTrue(verify_source_manifest(first, self.root, self.paths).ok)

    def test_changed_or_missing_numerical_source_fails_verification(self):
        manifest = self.manifest()
        (self.root / "model.py").write_text("MODEL = 2\n", encoding="utf-8")
        changed = verify_source_manifest(manifest, self.root, self.paths)
        self.assertEqual(changed.changed_files, ("model.py",))
        with self.assertRaisesRegex(ValueError, "changed files: model.py"):
            changed.assert_valid()
        (self.root / "solver.py").unlink()
        missing = verify_source_manifest(manifest, self.root, self.paths)
        self.assertEqual(missing.missing_files, ("solver.py",))

    def test_dependency_allowlist_changes_are_reported(self):
        manifest = self.manifest()
        (self.root / "new.py").write_text("NEW = True\n", encoding="utf-8")
        result = verify_source_manifest(
            manifest,
            self.root,
            ("model.py", "new.py"),
        )
        self.assertEqual(result.added_dependencies, ("new.py",))
        self.assertEqual(result.removed_dependencies, ("solver.py",))
        self.assertFalse(result.ok)

    def test_documentation_outside_allowlist_does_not_invalidate_source(self):
        manifest = self.manifest()
        (self.root / "README.md").write_text("revised documentation\n", encoding="utf-8")
        self.assertTrue(verify_source_manifest(manifest, self.root, self.paths).ok)

    def test_package_source_must_be_the_code_resolved_by_the_runtime(self):
        package = self.root / "thermotwin"
        package.mkdir()
        (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        paths = ("thermotwin/__init__.py",)
        manifest = create_source_manifest(self.root, paths)
        verification = verify_source_manifest(manifest, self.root, paths)
        self.assertFalse(verification.ok)
        self.assertTrue(
            any("thermotwin resolves to" in item for item in verification.runtime_mismatches)
        )

    def test_runtime_and_digest_tampering_are_rejected(self):
        manifest = self.manifest()
        with self.assertRaisesRegex(ValueError, "digest does not match"):
            replace(manifest, python_version="0.0.0")
        payload = source_manifest_payload(manifest)
        payload["digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest does not match"):
            source_manifest_from_payload(payload)

    def test_invalid_paths_duplicates_symlinks_and_payloads_are_rejected(self):
        outside = self.root.parent / "outside.py"
        outside.write_text("outside\n", encoding="utf-8")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        for path in ("../outside.py", str(outside), "./model.py", "model\\.py"):
            with self.subTest(path=path):
                with self.assertRaises((TypeError, ValueError)):
                    create_source_manifest(self.root, (path,))
        with self.assertRaisesRegex(ValueError, "unique"):
            create_source_manifest(self.root, ("model.py", "model.py"))
        link = self.root / "link.py"
        try:
            link.symlink_to(self.root / "model.py")
        except (NotImplementedError, OSError):
            pass
        else:
            with self.assertRaisesRegex(ValueError, "symlink"):
                create_source_manifest(self.root, ("link.py",))
        payload = source_manifest_payload(self.manifest())
        payload["revision"] = "fake"
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            source_manifest_from_payload(payload)

    def test_committed_paths_must_match_head_blobs(self):
        subprocess.run(("git", "init", "-q", str(self.root)), check=True)
        subprocess.run(
            ("git", "-C", str(self.root), "add", "model.py", "solver.py"),
            check=True,
        )
        subprocess.run(
            (
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=ThermoTwin Test",
                "-c",
                "user.email=thermotwin@example.invalid",
                "commit",
                "-q",
                "-m",
                "freeze sources",
            ),
            check=True,
        )
        verified = verify_committed_paths(self.root, self.paths)
        self.assertEqual(verified.paths, ("model.py", "solver.py"))
        self.assertIn(len(verified.commit), (40, 64))

        (self.root / "model.py").write_text("MODEL = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from committed HEAD"):
            verify_committed_paths(self.root, self.paths)
        (self.root / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not committed at HEAD"):
            verify_committed_paths(self.root, ("README.md",))

    def test_committed_verification_records_cannot_be_fabricated(self):
        with self.assertRaisesRegex(ValueError, "issued by Git verification"):
            CommittedPathVerification(
                str(self.root.resolve()),
                "0" * 40,
                ("model.py",),
                object(),
            )

    def test_committed_artifact_chain_checks_artifact_and_manifest_sources(self):
        manifest = self.manifest()
        artifact = self.root / "artifact.json"
        artifact.write_text("{}\n", encoding="utf-8")
        subprocess.run(("git", "init", "-q", str(self.root)), check=True)
        subprocess.run(
            (
                "git",
                "-C",
                str(self.root),
                "add",
                "artifact.json",
                "model.py",
                "solver.py",
            ),
            check=True,
        )
        subprocess.run(
            (
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=ThermoTwin Test",
                "-c",
                "user.email=thermotwin@example.invalid",
                "commit",
                "-q",
                "-m",
                "freeze artifact chain",
            ),
            check=True,
        )
        verified = verify_committed_artifact_chain(self.root, artifact, manifest)
        self.assertEqual(
            verified.paths,
            ("artifact.json", "model.py", "solver.py"),
        )

        artifact.write_text('{"changed": true}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact.json"):
            verify_committed_artifact_chain(self.root, artifact, manifest)
        artifact.write_text("{}\n", encoding="utf-8")
        (self.root / "solver.py").write_text(
            "def solve(): return 3\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "solver.py"):
            verify_committed_artifact_chain(self.root, artifact, manifest)


if __name__ == "__main__":
    unittest.main()
