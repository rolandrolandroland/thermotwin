"""Content-addressed numerical provenance for reserved decision campaigns."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import platform
import subprocess
from typing import Iterable, Mapping, Optional, Tuple, Union


SOURCE_MANIFEST_SCHEMA_VERSION = 1
SOURCE_MANIFEST_PROTOCOL_VERSION = "thermotwin-operating-decision-source-v1"
_HEX_DIGITS = frozenset("0123456789abcdef")
_COMMITTED_PATH_VERIFICATION_SEAL = object()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _normalize_relative_path(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("source path must be a string")
    if not value or value != value.strip() or "\\" in value:
        raise ValueError("source path must be a nonempty normalized POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise ValueError("source path must stay within the repository")
    return value


def _resolve_source(root: Path, relative_path: str) -> Path:
    relative_path = _normalize_relative_path(relative_path)
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    if candidate.is_symlink():
        raise ValueError(f"numerical source cannot be a symlink: {relative_path}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"numerical source is missing: {relative_path}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"numerical source escapes repository: {relative_path}") from error
    if not resolved.is_file():
        raise ValueError(f"numerical source is not a regular file: {relative_path}")
    return resolved


def _module_name_from_source_path(relative_path: str) -> Optional[str]:
    path = PurePosixPath(relative_path)
    if not path.parts or path.parts[0] != "thermotwin" or path.suffix != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def _runtime_origin_mismatches(root: Path, source_paths: Iterable[str]) -> Tuple[str, ...]:
    """Prove import resolution points at the repository whose bytes were hashed."""

    mismatches = []
    for relative_path in source_paths:
        module_name = _module_name_from_source_path(relative_path)
        if module_name is None:
            continue
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError) as error:
            mismatches.append(f"{module_name} cannot resolve ({type(error).__name__})")
            continue
        if spec is None or spec.origin is None:
            mismatches.append(f"{module_name} has no concrete source origin")
            continue
        expected = root.joinpath(*PurePosixPath(relative_path).parts).resolve()
        try:
            actual = Path(spec.origin).resolve(strict=True)
        except (OSError, RuntimeError):
            mismatches.append(f"{module_name} source origin is unreadable")
            continue
        if actual != expected:
            mismatches.append(
                f"{module_name} resolves to {actual}, expected {expected}"
            )
    return tuple(mismatches)


@dataclass(frozen=True, order=True)
class SourceFileHash:
    """One repository-relative numerical dependency."""

    path: str
    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_relative_path(self.path))
        _validate_digest("source-file hash", self.sha256)
        if (
            not isinstance(self.byte_count, int)
            or isinstance(self.byte_count, bool)
            or self.byte_count < 0
        ):
            raise ValueError("source-file byte count must be a nonnegative integer")


@dataclass(frozen=True)
class NumericalSourceManifest:
    """Frozen content and interpreter identity for numerical execution."""

    schema_version: int
    protocol_version: str
    python_implementation: str
    python_version: str
    files: Tuple[SourceFileHash, ...]
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("source-manifest schema version is unsupported")
        if (
            not isinstance(self.protocol_version, str)
            or not self.protocol_version.strip()
            or self.protocol_version != self.protocol_version.strip()
        ):
            raise ValueError("source-manifest protocol version must be nonempty")
        for name, value in (
            ("Python implementation", self.python_implementation),
            ("Python version", self.python_version),
        ):
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        files = tuple(self.files)
        object.__setattr__(self, "files", files)
        if not files or any(not isinstance(item, SourceFileHash) for item in files):
            raise ValueError("source manifest needs at least one file hash")
        paths = tuple(item.path for item in files)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("source-manifest files must be unique and sorted")
        _validate_digest("source-manifest digest", self.digest)
        if self.digest != _manifest_digest_from_fields(
            schema_version=self.schema_version,
            protocol_version=self.protocol_version,
            python_implementation=self.python_implementation,
            python_version=self.python_version,
            files=files,
        ):
            raise ValueError("source-manifest digest does not match its contents")


@dataclass(frozen=True)
class SourceManifestVerification:
    """A complete comparison between a frozen manifest and executing source."""

    added_dependencies: Tuple[str, ...]
    removed_dependencies: Tuple[str, ...]
    missing_files: Tuple[str, ...]
    changed_files: Tuple[str, ...]
    runtime_mismatches: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.added_dependencies,
                self.removed_dependencies,
                self.missing_files,
                self.changed_files,
                self.runtime_mismatches,
            )
        )

    def assert_valid(self) -> None:
        if self.ok:
            return
        details = []
        for label, values in (
            ("added dependencies", self.added_dependencies),
            ("removed dependencies", self.removed_dependencies),
            ("missing files", self.missing_files),
            ("changed files", self.changed_files),
            ("runtime mismatches", self.runtime_mismatches),
        ):
            if values:
                details.append(f"{label}: {', '.join(values)}")
        raise ValueError("numerical source-manifest verification failed; " + "; ".join(details))


@dataclass(frozen=True)
class CommittedPathVerification:
    """Paths proven byte-identical to blobs in one repository commit."""

    repository_root: str
    commit: str
    paths: Tuple[str, ...]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _COMMITTED_PATH_VERIFICATION_SEAL:
            raise ValueError("committed verification must be issued by Git verification")
        root = Path(self.repository_root)
        if not root.is_absolute():
            raise ValueError("committed verification root must be absolute")
        if (
            not isinstance(self.commit, str)
            or len(self.commit) not in (40, 64)
            or any(character not in _HEX_DIGITS for character in self.commit)
        ):
            raise ValueError("committed verification needs a full Git object id")
        paths = tuple(_normalize_relative_path(item) for item in self.paths)
        object.__setattr__(self, "paths", paths)
        if not paths or paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("committed verification paths must be unique and sorted")


def current_python_runtime() -> Tuple[str, str]:
    """Return the interpreter fields frozen into a manifest."""

    return platform.python_implementation(), platform.python_version()


def _manifest_material(
    *,
    schema_version: int,
    protocol_version: str,
    python_implementation: str,
    python_version: str,
    files: Tuple[SourceFileHash, ...],
) -> dict:
    return {
        "schema_version": schema_version,
        "protocol_version": protocol_version,
        "runtime": {
            "python_implementation": python_implementation,
            "python_version": python_version,
        },
        "files": [
            {
                "path": item.path,
                "sha256": item.sha256,
                "byte_count": item.byte_count,
            }
            for item in files
        ],
    }


def _manifest_digest_from_fields(
    *,
    schema_version: int,
    protocol_version: str,
    python_implementation: str,
    python_version: str,
    files: Tuple[SourceFileHash, ...],
) -> str:
    material = _manifest_material(
        schema_version=schema_version,
        protocol_version=protocol_version,
        python_implementation=python_implementation,
        python_version=python_version,
        files=files,
    )
    return _sha256_bytes(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def create_source_manifest(
    repository_root: Path | str,
    source_paths: Iterable[str],
    *,
    protocol_version: str = SOURCE_MANIFEST_PROTOCOL_VERSION,
) -> NumericalSourceManifest:
    """Hash an explicit allowlist of numerical dependencies."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    normalized = tuple(_normalize_relative_path(item) for item in source_paths)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("numerical source allowlist must be nonempty and unique")
    files = []
    for relative_path in sorted(normalized):
        raw = _resolve_source(root, relative_path).read_bytes()
        files.append(SourceFileHash(relative_path, _sha256_bytes(raw), len(raw)))
    implementation, version = current_python_runtime()
    frozen_files = tuple(files)
    digest = _manifest_digest_from_fields(
        schema_version=SOURCE_MANIFEST_SCHEMA_VERSION,
        protocol_version=protocol_version,
        python_implementation=implementation,
        python_version=version,
        files=frozen_files,
    )
    return NumericalSourceManifest(
        schema_version=SOURCE_MANIFEST_SCHEMA_VERSION,
        protocol_version=protocol_version,
        python_implementation=implementation,
        python_version=version,
        files=frozen_files,
        digest=digest,
    )


def verify_source_manifest(
    manifest: NumericalSourceManifest,
    repository_root: Path | str,
    expected_source_paths: Iterable[str],
) -> SourceManifestVerification:
    """Compare the frozen allowlist, file bytes, and executing interpreter."""

    if not isinstance(manifest, NumericalSourceManifest):
        raise TypeError("manifest must be a NumericalSourceManifest")
    root = Path(repository_root).expanduser().resolve(strict=True)
    expected = tuple(_normalize_relative_path(item) for item in expected_source_paths)
    if len(set(expected)) != len(expected):
        raise ValueError("expected numerical dependency paths must be unique")
    expected_set = set(expected)
    frozen_set = {item.path for item in manifest.files}
    added = tuple(sorted(expected_set - frozen_set))
    removed = tuple(sorted(frozen_set - expected_set))
    missing = []
    changed = []
    for item in manifest.files:
        try:
            source = _resolve_source(root, item.path)
        except ValueError:
            missing.append(item.path)
            continue
        raw = source.read_bytes()
        if len(raw) != item.byte_count or _sha256_bytes(raw) != item.sha256:
            changed.append(item.path)
    implementation, version = current_python_runtime()
    runtime = list(_runtime_origin_mismatches(root, expected))
    if implementation != manifest.python_implementation:
        runtime.append(
            f"implementation {implementation!r} != {manifest.python_implementation!r}"
        )
    if version != manifest.python_version:
        runtime.append(f"version {version!r} != {manifest.python_version!r}")
    return SourceManifestVerification(
        added_dependencies=added,
        removed_dependencies=removed,
        missing_files=tuple(missing),
        changed_files=tuple(changed),
        runtime_mismatches=tuple(runtime),
    )


def _committed_relative_path(
    root: Path,
    value: Union[Path, str],
) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        try:
            relative = candidate.resolve(strict=True).relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("committed path must stay within the repository") from error
    else:
        relative = _normalize_relative_path(candidate.as_posix())
    _resolve_source(root, relative)
    return _normalize_relative_path(relative)


def _git_output(root: Path, arguments: Tuple[str, ...]) -> bytes:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise ValueError("Git is required to verify committed artifacts") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            "committed artifact verification failed"
            + (f": {detail}" if detail else "")
        )
    return result.stdout


def verify_committed_paths(
    repository_root: Path | str,
    paths: Iterable[Union[Path, str]],
) -> CommittedPathVerification:
    """Prove every path is tracked and byte-identical to the current HEAD blob."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    top_level = Path(
        _git_output(root, ("rev-parse", "--show-toplevel"))
        .decode("utf-8")
        .strip()
    ).resolve(strict=True)
    if top_level != root:
        raise ValueError("repository root differs from the Git top level")
    commit = (
        _git_output(root, ("rev-parse", "--verify", "HEAD^{commit}"))
        .decode("ascii")
        .strip()
    )
    normalized = tuple(sorted(_committed_relative_path(root, item) for item in paths))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("committed path set must be nonempty and unique")
    for relative in normalized:
        try:
            committed = _git_output(root, ("show", f"{commit}:{relative}"))
        except ValueError as error:
            raise ValueError(f"path is not committed at HEAD: {relative}") from error
        current = _resolve_source(root, relative).read_bytes()
        if current != committed:
            raise ValueError(f"path differs from committed HEAD: {relative}")
    return CommittedPathVerification(
        str(root),
        commit,
        normalized,
        _COMMITTED_PATH_VERIFICATION_SEAL,
    )


def verify_committed_artifact_chain(
    repository_root: Path | str,
    artifact_path: Path | str,
    manifest: NumericalSourceManifest,
) -> CommittedPathVerification:
    """Verify one serialized artifact and every source file it binds at HEAD."""

    if not isinstance(manifest, NumericalSourceManifest):
        raise TypeError("committed artifact chain needs a numerical source manifest")
    return verify_committed_paths(
        repository_root,
        (artifact_path, *(item.path for item in manifest.files)),
    )


def source_manifest_payload(manifest: NumericalSourceManifest) -> dict:
    """Return a JSON-compatible payload including its content digest."""

    if not isinstance(manifest, NumericalSourceManifest):
        raise TypeError("manifest must be a NumericalSourceManifest")
    payload = _manifest_material(
        schema_version=manifest.schema_version,
        protocol_version=manifest.protocol_version,
        python_implementation=manifest.python_implementation,
        python_version=manifest.python_version,
        files=manifest.files,
    )
    payload["digest"] = manifest.digest
    return payload


def source_manifest_from_payload(payload: Mapping[str, object]) -> NumericalSourceManifest:
    """Strictly reconstruct a manifest from serialized data."""

    if not isinstance(payload, Mapping):
        raise TypeError("source-manifest payload must be a mapping")
    if set(payload) != {"schema_version", "protocol_version", "runtime", "files", "digest"}:
        raise ValueError("source-manifest payload has unexpected fields")
    runtime = payload["runtime"]
    files_payload = payload["files"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "python_implementation",
        "python_version",
    }:
        raise ValueError("source-manifest runtime is malformed")
    if not isinstance(files_payload, list):
        raise ValueError("source-manifest files must be a list")
    files = []
    for item in files_payload:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "sha256",
            "byte_count",
        }:
            raise ValueError("source-manifest file entry is malformed")
        files.append(
            SourceFileHash(
                path=item["path"],  # type: ignore[arg-type]
                sha256=item["sha256"],  # type: ignore[arg-type]
                byte_count=item["byte_count"],  # type: ignore[arg-type]
            )
        )
    return NumericalSourceManifest(
        schema_version=payload["schema_version"],  # type: ignore[arg-type]
        protocol_version=payload["protocol_version"],  # type: ignore[arg-type]
        python_implementation=runtime["python_implementation"],  # type: ignore[arg-type]
        python_version=runtime["python_version"],  # type: ignore[arg-type]
        files=tuple(files),
        digest=payload["digest"],  # type: ignore[arg-type]
    )


__all__ = [
    "SOURCE_MANIFEST_PROTOCOL_VERSION",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "NumericalSourceManifest",
    "CommittedPathVerification",
    "SourceFileHash",
    "SourceManifestVerification",
    "create_source_manifest",
    "current_python_runtime",
    "source_manifest_from_payload",
    "source_manifest_payload",
    "verify_source_manifest",
    "verify_committed_paths",
    "verify_committed_artifact_chain",
]
