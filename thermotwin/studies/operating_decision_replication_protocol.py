"""Frozen protocol and source boundary for the corrected replication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Tuple

from .operating_decision_provenance import (
    NumericalSourceManifest,
    SourceManifestVerification,
    create_source_manifest,
    source_manifest_from_payload,
    source_manifest_payload,
    verify_source_manifest,
)
from .operating_decision_random_streams import RANDOM_STREAM_PROTOCOL_VERSION
from .operating_decision_realism import OperatingDecisionRealismConfig
from .operating_decision_replication import (
    CORRECTED_GENERATOR_VERSION,
    CORRECTED_REPLICATION_CONFIG,
    CORRECTED_REPLICATION_CAMPAIGN,
    RESERVED_EVALUATION_PARTITION_NAME,
    CorrectedPartition,
    corrected_physical_protocol_digest,
)


CORRECTED_REPLICATION_PROTOCOL_VERSION = "operating_decision_audit_replication_v2"
GENERATOR_FREEZE_SCHEMA_VERSION = 1

CORRECTED_NUMERICAL_SOURCE_PATHS = (
    "pyproject.toml",
    "thermotwin/__init__.py",
    "thermotwin/_public_api.py",
    "thermotwin/core/__init__.py",
    "thermotwin/core/controls.py",
    "thermotwin/design/__init__.py",
    "thermotwin/design/ag2se_substitution.py",
    "thermotwin/design/codesign/__init__.py",
    "thermotwin/design/codesign/campaign.py",
    "thermotwin/design/codesign/evaluation.py",
    "thermotwin/design/codesign/models.py",
    "thermotwin/design/codesign/optimization.py",
    "thermotwin/design/codesign/robustness.py",
    "thermotwin/design/codesign/sampling.py",
    "thermotwin/design/contact_process_window.py",
    "thermotwin/design/control_comparison.py",
    "thermotwin/design/literature_materials.py",
    "thermotwin/design/material_pair.py",
    "thermotwin/design/materials.py",
    "thermotwin/design/operating_map.py",
    "thermotwin/design/power_electronics.py",
    "thermotwin/design/pulse_map.py",
    "thermotwin/inference/__init__.py",
    "thermotwin/inference/contact_resistance.py",
    "thermotwin/inference/distributed_experiment_selection.py",
    "thermotwin/inference/distributed_identifiability.py",
    "thermotwin/inference/distributed_profile_likelihood.py",
    "thermotwin/inference/distributed_properties.py",
    "thermotwin/inference/distributed_regularization.py",
    "thermotwin/inference/experiment_selection.py",
    "thermotwin/inference/joint_thermal_parameters.py",
    "thermotwin/inference/sparse_sensors.py",
    "thermotwin/numerics/__init__.py",
    "thermotwin/numerics/integration.py",
    "thermotwin/numerics/matrices.py",
    "thermotwin/numerics/statistics.py",
    "thermotwin/observations/__init__.py",
    "thermotwin/observations/bias.py",
    "thermotwin/observations/distributed.py",
    "thermotwin/observations/hardware.py",
    "thermotwin/observations/lag.py",
    "thermotwin/observations/metadata.py",
    "thermotwin/observations/missingness.py",
    "thermotwin/observations/noise.py",
    "thermotwin/observations/test_stand.py",
    "thermotwin/physics/__init__.py",
    "thermotwin/physics/distributed.py",
    "thermotwin/physics/four_node.py",
    "thermotwin/physics/thermoelectric.py",
    "thermotwin/physics/two_node.py",
    "thermotwin/reports/__init__.py",
    "thermotwin/reports/operating_decision_replication.py",
    "thermotwin/simulation/__init__.py",
    "thermotwin/simulation/distributed.py",
    "thermotwin/simulation/four_node_diagnostics.py",
    "thermotwin/simulation/four_node_experiments.py",
    "thermotwin/simulation/interface_mass_mismatch.py",
    "thermotwin/simulation/operating_realism.py",
    "thermotwin/simulation/temperature_dependent_contact.py",
    "thermotwin/simulation/temporary_face_sensor.py",
    "thermotwin/simulation/two_node_diagnostics.py",
    "thermotwin/simulation/two_node_experiments.py",
    "thermotwin/studies/__init__.py",
    "thermotwin/studies/operating_decision.py",
    "thermotwin/studies/operating_decision_calibration.py",
    "thermotwin/studies/operating_decision_diagnostics.py",
    "thermotwin/studies/operating_decision_provenance.py",
    "thermotwin/studies/operating_decision_random_streams.py",
    "thermotwin/studies/operating_decision_realism.py",
    "thermotwin/studies/operating_decision_replication.py",
    "thermotwin/studies/operating_decision_replication_calibration.py",
    "thermotwin/studies/operating_decision_replication_guard.py",
    "thermotwin/studies/operating_decision_replication_protocol.py",
    "thermotwin/studies/operating_decision_resources.py",
    "thermotwin/studies/sensor_model_discrimination.py",
    "thermotwin/operating_decision_replication.py",
)


@dataclass(frozen=True)
class CorrectedReplicationPlan:
    """Predeclared semantic partitions; no arithmetic seed ranges are used."""

    campaign: str = CORRECTED_REPLICATION_CAMPAIGN
    gate_development_blocks: int = 20
    parent_calibration_blocks: int = 30
    parent_rehearsal_blocks: int = 20
    guard_development_blocks: int = 30
    guard_calibration_blocks: int = 30
    reserved_evaluation_blocks: int = 50
    bootstrap_draws: int = 20_000
    bootstrap_partition_name: str = "r2_bootstrap"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.campaign, str)
            or not self.campaign.strip()
            or self.campaign != self.campaign.strip()
        ):
            raise ValueError("corrected replication campaign must be nonempty")
        for name in (
            "gate_development_blocks",
            "parent_calibration_blocks",
            "parent_rehearsal_blocks",
            "guard_development_blocks",
            "guard_calibration_blocks",
            "reserved_evaluation_blocks",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.gate_development_blocks < 10:
            raise ValueError("gate development needs at least ten paired blocks")
        if self.parent_calibration_blocks < 10 or self.guard_calibration_blocks < 10:
            raise ValueError("conformal calibration needs at least ten paired blocks")
        if (
            not isinstance(self.bootstrap_draws, int)
            or isinstance(self.bootstrap_draws, bool)
            or self.bootstrap_draws < 1_000
        ):
            raise ValueError("paired bootstrap needs at least 1,000 draws")
        if (
            not isinstance(self.bootstrap_partition_name, str)
            or not self.bootstrap_partition_name.strip()
            or self.bootstrap_partition_name != self.bootstrap_partition_name.strip()
        ):
            raise ValueError("bootstrap partition name must be nonempty")
        names = tuple(partition.name for partition in self.partitions)
        if len(set((*names, self.bootstrap_partition_name))) != len(names) + 1:
            raise ValueError("corrected replication partition names must be unique")

    @property
    def partitions(self) -> Tuple[CorrectedPartition, ...]:
        definitions = (
            ("r2_gate_development", self.gate_development_blocks),
            ("r2_parent_calibration", self.parent_calibration_blocks),
            ("r2_parent_rehearsal", self.parent_rehearsal_blocks),
            ("r2_guard_development", self.guard_development_blocks),
            ("r2_guard_calibration", self.guard_calibration_blocks),
            (RESERVED_EVALUATION_PARTITION_NAME, self.reserved_evaluation_blocks),
        )
        return tuple(
            CorrectedPartition(name, count, self.campaign)
            for name, count in definitions
        )

    def partition(self, name: str) -> CorrectedPartition:
        matches = tuple(item for item in self.partitions if item.name == name)
        if len(matches) != 1:
            raise ValueError("corrected replication has no unique named partition")
        return matches[0]


def _plan_payload(plan: CorrectedReplicationPlan) -> dict:
    return {
        "campaign": plan.campaign,
        "partitions": [
            {"name": item.name, "paired_blocks_per_family": item.block_count}
            for item in plan.partitions
        ],
        "bootstrap": {
            "partition": plan.bootstrap_partition_name,
            "draws": plan.bootstrap_draws,
        },
    }


def _artifact_material(
    *,
    physical_protocol_digest: str,
    plan: CorrectedReplicationPlan,
    source_manifest: NumericalSourceManifest,
) -> dict:
    return {
        "schema_version": GENERATOR_FREEZE_SCHEMA_VERSION,
        "protocol_version": CORRECTED_REPLICATION_PROTOCOL_VERSION,
        "generator_version": CORRECTED_GENERATOR_VERSION,
        "random_stream_protocol_version": RANDOM_STREAM_PROTOCOL_VERSION,
        "physical_protocol_digest": physical_protocol_digest,
        "plan": _plan_payload(plan),
        "source_manifest": source_manifest_payload(source_manifest),
        "state": "generator_frozen_no_reserved_evaluation",
    }


def _artifact_digest(
    *,
    physical_protocol_digest: str,
    plan: CorrectedReplicationPlan,
    source_manifest: NumericalSourceManifest,
) -> str:
    encoded = json.dumps(
        _artifact_material(
            physical_protocol_digest=physical_protocol_digest,
            plan=plan,
            source_manifest=source_manifest,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CorrectedGeneratorFreeze:
    """A content-addressed generator freeze created before scientific data."""

    physical_protocol_digest: str
    plan: CorrectedReplicationPlan
    source_manifest: NumericalSourceManifest
    artifact_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.physical_protocol_digest, str)
            or len(self.physical_protocol_digest) != 64
        ):
            raise ValueError("physical protocol digest must be SHA-256")
        if not isinstance(self.plan, CorrectedReplicationPlan):
            raise TypeError("generator freeze needs a corrected replication plan")
        if not isinstance(self.source_manifest, NumericalSourceManifest):
            raise TypeError("generator freeze needs a numerical source manifest")
        expected = _artifact_digest(
            physical_protocol_digest=self.physical_protocol_digest,
            plan=self.plan,
            source_manifest=self.source_manifest,
        )
        if self.artifact_digest != expected:
            raise ValueError("generator-freeze digest does not match its contents")


def build_corrected_generator_freeze(
    repository_root: Path | str,
    config: OperatingDecisionRealismConfig = CORRECTED_REPLICATION_CONFIG,
    plan: CorrectedReplicationPlan = CorrectedReplicationPlan(),
    *,
    source_paths: Iterable[str] = CORRECTED_NUMERICAL_SOURCE_PATHS,
) -> CorrectedGeneratorFreeze:
    """Freeze numerical bytes, runtime, physics, and fresh partitions."""

    manifest = create_source_manifest(repository_root, source_paths)
    physical_digest = corrected_physical_protocol_digest(config)
    return CorrectedGeneratorFreeze(
        physical_protocol_digest=physical_digest,
        plan=plan,
        source_manifest=manifest,
        artifact_digest=_artifact_digest(
            physical_protocol_digest=physical_digest,
            plan=plan,
            source_manifest=manifest,
        ),
    )


def verify_corrected_generator_freeze(
    artifact: CorrectedGeneratorFreeze,
    repository_root: Path | str,
    config: OperatingDecisionRealismConfig = CORRECTED_REPLICATION_CONFIG,
    plan: CorrectedReplicationPlan = CorrectedReplicationPlan(),
    *,
    expected_source_paths: Iterable[str] = CORRECTED_NUMERICAL_SOURCE_PATHS,
) -> SourceManifestVerification:
    """Reject protocol, partition, runtime, or numerical-source drift."""

    if not isinstance(artifact, CorrectedGeneratorFreeze):
        raise TypeError("artifact must be a CorrectedGeneratorFreeze")
    if artifact.plan != plan:
        raise ValueError("corrected replication plan differs from the frozen artifact")
    if artifact.physical_protocol_digest != corrected_physical_protocol_digest(config):
        raise ValueError("physical protocol differs from the frozen artifact")
    verification = verify_source_manifest(
        artifact.source_manifest,
        repository_root,
        expected_source_paths,
    )
    verification.assert_valid()
    return verification


def corrected_generator_freeze_payload(artifact: CorrectedGeneratorFreeze) -> dict:
    if not isinstance(artifact, CorrectedGeneratorFreeze):
        raise TypeError("artifact must be a CorrectedGeneratorFreeze")
    payload = _artifact_material(
        physical_protocol_digest=artifact.physical_protocol_digest,
        plan=artifact.plan,
        source_manifest=artifact.source_manifest,
    )
    payload["artifact_digest"] = artifact.artifact_digest
    return payload


def _plan_from_payload(payload: Mapping[str, object]) -> CorrectedReplicationPlan:
    if set(payload) != {"campaign", "partitions", "bootstrap"}:
        raise ValueError("corrected replication plan payload is malformed")
    partitions = payload["partitions"]
    bootstrap = payload["bootstrap"]
    if not isinstance(partitions, list) or len(partitions) != 6:
        raise ValueError("corrected replication plan needs six partitions")
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {"partition", "draws"}:
        raise ValueError("corrected bootstrap payload is malformed")
    expected_names = (
        "r2_gate_development",
        "r2_parent_calibration",
        "r2_parent_rehearsal",
        "r2_guard_development",
        "r2_guard_calibration",
        RESERVED_EVALUATION_PARTITION_NAME,
    )
    counts = []
    for expected_name, item in zip(expected_names, partitions):
        if not isinstance(item, Mapping) or set(item) != {
            "name",
            "paired_blocks_per_family",
        }:
            raise ValueError("corrected partition payload is malformed")
        if item["name"] != expected_name:
            raise ValueError("corrected partition order or identity changed")
        counts.append(item["paired_blocks_per_family"])
    return CorrectedReplicationPlan(
        campaign=payload["campaign"],  # type: ignore[arg-type]
        gate_development_blocks=counts[0],  # type: ignore[arg-type]
        parent_calibration_blocks=counts[1],  # type: ignore[arg-type]
        parent_rehearsal_blocks=counts[2],  # type: ignore[arg-type]
        guard_development_blocks=counts[3],  # type: ignore[arg-type]
        guard_calibration_blocks=counts[4],  # type: ignore[arg-type]
        reserved_evaluation_blocks=counts[5],  # type: ignore[arg-type]
        bootstrap_draws=bootstrap["draws"],  # type: ignore[arg-type]
        bootstrap_partition_name=bootstrap["partition"],  # type: ignore[arg-type]
    )


def corrected_generator_freeze_from_payload(
    payload: Mapping[str, object],
) -> CorrectedGeneratorFreeze:
    expected = {
        "schema_version",
        "protocol_version",
        "generator_version",
        "random_stream_protocol_version",
        "physical_protocol_digest",
        "plan",
        "source_manifest",
        "state",
        "artifact_digest",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("generator-freeze payload has unexpected fields")
    if payload["schema_version"] != GENERATOR_FREEZE_SCHEMA_VERSION:
        raise ValueError("generator-freeze schema version changed")
    if payload["protocol_version"] != CORRECTED_REPLICATION_PROTOCOL_VERSION:
        raise ValueError("corrected replication protocol version changed")
    if payload["generator_version"] != CORRECTED_GENERATOR_VERSION:
        raise ValueError("corrected generator version changed")
    if payload["random_stream_protocol_version"] != RANDOM_STREAM_PROTOCOL_VERSION:
        raise ValueError("random-stream protocol version changed")
    if payload["state"] != "generator_frozen_no_reserved_evaluation":
        raise ValueError("generator-freeze state is invalid")
    plan_payload = payload["plan"]
    manifest_payload = payload["source_manifest"]
    if not isinstance(plan_payload, Mapping) or not isinstance(manifest_payload, Mapping):
        raise ValueError("generator-freeze nested payload is malformed")
    return CorrectedGeneratorFreeze(
        physical_protocol_digest=payload["physical_protocol_digest"],  # type: ignore[arg-type]
        plan=_plan_from_payload(plan_payload),
        source_manifest=source_manifest_from_payload(manifest_payload),
        artifact_digest=payload["artifact_digest"],  # type: ignore[arg-type]
    )


def save_corrected_generator_freeze(
    artifact: CorrectedGeneratorFreeze,
    path: Path | str,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        corrected_generator_freeze_payload(artifact),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return destination


def load_corrected_generator_freeze(path: Path | str) -> CorrectedGeneratorFreeze:
    source = Path(path).expanduser().resolve(strict=True)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return corrected_generator_freeze_from_payload(payload)


__all__ = [
    "CORRECTED_NUMERICAL_SOURCE_PATHS",
    "CORRECTED_REPLICATION_PROTOCOL_VERSION",
    "GENERATOR_FREEZE_SCHEMA_VERSION",
    "CorrectedGeneratorFreeze",
    "CorrectedReplicationPlan",
    "build_corrected_generator_freeze",
    "corrected_generator_freeze_from_payload",
    "corrected_generator_freeze_payload",
    "load_corrected_generator_freeze",
    "save_corrected_generator_freeze",
    "verify_corrected_generator_freeze",
]
