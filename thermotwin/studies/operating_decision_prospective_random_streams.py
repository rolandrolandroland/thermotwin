"""Semantic random streams for prospective operating-decision calculations.

The prospective selector simulates future measurements from acquisition-only
fits.  These streams are deliberately separate from the completed corrected
replication namespace.  A complete semantic key, rather than arithmetic seed
offsets, identifies every draw.

Parameter draws may be shared across actions as a declared common-random-number
pairing.  Probe draws belong only to the face-temperature action.  Run-bias and
white-noise streams are always specific to one action, run, and channel.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Optional, Tuple

from .operating_decision import (
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    default_fixed_policies,
)
from .sensor_model_discrimination import ALL_CHANNELS, MODEL_NAMES


PROSPECTIVE_RANDOM_STREAM_PROTOCOL_VERSION = (
    "thermotwin-operating-decision-prospective-rng-v1"
)
PROSPECTIVE_RANDOM_STREAM_DOMAIN = (
    "thermotwin.operating_decision.prospective.random_stream"
)

PARAMETER_DRAW = "parameter_draw"
PROBE_DRAW = "probe_draw"
RUN_BIAS = "run_bias"
WHITE_NOISE = "white_noise"
PROSPECTIVE_STREAM_PURPOSES = (
    PARAMETER_DRAW,
    PROBE_DRAW,
    RUN_BIAS,
    WHITE_NOISE,
)
PROSPECTIVE_ACQUISITION_ACTIONS = (
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    FIXED_FACE_TEMPERATURE,
)

_ACTION_INDEX = {
    action: index for index, action in enumerate(PROSPECTIVE_ACQUISITION_ACTIONS)
}
_ACTION_REGIMES = {
    policy.name: {
        regime.name: frozenset(regime.channels)
        for regime in policy.additional_regimes
    }
    for policy in default_fixed_policies()
    if policy.name in PROSPECTIVE_ACQUISITION_ACTIONS
}


def _validate_label(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, trimmed string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} cannot contain control characters")


def _validate_optional_label(name: str, value: Optional[str]) -> None:
    if value is not None:
        _validate_label(name, value)


def _validate_nonnegative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_sha256(name: str, value: str) -> None:
    _validate_label(name, value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ProspectiveRandomStreamNamespace:
    """Identity shared by every predictive stream for one acquisition case."""

    campaign: str
    partition: str
    block: int
    acquisition_evidence_digest: str
    protocol_version: str = PROSPECTIVE_RANDOM_STREAM_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _validate_label("protocol version", self.protocol_version)
        if self.protocol_version != PROSPECTIVE_RANDOM_STREAM_PROTOCOL_VERSION:
            raise ValueError("prospective random-stream protocol is unsupported")
        _validate_label("campaign", self.campaign)
        _validate_label("partition", self.partition)
        _validate_nonnegative_integer("block", self.block)
        _validate_sha256(
            "acquisition evidence digest",
            self.acquisition_evidence_digest,
        )


@dataclass(frozen=True)
class ProspectiveRandomStreamKey:
    """Complete semantic identity of one prospective pseudorandom stream."""

    protocol_version: str
    campaign: str
    partition: str
    block: int
    acquisition_evidence_digest: str
    generator_model: str
    draw_index: int
    purpose: str
    action: Optional[str] = None
    run: Optional[str] = None
    channel: Optional[str] = None

    def __post_init__(self) -> None:
        namespace = ProspectiveRandomStreamNamespace(
            campaign=self.campaign,
            partition=self.partition,
            block=self.block,
            acquisition_evidence_digest=self.acquisition_evidence_digest,
            protocol_version=self.protocol_version,
        )
        del namespace
        if self.generator_model not in MODEL_NAMES:
            raise ValueError("prospective stream has an unknown generator model")
        _validate_nonnegative_integer("draw index", self.draw_index)
        if self.purpose not in PROSPECTIVE_STREAM_PURPOSES:
            raise ValueError("prospective stream has an unknown purpose")
        _validate_optional_label("action", self.action)
        _validate_optional_label("run", self.run)
        _validate_optional_label("channel", self.channel)

        if self.purpose == PARAMETER_DRAW:
            if any(value is not None for value in (self.action, self.run, self.channel)):
                raise ValueError(
                    "parameter-draw keys must omit action, run, and channel"
                )
            return

        if self.purpose == PROBE_DRAW:
            if self.action != FIXED_FACE_TEMPERATURE:
                raise ValueError("probe-draw keys belong only to the face action")
            if self.run is not None or self.channel is not None:
                raise ValueError("probe-draw keys must omit run and channel")
            return

        if self.action not in PROSPECTIVE_ACQUISITION_ACTIONS:
            raise ValueError("observation streams need a prospective action")
        if self.run is None or self.channel is None:
            raise ValueError("bias/noise keys require action, run, and channel")
        regimes = _ACTION_REGIMES[self.action]
        if self.run not in regimes:
            raise ValueError("observation stream run does not belong to its action")
        if self.channel not in ALL_CHANNELS or self.channel not in regimes[self.run]:
            raise ValueError("observation stream channel does not belong to its run")

    @classmethod
    def from_namespace(
        cls,
        namespace: ProspectiveRandomStreamNamespace,
        *,
        generator_model: str,
        draw_index: int,
        purpose: str,
        action: Optional[str] = None,
        run: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> "ProspectiveRandomStreamKey":
        if not isinstance(namespace, ProspectiveRandomStreamNamespace):
            raise TypeError("namespace must be a prospective random-stream namespace")
        return cls(
            protocol_version=namespace.protocol_version,
            campaign=namespace.campaign,
            partition=namespace.partition,
            block=namespace.block,
            acquisition_evidence_digest=namespace.acquisition_evidence_digest,
            generator_model=generator_model,
            draw_index=draw_index,
            purpose=purpose,
            action=action,
            run=run,
            channel=channel,
        )

    def canonical_bytes(self) -> bytes:
        material = {
            "acquisition_evidence_digest": self.acquisition_evidence_digest,
            "action": self.action,
            "block": self.block,
            "campaign": self.campaign,
            "channel": self.channel,
            "domain": PROSPECTIVE_RANDOM_STREAM_DOMAIN,
            "draw_index": self.draw_index,
            "generator_model": self.generator_model,
            "partition": self.partition,
            "protocol_version": self.protocol_version,
            "purpose": self.purpose,
            "run": self.run,
        }
        return json.dumps(
            material,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


def seed_from_prospective_key(key: ProspectiveRandomStreamKey) -> int:
    """Render a semantic key as a full-width deterministic SHA-256 seed."""

    if not isinstance(key, ProspectiveRandomStreamKey):
        raise TypeError("key must be a prospective random-stream key")
    return int.from_bytes(hashlib.sha256(key.canonical_bytes()).digest(), "big")


@dataclass(frozen=True)
class ProspectiveRandomStream:
    key: ProspectiveRandomStreamKey

    def __post_init__(self) -> None:
        if not isinstance(self.key, ProspectiveRandomStreamKey):
            raise TypeError("prospective stream needs a prospective key")

    @property
    def seed(self) -> int:
        return seed_from_prospective_key(self.key)

    def new_generator(self) -> random.Random:
        """Return a fresh local generator independent of global random state."""

        return random.Random(self.seed)


def prospective_parameter_stream(
    namespace: ProspectiveRandomStreamNamespace,
    *,
    generator_model: str,
    draw_index: int,
) -> ProspectiveRandomStream:
    return ProspectiveRandomStream(
        ProspectiveRandomStreamKey.from_namespace(
            namespace,
            generator_model=generator_model,
            draw_index=draw_index,
            purpose=PARAMETER_DRAW,
        )
    )


def prospective_probe_stream(
    namespace: ProspectiveRandomStreamNamespace,
    *,
    generator_model: str,
    draw_index: int,
) -> ProspectiveRandomStream:
    return ProspectiveRandomStream(
        ProspectiveRandomStreamKey.from_namespace(
            namespace,
            generator_model=generator_model,
            draw_index=draw_index,
            purpose=PROBE_DRAW,
            action=FIXED_FACE_TEMPERATURE,
        )
    )


def prospective_observation_stream(
    namespace: ProspectiveRandomStreamNamespace,
    *,
    generator_model: str,
    draw_index: int,
    purpose: str,
    action: str,
    run: str,
    channel: str,
) -> ProspectiveRandomStream:
    if purpose not in (RUN_BIAS, WHITE_NOISE):
        raise ValueError("prospective observation purpose must be bias or white noise")
    return ProspectiveRandomStream(
        ProspectiveRandomStreamKey.from_namespace(
            namespace,
            generator_model=generator_model,
            draw_index=draw_index,
            purpose=purpose,
            action=action,
            run=run,
            channel=channel,
        )
    )


@dataclass(frozen=True)
class ProspectiveStreamUse:
    stream: ProspectiveRandomStream
    consumer: str
    shared_for_action: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.stream, ProspectiveRandomStream):
            raise TypeError("stream use needs a prospective random stream")
        _validate_label("consumer", self.consumer)
        if self.shared_for_action is not None:
            if self.stream.key.purpose != PARAMETER_DRAW:
                raise ValueError("only parameter draws may declare action sharing")
            if self.shared_for_action not in PROSPECTIVE_ACQUISITION_ACTIONS:
                raise ValueError("parameter sharing needs a prospective action")


@dataclass(frozen=True)
class DeclaredParameterSharing:
    key: ProspectiveRandomStreamKey
    actions: Tuple[str, ...]
    consumers: Tuple[str, ...]


@dataclass(frozen=True)
class UnintendedProspectiveStreamReuse:
    seed: int
    keys: Tuple[ProspectiveRandomStreamKey, ...]
    consumers: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ProspectiveRandomStreamAudit:
    use_count: int
    unique_key_count: int
    unique_seed_count: int
    declared_parameter_sharing: Tuple[DeclaredParameterSharing, ...]
    unintended_reuse: Tuple[UnintendedProspectiveStreamReuse, ...]

    @property
    def clean(self) -> bool:
        return not self.unintended_reuse

    def assert_clean(self) -> None:
        if self.unintended_reuse:
            reasons = ", ".join(
                sorted({item.reason for item in self.unintended_reuse})
            )
            raise ValueError(f"prospective random-stream audit failed: {reasons}")


class ProspectiveRandomStreamRegistry:
    """Collect stream consumers and audit reuse before evidence is accepted."""

    def __init__(self) -> None:
        self._uses = []

    @property
    def uses(self) -> Tuple[ProspectiveStreamUse, ...]:
        return tuple(self._uses)

    def register(
        self,
        stream: ProspectiveRandomStream,
        *,
        consumer: str,
        shared_for_action: Optional[str] = None,
    ) -> None:
        self._uses.append(
            ProspectiveStreamUse(
                stream=stream,
                consumer=consumer,
                shared_for_action=shared_for_action,
            )
        )

    def audit(self) -> ProspectiveRandomStreamAudit:
        uses = tuple(self._uses)
        by_key = {}
        by_seed = {}
        for use in uses:
            by_key.setdefault(use.stream.key, []).append(use)
            by_seed.setdefault(use.stream.seed, []).append(use)

        declared = []
        unintended = []
        for key in sorted(by_key, key=lambda item: item.canonical_bytes()):
            selected = tuple(by_key[key])
            if len(selected) <= 1:
                continue
            actions = tuple(item.shared_for_action for item in selected)
            valid_parameter_sharing = (
                key.purpose == PARAMETER_DRAW
                and all(action in PROSPECTIVE_ACQUISITION_ACTIONS for action in actions)
                and len(set(actions)) == len(actions)
            )
            if valid_parameter_sharing:
                declared.append(
                    DeclaredParameterSharing(
                        key=key,
                        actions=tuple(
                            sorted(
                                actions,  # type: ignore[arg-type]
                                key=lambda action: _ACTION_INDEX[action],
                            )
                        ),
                        consumers=tuple(sorted(item.consumer for item in selected)),
                    )
                )
            else:
                unintended.append(
                    UnintendedProspectiveStreamReuse(
                        seed=selected[0].stream.seed,
                        keys=(key,),
                        consumers=tuple(sorted(item.consumer for item in selected)),
                        reason="undeclared_exact_key_reuse",
                    )
                )

        for seed in sorted(by_seed):
            selected = tuple(by_seed[seed])
            keys = tuple(
                sorted(
                    {item.stream.key for item in selected},
                    key=lambda item: item.canonical_bytes(),
                )
            )
            if len(keys) <= 1:
                continue
            unintended.append(
                UnintendedProspectiveStreamReuse(
                    seed=seed,
                    keys=keys,
                    consumers=tuple(sorted(item.consumer for item in selected)),
                    reason="derived_seed_collision",
                )
            )

        unintended.sort(
            key=lambda item: (
                item.seed,
                item.reason,
                tuple(key.canonical_bytes() for key in item.keys),
            )
        )
        return ProspectiveRandomStreamAudit(
            use_count=len(uses),
            unique_key_count=len(by_key),
            unique_seed_count=len(by_seed),
            declared_parameter_sharing=tuple(declared),
            unintended_reuse=tuple(unintended),
        )


__all__ = [
    "DeclaredParameterSharing",
    "PARAMETER_DRAW",
    "PROBE_DRAW",
    "PROSPECTIVE_ACQUISITION_ACTIONS",
    "PROSPECTIVE_RANDOM_STREAM_DOMAIN",
    "PROSPECTIVE_RANDOM_STREAM_PROTOCOL_VERSION",
    "PROSPECTIVE_STREAM_PURPOSES",
    "ProspectiveRandomStream",
    "ProspectiveRandomStreamAudit",
    "ProspectiveRandomStreamKey",
    "ProspectiveRandomStreamNamespace",
    "ProspectiveRandomStreamRegistry",
    "ProspectiveStreamUse",
    "RUN_BIAS",
    "UnintendedProspectiveStreamReuse",
    "WHITE_NOISE",
    "prospective_observation_stream",
    "prospective_parameter_stream",
    "prospective_probe_stream",
    "seed_from_prospective_key",
]
