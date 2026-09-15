"""Versioned random streams for corrected operating-decision campaigns.

The original operating-decision generator formed seeds by adding numeric offsets.
That made two coordinates collide when a run offset happened to equal a channel
offset.  This module instead hashes a complete, typed stream key.  Comparison
members are deliberately not part of that key: paired families or policies may
reuse the same physical device or observation, but that reuse must be declared
to ``RandomStreamRegistry`` and pass its audit.

The module only allocates deterministic seeds.  A caller remains responsible for
recording the stream key alongside generated evidence and for choosing which
physical quantities a stream drives.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


RANDOM_STREAM_PROTOCOL_VERSION = "thermotwin-operating-decision-rng-v2"
DEVICE_TRUTH_STREAM = "device_truth"
OBSERVATION_STREAM = "observation"
CALIBRATION_STREAM = "calibration"
_STREAM_KEY_DOMAIN = "thermotwin.operating_decision.random_stream"


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


@dataclass(frozen=True)
class RandomStreamKey:
    """The complete identity of one independent pseudorandom stream.

    ``family`` is omitted for device properties intentionally shared across
    model families.  Observation streams always carry a family, run, and
    channel.  Device-truth streams never carry a run or channel.
    """

    protocol_version: str
    campaign: str
    partition: str
    block: int
    stream_kind: str
    purpose: str
    family: Optional[str] = None
    run: Optional[str] = None
    channel: Optional[str] = None

    def __post_init__(self) -> None:
        for name in (
            "protocol_version",
            "campaign",
            "partition",
            "stream_kind",
            "purpose",
        ):
            _validate_label(name, getattr(self, name))
        for name in ("family", "run", "channel"):
            _validate_optional_label(name, getattr(self, name))
        if not isinstance(self.block, int) or isinstance(self.block, bool):
            raise TypeError("block must be an integer")
        if self.block < 0:
            raise ValueError("block must be nonnegative")
        if self.stream_kind in (DEVICE_TRUTH_STREAM, CALIBRATION_STREAM):
            if self.run is not None or self.channel is not None:
                raise ValueError(
                    f"{self.stream_kind} stream keys cannot carry a run or channel"
                )
        elif self.stream_kind == OBSERVATION_STREAM:
            missing = tuple(
                name
                for name in ("family", "run", "channel")
                if getattr(self, name) is None
            )
            if missing:
                raise ValueError(
                    "observation stream keys require " + ", ".join(missing)
                )
        else:
            raise ValueError(
                f"stream_kind must be {DEVICE_TRUTH_STREAM!r} or "
                f"{OBSERVATION_STREAM!r} or {CALIBRATION_STREAM!r}"
            )

    def canonical_bytes(self) -> bytes:
        """Return the protocol-stable representation hashed into the seed."""

        material = {
            "block": self.block,
            "campaign": self.campaign,
            "channel": self.channel,
            "domain": _STREAM_KEY_DOMAIN,
            "family": self.family,
            "partition": self.partition,
            "protocol_version": self.protocol_version,
            "purpose": self.purpose,
            "run": self.run,
            "stream_kind": self.stream_kind,
        }
        return json.dumps(
            material,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


def seed_from_stream_key(key: RandomStreamKey) -> int:
    """Derive a full-width SHA-256 integer seed from ``key``."""

    if not isinstance(key, RandomStreamKey):
        raise TypeError("key must be a RandomStreamKey")
    return int.from_bytes(hashlib.sha256(key.canonical_bytes()).digest(), "big")


@dataclass(frozen=True)
class RandomStream:
    """A reproducible stream descriptor with a fresh-generator factory."""

    key: RandomStreamKey

    def __post_init__(self) -> None:
        if not isinstance(self.key, RandomStreamKey):
            raise TypeError("key must be a RandomStreamKey")

    @property
    def seed(self) -> int:
        return seed_from_stream_key(self.key)

    def new_generator(self) -> random.Random:
        """Return a new generator positioned at the start of this stream."""

        return random.Random(self.seed)


def device_truth_stream(
    *,
    campaign: str,
    partition: str,
    block: int,
    purpose: str,
    family: Optional[str] = None,
    protocol_version: str = RANDOM_STREAM_PROTOCOL_VERSION,
) -> RandomStream:
    """Allocate a device-level truth stream.

    Leave ``family`` as ``None`` for physical properties shared by all truth
    families.  Supply it for a family-specific mechanism such as a contact-law
    coefficient.
    """

    return RandomStream(
        RandomStreamKey(
            protocol_version=protocol_version,
            campaign=campaign,
            partition=partition,
            block=block,
            stream_kind=DEVICE_TRUTH_STREAM,
            purpose=purpose,
            family=family,
        )
    )


def observation_stream(
    *,
    campaign: str,
    partition: str,
    block: int,
    purpose: str,
    family: str,
    run: str,
    channel: str,
    protocol_version: str = RANDOM_STREAM_PROTOCOL_VERSION,
) -> RandomStream:
    """Allocate one family/run/channel observation stream.

    Policy is intentionally absent.  Policies evaluated on the same simulated
    measurement therefore obtain the same key, and the registry requires them
    to declare that common-random-number pairing.
    """

    return RandomStream(
        RandomStreamKey(
            protocol_version=protocol_version,
            campaign=campaign,
            partition=partition,
            block=block,
            stream_kind=OBSERVATION_STREAM,
            purpose=purpose,
            family=family,
            run=run,
            channel=channel,
        )
    )


def calibration_stream(
    *,
    campaign: str,
    partition: str,
    purpose: str,
    block: int = 0,
    family: Optional[str] = None,
    protocol_version: str = RANDOM_STREAM_PROTOCOL_VERSION,
) -> RandomStream:
    """Allocate a calibration, rehearsal, or bootstrap stream."""

    return RandomStream(
        RandomStreamKey(
            protocol_version=protocol_version,
            campaign=campaign,
            partition=partition,
            block=block,
            stream_kind=CALIBRATION_STREAM,
            purpose=purpose,
            family=family,
        )
    )


@dataclass(frozen=True)
class StreamUse:
    """One consumer of a stream, optionally in a declared pairing."""

    stream: RandomStream
    consumer: str
    pairing_member: Optional[str] = None
    pairing_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.stream, RandomStream):
            raise TypeError("stream must be a RandomStream")
        _validate_label("consumer", self.consumer)
        _validate_optional_label("pairing_member", self.pairing_member)
        _validate_optional_label("pairing_id", self.pairing_id)
        if self.pairing_id is not None and self.pairing_member is None:
            raise ValueError("a declared pairing requires a pairing member")


@dataclass(frozen=True)
class DeclaredStreamPairing:
    """A repeated key whose comparison-member sharing was explicitly declared."""

    key: RandomStreamKey
    pairing_id: str
    members: Tuple[str, ...]
    consumers: Tuple[str, ...]


@dataclass(frozen=True)
class UnintendedStreamReuse:
    """An exact key reuse or derived-seed collision that failed the audit."""

    seed: int
    keys: Tuple[RandomStreamKey, ...]
    consumers: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class RandomStreamAudit:
    """Summary of stream uniqueness and declared comparison-member sharing."""

    use_count: int
    unique_key_count: int
    unique_seed_count: int
    declared_pairings: Tuple[DeclaredStreamPairing, ...]
    unintended_reuses: Tuple[UnintendedStreamReuse, ...]

    @property
    def ok(self) -> bool:
        return not self.unintended_reuses

    def assert_clean(self) -> None:
        if not self.ok:
            details = "; ".join(item.reason for item in self.unintended_reuses)
            raise ValueError(f"random-stream audit failed: {details}")


def _key_sort_material(key: RandomStreamKey) -> bytes:
    return key.canonical_bytes()


def audit_stream_uses(uses: Iterable[StreamUse]) -> RandomStreamAudit:
    """Detect unintended seed reuse and validate declared pairings.

    Reusing an identical key is allowed only when every use supplies the same
    ``pairing_id`` and each use names a distinct comparison member. Different
    keys that ever derive the same SHA-256 seed are always reported as a
    collision.
    """

    recorded = tuple(uses)
    if any(not isinstance(item, StreamUse) for item in recorded):
        raise TypeError("uses must contain only StreamUse instances")

    by_seed = {}
    for item in recorded:
        by_seed.setdefault(item.stream.seed, []).append(item)

    pairings = []
    unintended = []
    for seed, group_values in sorted(by_seed.items()):
        group = tuple(group_values)
        if len(group) == 1:
            continue
        keys = tuple(
            sorted({item.stream.key for item in group}, key=_key_sort_material)
        )
        consumers = tuple(item.consumer for item in group)
        if len(keys) != 1:
            unintended.append(
                UnintendedStreamReuse(
                    seed=seed,
                    keys=keys,
                    consumers=consumers,
                    reason=(
                        "different stream keys derived the same seed for "
                        + ", ".join(consumers)
                    ),
                )
            )
            continue

        pairing_ids = {item.pairing_id for item in group}
        members = tuple(item.pairing_member for item in group)
        valid_pairing = (
            None not in pairing_ids
            and len(pairing_ids) == 1
            and None not in members
            and len(set(members)) == len(members)
        )
        if valid_pairing:
            pairing_id = next(iter(pairing_ids))
            pairings.append(
                DeclaredStreamPairing(
                    key=keys[0],
                    pairing_id=pairing_id,  # type: ignore[arg-type]
                    members=members,  # type: ignore[arg-type]
                    consumers=consumers,
                )
            )
            continue

        unintended.append(
            UnintendedStreamReuse(
                seed=seed,
                keys=keys,
                consumers=consumers,
                reason=(
                    "stream key reused without one shared pairing declaration "
                    "across distinct comparison members for " + ", ".join(consumers)
                ),
            )
        )

    return RandomStreamAudit(
        use_count=len(recorded),
        unique_key_count=len({item.stream.key for item in recorded}),
        unique_seed_count=len(by_seed),
        declared_pairings=tuple(pairings),
        unintended_reuses=tuple(unintended),
    )


class RandomStreamRegistry:
    """Collect stream consumers and audit the complete campaign manifest."""

    def __init__(self) -> None:
        self._uses = []

    @property
    def uses(self) -> Tuple[StreamUse, ...]:
        return tuple(self._uses)

    def register(
        self,
        stream: RandomStream,
        *,
        consumer: str,
        pairing_member: Optional[str] = None,
        pairing_id: Optional[str] = None,
    ) -> RandomStream:
        """Record a use and return ``stream`` for convenient inline calls."""

        self._uses.append(
            StreamUse(
                stream=stream,
                consumer=consumer,
                pairing_member=pairing_member,
                pairing_id=pairing_id,
            )
        )
        return stream

    def audit(self) -> RandomStreamAudit:
        return audit_stream_uses(self._uses)
