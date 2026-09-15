"""Machine-readable diagnostics for corrected operating-decision evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .operating_decision_random_streams import (
    RANDOM_STREAM_PROTOCOL_VERSION,
    RandomStreamAudit,
    RandomStreamKey,
    StreamUse,
)
from .operating_decision_replication import (
    CORRECTED_GENERATOR_VERSION,
    CorrectedPartitionResult,
    CorrectedTrialRecord,
)


CORRECTED_DIAGNOSTIC_SCHEMA_VERSION = 1


def _json_safe(value):
    """Encode expected nonfinite failure markers without nonstandard JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0.0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {"nonfinite_float": label}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _current_payload(current) -> dict:
    return {
        "transition_times_seconds": list(current.transition_times),
        "values_amperes": list(current.values),
    }


def _instrumentation_payload(instrumentation) -> dict:
    return {"temporary_face_sensor": instrumentation.temporary_face_sensor}


def _regime_payload(regime) -> dict:
    return {
        "name": regime.name,
        "phase": regime.phase,
        "channels": list(regime.channels),
        "current": _current_payload(regime.current),
    }


def _run_payload(run) -> dict:
    return {
        "regime": _regime_payload(run.regime),
        "instrumentation": _instrumentation_payload(run.instrumentation),
        "observations": [
            {
                "channel": item.channel,
                "time_seconds": item.time,
                "value": item.value,
            }
            for item in run.observations.values
        ],
    }


def _fit_payload(fit) -> dict:
    face_sensor = fit.face_sensor
    return {
        "model_name": fit.model_name,
        "parameter_names": list(fit.parameter_names),
        "log_multipliers": list(fit.log_multipliers),
        "physical_values": list(fit.physical_values),
        "interface_mass": fit.interface_mass,
        "series_resistance": fit.series_resistance,
        "temporary_face_sensor": (
            None
            if face_sensor is None
            else {
                "thermal_capacitance": face_sensor.thermal_capacitance,
                "response_time_constant": face_sensor.response_time_constant,
            }
        ),
        "objective": fit.objective,
        "covariance": [list(row) for row in fit.covariance],
        "reached_bound": fit.reached_bound,
        "evaluation_count": fit.evaluation_count,
        "optimizer": {
            "converged": fit.converged,
            "termination_reason": fit.termination_reason,
            "completed_iterations": fit.completed_iterations,
            "accepted_iterations": fit.accepted_iterations,
            "scaled_gradient_infinity_norm": fit.scaled_gradient_infinity_norm,
            "scaled_projected_gradient_infinity_norm": (
                fit.scaled_gradient_infinity_norm
            ),
            "last_step_infinity_norm": fit.last_step_infinity_norm,
            "last_relative_objective_reduction": (
                fit.last_relative_objective_reduction
            ),
        },
    }


def corrected_trial_payload(record: CorrectedTrialRecord) -> dict:
    """Serialize every retained value needed to audit one final decision."""

    if not isinstance(record, CorrectedTrialRecord):
        raise TypeError("record must be a CorrectedTrialRecord")
    saved = record.scored.saved
    envelope = saved.margin_envelope
    truth = record.truth
    return _json_safe({
        "identity": {
            "campaign": record.partition.campaign,
            "partition": record.partition.name,
            "block": record.block,
            "truth_condition": record.truth_condition,
            "device_token": record.case.case_id.device_token,
            "policy": record.case.policy.name,
            "final_schedule_id": record.case.case_id.final_schedule_id,
        },
        "truth_after_reveal": {
            "physical_values": list(truth.physical_values),
            "interface_mass": truth.interface_mass,
            "series_resistance": truth.series_resistance,
            "temporary_face_sensor": {
                "thermal_capacitance": truth.face_sensor.thermal_capacitance,
                "response_time_constant": truth.face_sensor.response_time_constant,
            },
            "contact_beta": truth.contact_beta,
            "final_cold_face_temperature": list(
                record.revealed.cold_face_temperature
            ),
            "true_margin": record.revealed.true_margin,
            "true_pass": record.scored.true_pass,
        },
        "acquisition_runs": [_run_payload(run) for run in record.case.acquisition_runs],
        "verification_run": _run_payload(record.case.verification_run),
        "final_regime": _regime_payload(record.case.final_regime),
        "saved_decision": {
            "decision": saved.decision,
            "reason": saved.decision_reason,
            "margin_envelope": (
                None
                if envelope is None
                else {"lower": envelope.lower, "upper": envelope.upper}
            ),
            "model_intervals": [item._asdict() for item in saved.model_intervals],
            "candidate_verifications": [
                {
                    "fit": _fit_payload(item.fit),
                    "normalized_score": item.normalized_score,
                    "passed": item.passed,
                    "failure_reason": item.failure_reason,
                }
                for item in saved.verifications
            ],
            "failures": [item._asdict() for item in saved.failures],
            "decision_computation_seconds": saved.decision_computation_seconds,
        },
        "errors": {
            "false_approval": record.scored.false_approval,
            "false_rejection": record.scored.false_rejection,
            "interval_covered": record.scored.interval_covered,
        },
        "resources": {
            "acquisition_run_count": record.scored.acquisition_run_count,
            "diagnostic_run_count": record.scored.diagnostic_run_count,
            "energized_schedule_duration_seconds": (
                record.scored.energized_schedule_time_seconds
            ),
            "extra_sensor_count": record.scored.extra_sensor_count,
            "nominal_selection_energy_proxy_joules": (
                record.nominal_selection_energy
            ),
            "realized_acquisition_terminal_energy_joules": (
                record.realized_energy.acquisition_energy
            ),
            "realized_verification_terminal_energy_joules": (
                record.realized_energy.verification_energy
            ),
            "realized_total_terminal_energy_joules": (
                record.realized_energy.total_diagnostic_energy
            ),
            "realized_runs": [
                {
                    "name": item.regime.name,
                    "phase": item.regime.phase,
                    "terminal_energy_joules": item.terminal_energy,
                    "instrumentation": _instrumentation_payload(item.instrumentation),
                }
                for item in (
                    *record.realized_energy.acquisition_runs,
                    record.realized_energy.verification_run,
                )
            ],
        },
    })


def _stream_key_payload(key: RandomStreamKey) -> dict:
    return {
        "protocol_version": key.protocol_version,
        "campaign": key.campaign,
        "partition": key.partition,
        "block": key.block,
        "stream_kind": key.stream_kind,
        "purpose": key.purpose,
        "family": key.family,
        "run": key.run,
        "channel": key.channel,
    }


def _stream_use_payload(use: StreamUse) -> dict:
    return {
        "key": _stream_key_payload(use.stream.key),
        "seed_sha256_integer_hex": f"{use.stream.seed:064x}",
        "consumer": use.consumer,
        "pairing_member": use.pairing_member,
        "pairing_id": use.pairing_id,
    }


def _stream_audit_payload(audit: RandomStreamAudit) -> dict:
    return {
        "ok": audit.ok,
        "use_count": audit.use_count,
        "unique_key_count": audit.unique_key_count,
        "unique_seed_count": audit.unique_seed_count,
        "declared_pairings": [
            {
                "key": _stream_key_payload(item.key),
                "pairing_id": item.pairing_id,
                "members": list(item.members),
                "consumers": list(item.consumers),
            }
            for item in audit.declared_pairings
        ],
        "unintended_reuses": [
            {
                "seed_sha256_integer_hex": f"{item.seed:064x}",
                "keys": [_stream_key_payload(key) for key in item.keys],
                "consumers": list(item.consumers),
                "reason": item.reason,
            }
            for item in audit.unintended_reuses
        ],
    }


def corrected_partition_diagnostic_payload(
    result: CorrectedPartitionResult,
) -> dict:
    """Serialize a complete partition independently of any visualization."""

    if not isinstance(result, CorrectedPartitionResult):
        raise TypeError("result must be a CorrectedPartitionResult")
    result.random_stream_audit.assert_clean()
    payload = {
        "schema_version": CORRECTED_DIAGNOSTIC_SCHEMA_VERSION,
        "generator_version": CORRECTED_GENERATOR_VERSION,
        "random_stream_protocol_version": RANDOM_STREAM_PROTOCOL_VERSION,
        "partition": {
            "campaign": result.partition.campaign,
            "name": result.partition.name,
            "paired_blocks_per_family": result.partition.block_count,
        },
        "stream_manifest": [
            _stream_use_payload(item) for item in result.random_stream_uses
        ],
        "stream_audit": _stream_audit_payload(result.random_stream_audit),
        "records": [corrected_trial_payload(item) for item in result.records],
        "measurement_boundaries": {
            "energized_schedule_duration_excludes_resets": True,
            "terminal_energy_excludes_sensor_electronics_and_resets": True,
            "decision_computation_time_is_not_a_complete_resource_metric": True,
        },
    }
    payload = _json_safe(payload)
    evidence_payload = json.loads(json.dumps(payload, allow_nan=False))
    for record in evidence_payload["records"]:
        record["saved_decision"].pop("decision_computation_seconds")
    payload["deterministic_evidence_digest"] = hashlib.sha256(
        json.dumps(
            evidence_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def corrected_partition_evidence_digest(
    result: CorrectedPartitionResult,
) -> str:
    """Hash deterministic observations, truth, fits, resources, and streams."""

    return corrected_partition_diagnostic_payload(result)[
        "deterministic_evidence_digest"
    ]


def save_diagnostic_payload(payload: Mapping[str, Any], path: Path | str) -> Path:
    """Save diagnostics once, refusing overwrite and nonfinite JSON values."""

    if not isinstance(payload, Mapping):
        raise TypeError("diagnostic payload must be a mapping")
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return destination


def save_corrected_partition_diagnostics(
    result: CorrectedPartitionResult,
    path: Path | str,
) -> Path:
    return save_diagnostic_payload(corrected_partition_diagnostic_payload(result), path)


__all__ = [
    "CORRECTED_DIAGNOSTIC_SCHEMA_VERSION",
    "corrected_partition_diagnostic_payload",
    "corrected_partition_evidence_digest",
    "corrected_trial_payload",
    "save_corrected_partition_diagnostics",
    "save_diagnostic_payload",
]
