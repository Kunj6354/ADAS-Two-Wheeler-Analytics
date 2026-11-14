"""Draft M06 adjudication transition checks. No confirmation authority."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence


REQUIRED = frozenset({
    "recording_id",
    "reference_source_id",
    "reference_source_sha256",
    "annotation_policy_version",
    "event_start_ms",
    "event_end_ms",
    "previous_state",
    "current_state",
    "transition_reason",
    "decision_actor_id",
    "decision_timestamp",
    "audit_event_id",
})

ALLOWED = frozenset({
    ("PENDING", "UNRESOLVED"),
    ("UNRESOLVED", "PENDING"),
})


@dataclass(frozen=True)
class AdjudicationCheck:
    valid: bool
    errors: tuple[str, ...]
    ground_truth_confirmed: bool = False


def validate_transition(record):
    """Check a draft transition without authorizing confirmation."""
    if not isinstance(record, Mapping):
        return AdjudicationCheck(False, ("NOT_A_MAPPING",))

    errors = [
        f"MISSING_{name.upper()}"
        for name in sorted(REQUIRED - record.keys())
    ]
    if errors:
        return AdjudicationCheck(False, tuple(errors))

    for name in (
        "recording_id", "reference_source_id",
        "annotation_policy_version", "transition_reason",
        "decision_actor_id", "decision_timestamp", "audit_event_id"
    ):
        value = record[name]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"INVALID_{name.upper()}")

    reference = record["reference_source_sha256"]
    if (
        not isinstance(reference, str)
        or len(reference) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in reference)
    ):
        errors.append("INVALID_REFERENCE_SHA256")

    start = record["event_start_ms"]
    end = record["event_end_ms"]
    if (
        type(start) is not int or type(end) is not int
        or start < 0 or end <= start
    ):
        errors.append("INVALID_EVENT_INTERVAL")

    previous = record["previous_state"]
    current = record["current_state"]
    if not isinstance(previous, str) or not isinstance(current, str):
        errors.append("INVALID_STATE_TYPE")
    elif previous == "CONFIRMED" or current == "CONFIRMED":
        errors.append("CONFIRMATION_DISABLED")
    elif (previous, current) not in ALLOWED:
        errors.append("TRANSITION_NOT_ALLOWED")

    if previous == "UNRESOLVED" and current == "PENDING":
        evidence = record.get("evidence_or_review_reference")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append("REOPENING_EVIDENCE_REQUIRED")

    return AdjudicationCheck(not errors, tuple(errors))


def validate_history(records):
    """Check one event's ordered, append-only draft transition history."""
    if not isinstance(records, (list, tuple)) or not records:
        return AdjudicationCheck(False, ("EMPTY_OR_INVALID_HISTORY",))

    errors = []
    seen = set()
    previous_record = None
    identity_fields = (
        "recording_id", "reference_source_id",
        "reference_source_sha256", "annotation_policy_version",
        "event_start_ms", "event_end_ms",
    )

    for index, record in enumerate(records):
        result = validate_transition(record)
        errors.extend(f"EVENT_{index}:{e}" for e in result.errors)

        if not isinstance(record, Mapping):
            continue

        event_id = record.get("audit_event_id")
        if isinstance(event_id, str) and event_id.strip():
            if event_id in seen:
                errors.append(f"EVENT_{index}:DUPLICATE_AUDIT_ID")
            seen.add(event_id)

        if index == 0 and record.get("previous_state") != "PENDING":
            errors.append("INVALID_INITIAL_STATE")

        if previous_record is not None:
            if record.get("previous_state") != previous_record.get(
                "current_state"
            ):
                errors.append(f"EVENT_{index}:BROKEN_STATE_CHAIN")

            for field in identity_fields:
                if record.get(field) != previous_record.get(field):
                    errors.append(
                        f"EVENT_{index}:EVENT_IDENTITY_CHANGED_{field.upper()}"
                    )

        previous_record = record

    return AdjudicationCheck(not errors, tuple(errors))
