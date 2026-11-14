"""M06 draft reference-record validation. No ground-truth approval."""

from dataclasses import dataclass
import re


CLASSES = frozenset({
    "normal_driving", "rash_driving",
    "normal_turn", "rash_turn",
})

UNCERTAINTY = frozenset({
    "INSUFFICIENT_EVIDENCE",
    "UNSYNCHRONIZED_REFERENCE",
    "UNCLEAR_EVENT_BOUNDARY",
    "UNDEFINED_BEHAVIOR_CRITERIA",
    "REVIEWER_DISAGREEMENT",
    "OTHER_DOCUMENTED_REASON",
})

REQUIRED = frozenset({
    "recording_id",
    "annotation_policy_version",
    "reference_source_id",
    "reference_source_sha256",
    "timebase_id",
    "event_start_ms",
    "event_end_ms",
    "reviewer_id",
    "observed_maneuver",
    "reviewer_decision",
    "uncertainty_reason",
    "provenance_notes",
})

SHA256 = re.compile("[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]
    ground_truth_confirmed: bool = False


def validate_review(record):
    """Validate one draft reviewer record without modifying it."""
    errors = []

    if not isinstance(record, dict):
        return ValidationResult(False, ("NOT_A_MAPPING",))

    for field in sorted(REQUIRED - record.keys()):
        errors.append(f"MISSING_{field.upper()}")

    if errors:
        return ValidationResult(False, tuple(errors))

    for field in (
        "recording_id", "annotation_policy_version",
        "reference_source_id", "timebase_id", "reviewer_id"
    ):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"INVALID_{field.upper()}")

    digest = record["reference_source_sha256"]
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        errors.append("INVALID_REFERENCE_SHA256")

    start = record["event_start_ms"]
    end = record["event_end_ms"]

    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, int) or not isinstance(end, int)
        or start < 0 or end <= start
    ):
        errors.append("INVALID_EVENT_INTERVAL")

    decision = record["reviewer_decision"]
    if decision not in ("SUPPORTED", "UNRESOLVED"):
        errors.append("INVALID_REVIEWER_DECISION")

    maneuver = record["observed_maneuver"]
    if maneuver is not None and (not isinstance(maneuver, str) or maneuver not in CLASSES):
        errors.append("INVALID_MANEUVER")

    reason = record["uncertainty_reason"]
    if reason is not None and (not isinstance(reason, str) or reason not in UNCERTAINTY):
        errors.append("INVALID_UNCERTAINTY_REASON")

    if decision == "SUPPORTED" and maneuver is None:
        errors.append("SUPPORTED_REQUIRES_MANEUVER")

    if decision == "UNRESOLVED" and reason is None:
        errors.append("UNRESOLVED_REQUIRES_REASON")

    if not isinstance(record["provenance_notes"], str):
        errors.append("INVALID_PROVENANCE_NOTES")

    return ValidationResult(
        valid=not errors,
        errors=tuple(errors),
        ground_truth_confirmed=False,
    )


def validate_review_pair(first, second):
    """Check reviewer independence and comparable event references."""
    a = validate_review(first)
    b = validate_review(second)
    errors = list(a.errors) + list(b.errors)

    if not (a.valid and b.valid):
        return ValidationResult(False, tuple(errors))

    if first["reviewer_id"] == second["reviewer_id"]:
        errors.append("REVIEWERS_NOT_DISTINCT")

    for field in (
        "recording_id", "annotation_policy_version",
        "reference_source_id", "reference_source_sha256",
        "timebase_id", "event_start_ms", "event_end_ms"
    ):
        if first[field] != second[field]:
            errors.append(f"REFERENCE_MISMATCH_{field.upper()}")

    if (
        first["reviewer_decision"] != second["reviewer_decision"]
        or first["observed_maneuver"] != second["observed_maneuver"]
    ):
        errors.append("REVIEWER_DISAGREEMENT")

    return ValidationResult(
        valid=not errors,
        errors=tuple(errors),
        ground_truth_confirmed=False,
    )
