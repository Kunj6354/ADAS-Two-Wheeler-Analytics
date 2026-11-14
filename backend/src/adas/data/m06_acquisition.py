"""Draft M06 session metadata checks; not collection authorization."""

from collections.abc import Mapping
from dataclasses import dataclass
import math


REQUIRED_FIELDS = frozenset({
    "session_id",
    "sensor_identity",
    "firmware_sha256",
    "mounting_record",
    "calibration_record",
    "timebase_record",
    "reference_evidence",
    "annotation_policy_version",
    "collection_approval_reference",
})

TEXT_FIELDS = frozenset({
    "session_id",
    "annotation_policy_version",
    "collection_approval_reference",
})

DETAIL_FIELDS = frozenset({
    "sensor_identity",
    "mounting_record",
    "calibration_record",
    "timebase_record",
    "reference_evidence",
})


@dataclass(frozen=True)
class AcquisitionMetadataCheck:
    structurally_complete: bool
    errors: tuple[str, ...]
    acquisition_authorized: bool = False
    ground_truth_verified: bool = False


def _valid_detail(value, *, key="", depth=0):
    """Recursively check shape only; never certify evidence."""
    if depth > 16:
        return False

    if isinstance(value, Mapping):
        return bool(value) and all(
            isinstance(k, str)
            and bool(k.strip())
            and _valid_detail(v, key=k, depth=depth + 1)
            for k, v in value.items()
        )

    if isinstance(value, (list, tuple)):
        return bool(value) and all(
            _valid_detail(v, depth=depth + 1) for v in value
        )

    if isinstance(value, bool) or value is None:
        return False

    if isinstance(value, str):
        if not value.strip():
            return False
        if key == "sha256":
            return (
                len(value) == 64
                and all(c in "0123456789abcdefABCDEF" for c in value)
            )
        return True

    if key == "sha256":
        return False

    if type(value) in (int, float):
        return math.isfinite(value)

    return False


def validate_session_metadata(record):
    """Validate draft metadata without resolving evidence or approvals."""
    if not isinstance(record, Mapping):
        return AcquisitionMetadataCheck(
            False, ("NOT_A_MAPPING",)
        )

    errors = [
        f"MISSING_{name.upper()}"
        for name in sorted(REQUIRED_FIELDS - record.keys())
    ]

    for name in sorted(TEXT_FIELDS & record.keys()):
        value = record[name]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"INVALID_{name.upper()}")

    if "firmware_sha256" in record:
        value = record["firmware_sha256"]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in value)
        ):
            errors.append("INVALID_FIRMWARE_SHA256")

    for name in sorted(DETAIL_FIELDS & record.keys()):
        value = record[name]
        if not isinstance(value, Mapping) or not _valid_detail(value):
            errors.append(f"INCOMPLETE_{name.upper()}")

    return AcquisitionMetadataCheck(
        structurally_complete=not errors,
        errors=tuple(errors),
        acquisition_authorized=False,
        ground_truth_verified=False,
    )
