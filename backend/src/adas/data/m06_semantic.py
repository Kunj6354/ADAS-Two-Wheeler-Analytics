"""Draft contract-driven M06 metadata semantics; no approval authority."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
import hashlib
import json


@dataclass(frozen=True)
class SemanticCheck:
    structurally_valid: bool
    errors: tuple[str, ...]
    scientific_evidence_verified: bool = False
    acquisition_authorized: bool = False
    ground_truth_verified: bool = False


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _valid(value, kind):
    if kind == "text":
        return isinstance(value, str) and bool(value.strip())
    if kind == "sha256":
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdefABCDEF" for c in value)
        )
    if kind == "finite_number":
        return _finite(value)
    if kind == "nonnegative_integer":
        return type(value) is int and value >= 0
    if kind == "object":
        return (
            isinstance(value, Mapping)
            and bool(value)
            and all(
                isinstance(k, str) and bool(k.strip())
                and isinstance(v, str) and bool(v.strip())
                for k, v in value.items()
            )
        )
    if kind in ("finite_number_mapping", "text_mapping"):
        if not isinstance(value, Mapping) or not value:
            return False
        child_kind = (
            "finite_number" if kind == "finite_number_mapping"
            else "text"
        )
        return all(
            isinstance(k, str)
            and bool(k.strip())
            and _valid(v, child_kind)
            for k, v in value.items()
        )
    return False


def validate_semantic_metadata(record, parent, nested):
    """Check proposed field representations, not physical validity."""
    if not isinstance(record, Mapping):
        return SemanticCheck(False, ("NOT_A_MAPPING",))
    if not isinstance(parent, Mapping) or not isinstance(nested, Mapping):
        return SemanticCheck(False, ("INVALID_CONTRACT_INPUT",))

    # Verify exact frozen draft contract revisions.
    # Canonical serialization must match the contract creation format.
    try:
        parent_bytes = (
            json.dumps(parent, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        nested_bytes = (
            json.dumps(nested, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        if (
            hashlib.sha256(parent_bytes).hexdigest()
            != "e3b24ddb592bb16950e49a4e842b5ed5de731960e5f7da089493920c37f08e1c"
            or hashlib.sha256(nested_bytes).hexdigest()
            != "cf75bab89bd673ef50fd112540c9a09cc2db248270a932169ba6d6ce41d9b86c"
        ):
            return SemanticCheck(False, ("CONTRACT_INTEGRITY_MISMATCH",))
    except (TypeError, ValueError, OverflowError, RecursionError):
        return SemanticCheck(False, ("INVALID_CONTRACT_SERIALIZATION",))

    # Fail closed on malformed, inconsistent or unexpectedly approved
    # contract inputs. Caller must separately verify file identity.
    try:
        if (
            parent["status"] != "DRAFT_NOT_APPROVED"
            or nested["status"] != "DRAFT_NOT_APPROVED"
            or nested["parent_contract_id"] != parent["contract_id"]
            or any(v is not False for v in parent["policy"].values())
            or any(v is not False for v in nested["safety_gates"].values())
        ):
            return SemanticCheck(False, ("CONTRACT_AUTHORITY_MISMATCH",))

        fields = parent["field_definitions"]
        shapes = nested["nested_definitions"]

        declared_objects = {
            f"{group}.{field}"
            for group, definitions in fields.items()
            for field, spec in definitions.items()
            if spec["proposed_type"] == "object"
        }
        if declared_objects != set(shapes):
            return SemanticCheck(False, ("CONTRACT_SHAPE_MISMATCH",))

        errors = []

        for group, definitions in fields.items():
            group_value = record.get(group)
            if not isinstance(group_value, Mapping):
                errors.append(f"MISSING_OR_INVALID_GROUP:{group}")
                continue

            for field, specification in definitions.items():
                path = f"{group}.{field}"
                if field not in group_value:
                    errors.append(f"MISSING:{path}")
                    continue

                value = group_value[field]
                kind = specification["proposed_type"]

                if kind == "object":
                    shape = shapes[path]["fields"]
                    if not isinstance(value, Mapping):
                        errors.append(f"INVALID_OBJECT:{path}")
                        continue
                    for subfield, subkind in shape.items():
                        if subfield not in value:
                            errors.append(f"MISSING:{path}.{subfield}")
                        elif not _valid(value[subfield], subkind):
                            errors.append(f"INVALID:{path}.{subfield}")
                elif not _valid(value, kind):
                    errors.append(f"INVALID:{path}")

        # Exact field coverage: unknown fields cannot be silently ignored.
        for group in record:
            if group not in fields:
                errors.append(f"UNKNOWN_GROUP:{group}")

        for group, definitions in fields.items():
            group_value = record.get(group)
            if not isinstance(group_value, Mapping):
                continue

            for field in group_value:
                if field not in definitions:
                    errors.append(f"UNKNOWN_FIELD:{group}.{field}")

            for field, specification in definitions.items():
                if specification["proposed_type"] != "object":
                    continue

                value = group_value.get(field)
                if not isinstance(value, Mapping):
                    continue

                path = f"{group}.{field}"
                allowed_subfields = shapes[path]["fields"]

                for subfield in value:
                    if subfield not in allowed_subfields:
                        errors.append(
                            f"UNKNOWN_NESTED_FIELD:{path}.{subfield}"
                        )

        timing_group = record.get("timing")
        if isinstance(timing_group, Mapping):
            uncertainty = timing_group.get("alignment_uncertainty_ms")
            if _finite(uncertainty) and uncertainty < 0:
                errors.append("NEGATIVE_ALIGNMENT_UNCERTAINTY")

        calibration = record.get("calibration")
        if isinstance(calibration, Mapping):
            offsets = calibration.get("offset_values")
            units = calibration.get("offset_units")
            if isinstance(offsets, Mapping) and isinstance(units, Mapping):
                channels = offsets.get("channel_values")
                channel_units = units.get("channel_units")
                if isinstance(channels, Mapping) and isinstance(
                    channel_units, Mapping
                ):
                    if set(channels) != set(channel_units):
                        errors.append("CALIBRATION_CHANNEL_UNIT_MISMATCH")

        timing = record.get("timing")
        if isinstance(timing, Mapping):
            samples = timing.get("sampling_interval_observations")
            if isinstance(samples, Mapping):
                count = samples.get("interval_count")
                minimum = samples.get("minimum")
                median = samples.get("median")
                maximum = samples.get("maximum")
                if (
                    _valid(count, "nonnegative_integer")
                    and all(_finite(v) for v in (minimum, median, maximum))
                ):
                    if count == 0 or not (
                        0 <= minimum <= median <= maximum
                    ):
                        errors.append("INCOHERENT_SAMPLING_INTERVALS")

        return SemanticCheck(not errors, tuple(errors))
    except (KeyError, TypeError, AttributeError, ValueError):
        return SemanticCheck(False, ("MALFORMED_CONTRACT",))
