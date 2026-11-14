"""Strict, in-memory validation of draft grouped-fold manifests."""

from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
import json
import re

from adas.evaluation.grouped_folds import (
    FoldLimits,
    validate_grouped_folds,
)

SCHEMA_VERSION = "adas.grouped-fold-manifest.v1"

REQUIRED_FIELDS = frozenset({
    "schema_version",
    "approval_status",
    "dataset_manifest_sha256",
    "component_sha256",
    "class_names",
    "fold_limits",
    "folds",
    "canonical_partition_sha256",
    "recording_group_count",
    "eligible_window_count",
})


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def canonical_partition_sha256(folds):
    """Hash fold membership independently of fold labels and ordering."""
    if not isinstance(folds, Mapping) or not folds:
        raise ValueError("Invalid folds")

    normalized = []

    for name, groups in folds.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid fold identifier")

        if isinstance(groups, (str, bytes)):
            raise ValueError("Invalid fold members")

        try:
            values = tuple(groups)
        except TypeError as error:
            raise ValueError("Invalid fold members") from error

        if any(
            not isinstance(item, str) or not item.strip()
            for item in values
        ):
            raise ValueError("Invalid recording identifier")

        normalized.append(tuple(sorted(values)))

    canonical = sorted(normalized)
    payload = json.dumps(
        canonical,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return sha256(payload).hexdigest()


def validate_fold_manifest(
    manifest,
    recording_counts,
    classes,
    *,
    expected_dataset_sha256,
    expected_component_sha256,
    limits=FoldLimits(),
):
    """Validate a DRAFT manifest against separately trusted inputs.

    Does not approve a protocol or verify rider/session independence.
    Does not read or write files.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("Manifest must be a mapping")

    if set(manifest) != REQUIRED_FIELDS:
        raise ValueError("Missing or unexpected manifest fields")

    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema version")

    if manifest["approval_status"] != "DRAFT":
        raise ValueError("Only draft manifests supported")

    supplied_dataset_hash = manifest["dataset_manifest_sha256"]

    if not _valid_sha256(expected_dataset_sha256):
        raise ValueError("Invalid trusted dataset fingerprint")

    if not _valid_sha256(supplied_dataset_hash):
        raise ValueError("Invalid dataset fingerprint")

    if supplied_dataset_hash != expected_dataset_sha256:
        raise ValueError("Dataset fingerprint mismatch")

    supplied_components = manifest["component_sha256"]

    if (
        not isinstance(supplied_components, Mapping)
        or not isinstance(expected_component_sha256, Mapping)
        or not expected_component_sha256
        or set(supplied_components) != set(expected_component_sha256)
    ):
        raise ValueError("Component fingerprint inventory mismatch")

    for name, expected_hash in expected_component_sha256.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid component identifier")

        supplied_hash = supplied_components[name]

        if not _valid_sha256(expected_hash):
            raise ValueError("Invalid trusted component fingerprint")

        if not _valid_sha256(supplied_hash):
            raise ValueError("Invalid component fingerprint")

        if supplied_hash != expected_hash:
            raise ValueError("Component fingerprint mismatch")

    if (
        not isinstance(manifest["class_names"], (tuple, list))
        or list(manifest["class_names"]) != list(classes)
    ):
        raise ValueError("Class order mismatch")

    if (
        not isinstance(manifest["fold_limits"], Mapping)
        or dict(manifest["fold_limits"]) != asdict(limits)
    ):
        raise ValueError("Fold-limit mismatch")

    folds = manifest["folds"]

    if not isinstance(folds, Mapping):
        raise ValueError("Fold assignment must be a mapping")

    if (
        not _valid_sha256(manifest["canonical_partition_sha256"])
        or manifest["canonical_partition_sha256"]
        != canonical_partition_sha256(folds)
    ):
        raise ValueError("Partition fingerprint mismatch")

    def exact_count(value, expected):
        return (
            type(value) is int
            and value == expected
        )

    if not exact_count(
        manifest["recording_group_count"],
        len(recording_counts),
    ):
        raise ValueError("Recording-group count mismatch")

    expected_windows = sum(
        sum(values.values())
        for values in recording_counts.values()
    )

    if not exact_count(
        manifest["eligible_window_count"],
        expected_windows,
    ):
        raise ValueError("Eligible-window count mismatch")

    quality = validate_grouped_folds(
        folds, recording_counts, classes, limits
    )

    if not quality["eligible"]:
        raise ValueError(
            "Fold quality violations: "
            + ", ".join(quality["violations"])
        )

    return {
        "valid": True,
        "approval_status": "DRAFT",
        "recording_group_count": len(recording_counts),
        "eligible_window_count": expected_windows,
        "fold_count": len(folds),
        "canonical_partition_sha256":
            manifest["canonical_partition_sha256"],
    }
