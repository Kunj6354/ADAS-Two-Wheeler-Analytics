"""Matched-row research feature ablations; no ML authorization."""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from adas.features.extraction import FEATURE_NAMES
from adas.features.angle_rate import RATE_FEATURE_NAMES


ALL_FEATURES = tuple(FEATURE_NAMES) + tuple(RATE_FEATURE_NAMES)

CLASS_NAMES = (
    "normal_driving",
    "rash_driving",
    "normal_turn",
    "rash_turn",
)

VARIANT_COLUMNS = {
    "accelerometer_only": tuple(range(15)),
    "accelerometer_plus_rates":
        tuple(range(15)) + tuple(range(30, 39)),
    "statistical_baseline": tuple(range(30)),
    "combined_39": tuple(range(39)),
}


@dataclass(frozen=True)
class FeatureVariant:
    name: str
    feature_names: tuple[str, ...]
    identities: tuple[tuple, ...]
    groups: tuple[str, ...]
    labels: tuple[str, ...]
    values: np.ndarray


def build_matched_ablations(
    values,
    identities,
    groups,
    labels,
    *,
    feature_names=ALL_FEATURES,
):
    """Produce matched research-only views from one validated inventory."""

    names = tuple(feature_names)
    identities = tuple(identities)
    groups = tuple(groups)
    labels = tuple(labels)

    if names != ALL_FEATURES:
        raise ValueError("Feature name or ordering mismatch")

    matrix = np.asarray(values, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[1] != 39:
        raise ValueError("Expected 39-column feature matrix")

    if matrix.shape[0] == 0:
        raise ValueError("Empty feature matrix")

    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite feature values")

    count = matrix.shape[0]

    if not (
        len(identities) == len(groups) == len(labels) == count
    ):
        raise ValueError("Row metadata length mismatch")

    if any(
        not isinstance(identity, tuple)
        or len(identity) != 4
        or not isinstance(identity[0], str)
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in identity[1:]
        )
        for identity in identities
    ):
        raise ValueError("Invalid canonical window identity")

    if len(set(identities)) != count:
        raise ValueError("Duplicate window identity")

    if any(
        not isinstance(group, str) or not group
        for group in groups
    ):
        raise ValueError("Invalid recording group")

    if any(
        identity[0] != group
        for identity, group in zip(identities, groups)
    ):
        raise ValueError("Identity and recording-group mismatch")

    if any(label not in CLASS_NAMES for label in labels):
        raise ValueError("Unexpected behavior class")

    result = {}

    for variant_name, columns in VARIANT_COLUMNS.items():
        selected = np.array(matrix[:, columns], copy=True)
        selected.setflags(write=False)

        result[variant_name] = FeatureVariant(
            name=variant_name,
            feature_names=tuple(names[index] for index in columns),
            identities=identities,
            groups=groups,
            labels=labels,
            values=selected,
        )

    return result


def validate_matched_ablations(variants: Mapping):
    """Check identity, metadata and feature-selection consistency."""

    if set(variants) != set(VARIANT_COLUMNS):
        raise ValueError("Variant coverage mismatch")

    reference = variants["combined_39"]

    if reference.values.ndim != 2:
        raise ValueError("Invalid combined matrix")

    row_count = reference.values.shape[0]

    if not (
        len(reference.identities)
        == len(reference.groups)
        == len(reference.labels)
        == row_count
    ):
        raise ValueError("Invalid reference metadata")

    if len(set(reference.identities)) != row_count:
        raise ValueError("Duplicate reference identities")

    if reference.feature_names != ALL_FEATURES:
        raise ValueError("Invalid combined feature names")

    if reference.values.shape[1] != 39:
        raise ValueError("Invalid combined feature dimensions")

    if not np.isfinite(reference.values).all():
        raise ValueError("Invalid combined feature values")

    for name, columns in VARIANT_COLUMNS.items():
        variant = variants[name]

        if variant.name != name:
            raise ValueError("Variant name mismatch")

        if (
            variant.identities != reference.identities
            or variant.groups != reference.groups
            or variant.labels != reference.labels
        ):
            raise ValueError("Unmatched row identity or metadata")

        expected_names = tuple(
            ALL_FEATURES[index] for index in columns
        )

        if variant.feature_names != expected_names:
            raise ValueError("Feature order mismatch")

        if variant.values.shape != (row_count, len(columns)):
            raise ValueError("Variant dimensions mismatch")

        if not np.isfinite(variant.values).all():
            raise ValueError("Nonfinite variant values")

        if not np.array_equal(
            variant.values,
            reference.values[:, columns],
        ):
            raise ValueError("Variant feature values mismatch")

    return {
        "valid": True,
        "row_count": row_count,
        "variant_dimensions": {
            name: len(columns)
            for name, columns in VARIANT_COLUMNS.items()
        },
    }
