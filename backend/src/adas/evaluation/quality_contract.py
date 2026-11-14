"""Exploratory grouped-partition quality eligibility.

No partition generation, ranking, model fitting or file I/O.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from adas.evaluation.partition_contract import (
    validate_partition_class_coverage,
)


@dataclass(frozen=True)
class QualityLimits:
    train_min_share: float = 0.50
    evaluation_min_share: float = 0.10
    evaluation_max_share: float = 0.35
    max_class_concentration: float = 0.80
    minimum_test_windows: int = 10
    minimum_validation_windows: int = 10
    minimum_train_class_groups: int = 3
    minimum_evaluation_class_groups: int = 2


def evaluate_partition_quality(
    partitions: Mapping[str, Sequence[str]],
    recording_counts: Mapping[str, Mapping[str, int]],
    classes: Sequence[str],
    limits: QualityLimits = QualityLimits(),
) -> dict:
    """Return eligibility and diagnostics without selecting a split."""
    from math import isfinite

    if not isinstance(limits, QualityLimits):
        raise ValueError("Invalid limits object")

    fractions = (
        limits.train_min_share,
        limits.evaluation_min_share,
        limits.evaluation_max_share,
        limits.max_class_concentration,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 < value <= 1
        for value in fractions
    ):
        raise ValueError("Invalid fraction limit")

    if limits.evaluation_min_share > limits.evaluation_max_share:
        raise ValueError("Invalid evaluation share interval")

    integer_limits = (
        limits.minimum_test_windows,
        limits.minimum_validation_windows,
        limits.minimum_train_class_groups,
        limits.minimum_evaluation_class_groups,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for value in integer_limits
    ):
        raise ValueError("Invalid integer limit")

    # Structural errors remain errors, rather than ordinary failures.
    class_totals = validate_partition_class_coverage(
        partitions,
        recording_counts,
        classes,
        minimum_test_windows=1,
    )

    total_windows = sum(
        sum(counts.values()) for counts in recording_counts.values()
    )
    diagnostics = {}
    violations = []

    for partition in ("train", "validation", "test"):
        groups = tuple(partitions[partition])
        totals = class_totals[partition]
        windows = sum(totals.values())
        share = windows / total_windows

        support = {
            label: sum(
                recording_counts[group].get(label, 0) > 0
                for group in groups
            )
            for label in classes
        }
        concentration = {
            label: max(
                recording_counts[group].get(label, 0)
                for group in groups
            ) / totals[label]
            for label in classes
        }

        diagnostics[partition] = {
            "windows": windows,
            "share": share,
            "class_counts": dict(totals),
            "class_group_support": support,
            "class_concentration": concentration,
        }

        if partition == "train":
            if share < limits.train_min_share:
                violations.append("train_share")
            minimum_groups = limits.minimum_train_class_groups
        else:
            if not (
                limits.evaluation_min_share
                <= share
                <= limits.evaluation_max_share
            ):
                violations.append(f"{partition}_share")
            minimum_groups = limits.minimum_evaluation_class_groups
            minimum_windows = (
                limits.minimum_validation_windows
                if partition == "validation"
                else limits.minimum_test_windows
            )
            if any(
                totals[label] < minimum_windows for label in classes
            ):
                violations.append(f"{partition}_class_windows")

        if any(
            support[label] < minimum_groups for label in classes
        ):
            violations.append(f"{partition}_class_groups")

        if any(
            concentration[label] > limits.max_class_concentration
            for label in classes
        ):
            violations.append(f"{partition}_class_concentration")

    return {
        "eligible": not violations,
        "violations": tuple(violations),
        "diagnostics": diagnostics,
    }
