"""Independent grouped-fold validation for offline research."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class FoldLimits:
    n_folds: int = 4
    groups_per_fold: int = 6
    minimum_class_groups: int = 2
    minimum_class_windows: int = 10
    minimum_window_share: float = 0.10
    maximum_window_share: float = 0.40
    maximum_class_concentration: float = 0.80


def validate_grouped_folds(
    folds: Mapping[str, Sequence[str]],
    recording_counts: Mapping[str, Mapping[str, int]],
    classes: Sequence[str],
    limits: FoldLimits = FoldLimits(),
) -> dict:
    """Validate structural isolation and exploratory fold quality.

    Structural violations raise ValueError.
    Quality failures return eligible=False.
    """
    if not isinstance(limits, FoldLimits):
        raise ValueError("Invalid fold limits")

    integers = (
        limits.n_folds,
        limits.groups_per_fold,
        limits.minimum_class_groups,
        limits.minimum_class_windows,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for value in integers
    ):
        raise ValueError("Invalid integer limit")

    fractions = (
        limits.minimum_window_share,
        limits.maximum_window_share,
        limits.maximum_class_concentration,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 < value <= 1
        for value in fractions
    ):
        raise ValueError("Invalid fraction limit")

    if limits.minimum_window_share > limits.maximum_window_share:
        raise ValueError("Invalid fold share interval")

    if (
        limits.minimum_class_groups > limits.groups_per_fold
        or limits.n_folds * limits.minimum_window_share > 1
    ):
        raise ValueError("Internally inconsistent limits")

    if (
        not classes
        or any(not isinstance(c, str) or not c.strip() for c in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("Invalid classes")

    if not recording_counts:
        raise ValueError("Missing recording counts")

    for group, counts in recording_counts.items():
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Invalid recording identifier")

        if not isinstance(counts, Mapping):
            raise ValueError("Invalid recording counts")

        if set(counts) - set(classes):
            raise ValueError("Unknown class")

        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in counts.values()
        ):
            raise ValueError("Invalid window count")

        if sum(counts.values()) == 0:
            raise ValueError("Empty recording")

    if len(folds) != limits.n_folds:
        raise ValueError("Incorrect number of folds")

    seen = set()
    diagnostics = {}
    violations = []

    total_windows = sum(
        sum(counts.values())
        for counts in recording_counts.values()
    )

    for fold_name, groups in folds.items():
        if not isinstance(fold_name, str) or not fold_name.strip():
            raise ValueError("Invalid fold identifier")

        if isinstance(groups, (str, bytes)):
            raise ValueError("Fold groups must be a sequence")

        if len(groups) == 0:
            raise ValueError("Empty fold")

        if len(groups) != len(set(groups)):
            raise ValueError("Duplicate group within fold")

        for group in groups:
            if group not in recording_counts:
                raise ValueError("Unknown recording group")
            if group in seen:
                raise ValueError("Recording reused across folds")
            seen.add(group)

        totals = Counter()
        for group in groups:
            totals.update(recording_counts[group])

        windows = sum(totals.values())
        share = windows / total_windows

        support = {}
        concentration = {}

        for label in classes:
            values = [
                recording_counts[group].get(label, 0)
                for group in groups
            ]
            contributing = [value for value in values if value > 0]

            support[label] = len(contributing)
            concentration[label] = (
                max(contributing) / sum(contributing)
                if contributing else 1.0
            )

        if len(groups) != limits.groups_per_fold:
            violations.append(f"{fold_name}_group_count")

        if not (
            limits.minimum_window_share
            <= share
            <= limits.maximum_window_share
        ):
            violations.append(f"{fold_name}_window_share")

        if any(
            totals[label] < limits.minimum_class_windows
            for label in classes
        ):
            violations.append(f"{fold_name}_class_windows")

        if any(
            support[label] < limits.minimum_class_groups
            for label in classes
        ):
            violations.append(f"{fold_name}_class_groups")

        if any(
            concentration[label] > limits.maximum_class_concentration
            for label in classes
        ):
            violations.append(f"{fold_name}_class_concentration")

        diagnostics[fold_name] = {
            "group_count": len(groups),
            "windows": windows,
            "share": share,
            "class_counts": dict(totals),
            "class_group_support": support,
            "class_concentration": concentration,
        }

    if seen != set(recording_counts):
        raise ValueError("Missing recording-group assignments")

    return {
        "eligible": not violations,
        "violations": tuple(violations),
        "diagnostics": diagnostics,
    }
