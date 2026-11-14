"""Structural and class-support validation of nested grouped folds."""

from collections.abc import Mapping, Sequence

from adas.evaluation.grouped_folds import (
    FoldLimits,
    validate_grouped_folds,
)


def validate_nested_folds(
    outer_folds: Mapping[str, Sequence[str]],
    inner_by_outer: Mapping[str, Mapping[str, Sequence[str]]],
    recording_counts: Mapping[str, Mapping[str, int]],
    classes: Sequence[str],
    *,
    outer_limits: FoldLimits = FoldLimits(),
    inner_limits: FoldLimits = FoldLimits(
        n_folds=3, groups_per_fold=6
    ),
    minimum_training_class_groups: int = 2,
    minimum_training_class_windows: int = 10,
) -> dict:
    """Validate nested partitions without selecting or training models."""

    for value in (
        minimum_training_class_groups,
        minimum_training_class_windows,
    ):
        if type(value) is not int or value < 1:
            raise ValueError("Invalid inner-training minimum")

    outer_result = validate_grouped_folds(
        outer_folds, recording_counts, classes, outer_limits
    )
    if not outer_result["eligible"]:
        raise ValueError("Outer quality contract failed")

    if not isinstance(inner_by_outer, Mapping):
        raise ValueError("Invalid nested fold mapping")
    if set(inner_by_outer) != set(outer_folds):
        raise ValueError("Nested outer-fold coverage mismatch")

    all_groups = set(recording_counts)
    summary = {}

    for outer_name, held_out in outer_folds.items():
        held_out_set = set(held_out)
        development = all_groups - held_out_set

        inner = inner_by_outer[outer_name]
        if not isinstance(inner, Mapping):
            raise ValueError("Invalid inner fold mapping")

        development_counts = {
            name: recording_counts[name]
            for name in development
        }

        inner_result = validate_grouped_folds(
            inner, development_counts, classes, inner_limits
        )
        if not inner_result["eligible"]:
            raise ValueError("Inner quality contract failed")

        inner_summaries = {}

        for inner_name, validation_groups in inner.items():
            validation_set = set(validation_groups)
            training = development - validation_set

            if validation_set & held_out_set:
                raise ValueError("Outer evaluation leakage")

            if not training:
                raise ValueError("Empty inner training set")

            support = {}
            class_windows = {}

            for label in classes:
                values = [
                    recording_counts[name].get(label, 0)
                    for name in training
                ]
                support[label] = sum(v > 0 for v in values)
                class_windows[label] = sum(values)

                if support[label] < minimum_training_class_groups:
                    raise ValueError(
                        "Insufficient inner-training class groups"
                    )

                if class_windows[label] < minimum_training_class_windows:
                    raise ValueError(
                        "Insufficient inner-training class windows"
                    )

            inner_summaries[inner_name] = {
                "training_groups": len(training),
                "validation_groups": len(validation_set),
                "training_class_support": support,
                "training_class_windows": class_windows,
            }

        summary[outer_name] = inner_summaries

    return {
        "valid": True,
        "outer_fold_count": len(outer_folds),
        "inner_evaluations": sum(len(v) for v in summary.values()),
        "diagnostics": summary,
    }
