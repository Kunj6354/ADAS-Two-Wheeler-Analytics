"""Research-only recording-cluster percentile bootstrap."""

import numpy as np

from adas.evaluation.classification_metrics import (
    CLASS_NAMES,
    evaluate_classification,
)


def recording_cluster_bootstrap(
    y_true,
    y_pred,
    recording_groups,
    *,
    repetitions=1000,
    seed=2026,
    confidence_level=0.95,
):
    """Resample complete recording groups for descriptive uncertainty."""

    if any(
        isinstance(item, (str, bytes))
        for item in (y_true, y_pred, recording_groups)
    ):
        raise ValueError("Expected label and recording sequences")

    true = tuple(y_true)
    predicted = tuple(y_pred)
    groups = tuple(recording_groups)

    if not true or not (
        len(true) == len(predicted) == len(groups)
    ):
        raise ValueError("Empty or inconsistent observation arrays")

    if any(
        not isinstance(group, str) or not group
        for group in groups
    ):
        raise ValueError("Invalid recording group")

    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("Invalid bootstrap repetition count")

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("Invalid random seed")

    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("Invalid confidence level")

    # This validates labels and complete true-class coverage.
    original = evaluate_classification(true, predicted)

    unique_groups = tuple(sorted(set(groups)))

    if len(unique_groups) < 2:
        raise ValueError("At least two recording groups required")

    by_group = {
        group: tuple(
            index
            for index, row_group in enumerate(groups)
            if row_group == group
        )
        for group in unique_groups
    }

    generator = np.random.default_rng(seed)

    macro_values = []
    balanced_values = []
    rejected_coverage = 0

    for _ in range(repetitions):
        sampled_group_indices = generator.integers(
            0, len(unique_groups), size=len(unique_groups)
        )

        sampled_rows = tuple(
            index
            for group_index in sampled_group_indices
            for index in by_group[unique_groups[int(group_index)]]
        )

        sampled_true = tuple(true[i] for i in sampled_rows)

        if set(sampled_true) != set(CLASS_NAMES):
            rejected_coverage += 1
            continue

        sampled_predicted = tuple(
            predicted[i] for i in sampled_rows
        )

        metrics = evaluate_classification(
            sampled_true, sampled_predicted
        )

        macro_values.append(metrics["macro_f1"])
        balanced_values.append(metrics["balanced_accuracy"])

    valid = len(macro_values)

    if valid == 0:
        raise ValueError(
            "No bootstrap resample had all four true classes"
        )

    tail = (1.0 - confidence_level) / 2.0

    def interval(values):
        lower, upper = np.quantile(
            np.asarray(values, dtype=float),
            (tail, 1.0 - tail),
        )
        return (float(lower), float(upper))

    return {
        "valid": True,
        "recording_count": len(unique_groups),
        "window_count": len(true),
        "requested_repetitions": repetitions,
        "valid_repetitions": valid,
        "rejected_class_coverage": rejected_coverage,
        "valid_fraction": valid / repetitions,
        "seed": seed,
        "confidence_level": float(confidence_level),
        "macro_f1": {
            "point_estimate": original["macro_f1"],
            "percentile_interval": interval(macro_values),
        },
        "balanced_accuracy": {
            "point_estimate": original["balanced_accuracy"],
            "percentile_interval": interval(balanced_values),
        },
        "method": (
            "Recording-group bootstrap with replacement; "
            "fixed number of draws; complete-group resampling; "
            "window-weighted metrics; percentile intervals"
        ),
        "class_coverage_policy": (
            "Resamples missing any true class are excluded and counted"
        ),
        "interpretation": (
            "Descriptive research estimate, not approved statistical "
            "inference or evidence of rider independence"
        ),
    }
