"""Recording-aware, outer-fold classification reporting."""

from collections import Counter
from collections.abc import Mapping
from math import isfinite

from adas.evaluation.classification_metrics import (
    CLASS_NAMES,
    evaluate_classification,
)


def evaluate_grouped_predictions(
    y_true,
    y_pred,
    row_groups,
    outer_folds,
):
    """Evaluate supplied predictions; no fitting or model selection."""

    if any(isinstance(v, (str, bytes)) for v in (y_true, y_pred, row_groups)):
        raise ValueError("Expected label and group sequences")

    true = tuple(y_true)
    pred = tuple(y_pred)
    groups = tuple(row_groups)

    if not true or len(true) != len(pred) or len(true) != len(groups):
        raise ValueError("Empty or inconsistent row inventory")

    if any(label not in CLASS_NAMES for label in true + pred):
        raise ValueError("Unexpected behavior class")

    if any(not isinstance(g, str) or not g for g in groups):
        raise ValueError("Invalid recording group")

    if not isinstance(outer_folds, Mapping) or not outer_folds:
        raise ValueError("Missing outer-fold mapping")

    assigned = []

    for fold, members in outer_folds.items():
        if not isinstance(fold, str) or not fold:
            raise ValueError("Invalid fold name")
        if isinstance(members, (str, bytes)):
            raise ValueError("Fold members must be a sequence")

        members = tuple(members)
        if not members or len(members) != len(set(members)):
            raise ValueError("Empty or duplicate fold members")
        assigned.extend(members)

    if len(assigned) != len(set(assigned)):
        raise ValueError("Recording present in multiple outer folds")

    if set(assigned) != set(groups):
        raise ValueError("Outer-fold recording coverage mismatch")

    per_fold = {}
    per_recording = {}
    confusion_total = [[0] * 4 for _ in range(4)]

    for fold in sorted(outer_folds):
        members = set(outer_folds[fold])
        indices = [i for i, g in enumerate(groups) if g in members]

        fold_true = tuple(true[i] for i in indices)
        fold_pred = tuple(pred[i] for i in indices)

        # Four-class coverage is mandatory for outer fold metrics.
        metrics = evaluate_classification(fold_true, fold_pred)
        per_fold[fold] = metrics

        for i, row in enumerate(metrics["confusion_matrix"]):
            for j, value in enumerate(row):
                confusion_total[i][j] += value

        for group in sorted(members):
            if group in per_recording:
                raise ValueError("Duplicate recording report")

            positions = [i for i in indices if groups[i] == group]
            actual = tuple(true[i] for i in positions)
            guessed = tuple(pred[i] for i in positions)

            support = Counter(actual)
            predicted_support = Counter(guessed)
            correct = sum(a == b for a, b in zip(actual, guessed))

            # Sparse recordings must not receive a fabricated
            # full four-class macro-F1.
            full_metrics = (
                evaluate_classification(actual, guessed)
                if all(support[name] > 0 for name in CLASS_NAMES)
                else None
            )

            per_recording[group] = {
                "outer_fold": fold,
                "sample_count": len(positions),
                "accuracy": correct / len(positions),
                "class_support": {
                    name: support[name] for name in CLASS_NAMES
                },
                "predicted_support": {
                    name: predicted_support[name] for name in CLASS_NAMES
                },
                "four_class_macro_f1": (
                    full_metrics["macro_f1"]
                    if full_metrics is not None else None
                ),
                "four_class_coverage": full_metrics is not None,
            }

    total_samples = len(true)
    correct_total = sum(
        confusion_total[i][i] for i in range(4)
    )

    if sum(map(sum, confusion_total)) != total_samples:
        raise ValueError("Confusion count reconciliation failed")

    return {
        "valid": True,
        "sample_count": total_samples,
        "recording_count": len(per_recording),
        "outer_fold_count": len(per_fold),
        "overall_accuracy": correct_total / total_samples,
        "pooled_confusion_matrix": tuple(
            tuple(row) for row in confusion_total
        ),
        "per_fold": per_fold,
        "per_recording": per_recording,
        "aggregation_policy": (
            "Report fold metrics individually; pooled confusion "
            "is descriptive and must not substitute for grouped uncertainty"
        ),
    }
