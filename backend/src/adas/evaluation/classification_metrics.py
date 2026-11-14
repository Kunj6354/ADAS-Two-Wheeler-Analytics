"""Four-class offline evaluation metrics with explicit class coverage."""

from collections import Counter
from collections.abc import Sequence

import numpy as np


CLASS_NAMES = (
    "normal_driving",
    "rash_driving",
    "normal_turn",
    "rash_turn",
)


def evaluate_classification(y_true, y_pred, *, classes=CLASS_NAMES):
    """Evaluate supplied labels; never train or select a classifier."""

    names = tuple(classes)

    if names != CLASS_NAMES:
        raise ValueError("Unexpected class names or ordering")

    if isinstance(y_true, (str, bytes)) or isinstance(
        y_pred, (str, bytes)
    ):
        raise ValueError("Labels must be sequences")

    true = tuple(y_true)
    predicted = tuple(y_pred)

    if not true:
        raise ValueError("Empty evaluation labels")

    if len(true) != len(predicted):
        raise ValueError("Prediction length mismatch")

    if any(
        not isinstance(value, str) or value not in CLASS_NAMES
        for value in true + predicted
    ):
        raise ValueError("Unknown or invalid class label")

    true_counts = Counter(true)

    if any(true_counts[name] == 0 for name in CLASS_NAMES):
        raise ValueError("Evaluation partition lacks a true class")

    indices = {name: i for i, name in enumerate(CLASS_NAMES)}
    confusion = np.zeros((4, 4), dtype=np.int64)

    for actual, guess in zip(true, predicted):
        confusion[indices[actual], indices[guess]] += 1

    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    correct = np.diag(confusion)

    per_class = {}

    for i, name in enumerate(CLASS_NAMES):
        tp = int(correct[i])
        actual_count = int(support[i])
        predicted_count = int(predicted_support[i])

        precision = (
            tp / predicted_count if predicted_count else 0.0
        )
        recall = tp / actual_count
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        per_class[name] = {
            "support": actual_count,
            "predicted_count": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_f1 = sum(
        per_class[name]["f1"] for name in CLASS_NAMES
    ) / 4.0

    balanced_accuracy = sum(
        per_class[name]["recall"] for name in CLASS_NAMES
    ) / 4.0

    return {
        "class_names": CLASS_NAMES,
        "sample_count": len(true),
        "accuracy": float(correct.sum() / len(true)),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "confusion_matrix": tuple(
            tuple(int(value) for value in row)
            for row in confusion
        ),
        "per_class": per_class,
        "zero_division_policy": "zero predicted support -> precision 0",
        "absent_true_class_policy": "reject",
    }
