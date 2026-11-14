"""D7 frozen-inner-fold execution coordinator.

No dataset loading, persistence, or split generation.
"""

from dataclasses import dataclass
from collections import Counter
from collections.abc import Mapping

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from adas.evaluation.d6_execution import fit_predict
from adas.evaluation.d7_selection import (
    CLASS_NAMES,
    approved_candidates,
    select_candidate,
)


@dataclass(frozen=True)
class InnerOutcome:
    candidate_identity: tuple
    inner_fold: str
    status: str
    macro_f1: float | None
    balanced_accuracy: float | None
    error: str | None


@dataclass(frozen=True)
class InnerSelection:
    winner_identity: tuple | None
    mean_macro_f1: float | None
    mean_balanced_accuracy: float | None
    outcomes: tuple[InnerOutcome, ...]
    status: str


def evaluate_inner(
    values,
    labels,
    groups,
    feature_names,
    outer_test_groups,
    inner_validation_groups: Mapping,
):
    """Evaluate exact D6 grid within one frozen outer-development set."""

    x = np.asarray(values, dtype=np.float64)
    y = tuple(labels)
    g = tuple(groups)
    names = tuple(feature_names)

    if (
        x.ndim != 2
        or x.shape[0] == 0
        or x.shape[1] != len(names)
        or len(y) != len(g) != len(x)
        or not np.isfinite(x).all()
        or not names
        or len(set(names)) != len(names)
    ):
        raise ValueError("Invalid inner evaluation input")

    if any(label not in CLASS_NAMES for label in y):
        raise ValueError("Unknown class")

    if any(not isinstance(group, str) or not group for group in g):
        raise ValueError("Invalid group")

    observed = set(g)
    held_out = set(outer_test_groups)

    if observed & held_out or not held_out:
        raise ValueError("Outer-test recording present in inner evaluation")

    if set(inner_validation_groups) != {"fold_1", "fold_2", "fold_3"}:
        raise ValueError("Expected three frozen inner folds")

    validation_sets = {
        name: set(inner_validation_groups[name])
        for name in ("fold_1", "fold_2", "fold_3")
    }

    if any(len(items) != 6 for items in validation_sets.values()):
        raise ValueError("Invalid inner validation group count")

    assignments = [
        group for items in validation_sets.values() for group in items
    ]

    if (
        len(assignments) != 18
        or len(set(assignments)) != 18
        or set(assignments) != observed
        or len(observed) != 18
    ):
        raise ValueError("Incomplete or overlapping inner group coverage")

    if any(items & held_out for items in validation_sets.values()):
        raise ValueError("Outer-test group leaked into inner validation")

    candidates = approved_candidates()
    scores = {}
    outcomes = []

    for candidate in candidates:
        fold_scores = []
        valid = True

        for fold_name in ("fold_1", "fold_2", "fold_3"):
            validation = validation_sets[fold_name]
            training = observed - validation

            train_mask = np.asarray([v in training for v in g])
            valid_mask = np.asarray([v in validation for v in g])

            actual = tuple(
                label for label, included in zip(y, valid_mask)
                if included
            )

            if (
                not train_mask.any()
                or not valid_mask.any()
                or set(actual) != set(CLASS_NAMES)
            ):
                raise ValueError(
                    f"Frozen validation class/row coverage violation: {fold_name}"
                )

            try:
                predicted = fit_predict(
                    candidate,
                    x[train_mask],
                    tuple(
                        label for label, included in zip(y, train_mask)
                        if included
                    ),
                    tuple(
                        group for group, included in zip(g, train_mask)
                        if included
                    ),
                    x[valid_mask],
                    training_groups=training,
                    excluded_groups=validation | held_out,
                    feature_names=names,
                )

                if len(predicted) != len(actual):
                    raise ValueError("Prediction count mismatch")

                if any(label not in CLASS_NAMES for label in predicted):
                    raise ValueError("Unexpected prediction class")

                f1 = float(f1_score(
                    actual,
                    predicted,
                    labels=CLASS_NAMES,
                    average="macro",
                    zero_division=0,
                ))
                balanced = float(balanced_accuracy_score(
                    actual,
                    predicted,
                ))

                if not (
                    np.isfinite(f1)
                    and np.isfinite(balanced)
                    and 0 <= f1 <= 1
                    and 0 <= balanced <= 1
                ):
                    raise ValueError("Invalid classification score")

                fold_scores.append((f1, balanced))
                outcomes.append(InnerOutcome(
                    candidate.identity, fold_name, "PASS",
                    f1, balanced, None,
                ))

            except Exception as exc:
                valid = False
                outcomes.append(InnerOutcome(
                    candidate.identity, fold_name, "FAILED",
                    None, None,
                    f"{type(exc).__name__}: {exc}",
                ))

        if valid and len(fold_scores) == 3:
            scores[candidate.identity] = tuple(fold_scores)

    if not scores:
        return InnerSelection(
            None, None, None, tuple(outcomes), "INCOMPLETE",
        )

    winner, mean_f1, mean_balanced = select_candidate(scores)

    return InnerSelection(
        winner.identity,
        mean_f1,
        mean_balanced,
        tuple(outcomes),
        "PASS",
    )
