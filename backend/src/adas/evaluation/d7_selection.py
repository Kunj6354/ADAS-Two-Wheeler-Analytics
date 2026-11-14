"""Approved D6 candidate grid and D7 deterministic score selection.

No dataset access, classifier fitting, or model evaluation here.
"""

from dataclasses import dataclass
from math import fsum, isfinite


TOLERANCE = 1e-12
SEED = 20261025

CLASS_NAMES = (
    "normal_driving",
    "rash_driving",
    "normal_turn",
    "rash_turn",
)


@dataclass(frozen=True)
class Candidate:
    family: str
    parameters: tuple[tuple[str, object], ...]

    @property
    def identity(self):
        return (self.family, self.parameters)


def approved_candidates():
    result = [Candidate("dummy", (("strategy", "most_frequent"),))]

    for c in (0.1, 1.0, 10.0):
        result.append(Candidate(
            "logistic_regression",
            (("C", c),),
        ))

    for estimators in (100, 200):
        for depth in (5, 10, None):
            for leaf in (1, 5):
                result.append(Candidate(
                    "random_forest",
                    (
                        ("n_estimators", estimators),
                        ("max_depth", depth),
                        ("min_samples_leaf", leaf),
                    ),
                ))

    candidates = tuple(result)

    if len(candidates) != 16:
        raise AssertionError("Invalid approved candidate count")

    if len({item.identity for item in candidates}) != 16:
        raise AssertionError("Duplicate candidate configuration")

    return candidates


def select_candidate(results):
    """Select using the approved D7 tie-break procedure.

    results maps candidate identities to three pairs:
        ((macro_f1, balanced_accuracy), ...)

    A candidate is ineligible if its result is absent,
    malformed, nonfinite, or has fewer than three folds.

    Failures must be logged by the caller separately.
    """

    candidates = approved_candidates()
    known = {candidate.identity for candidate in candidates}

    if any(identity not in known for identity in results):
        raise ValueError("Unexpected candidate identity")

    eligible = []

    for candidate in candidates:
        folds = results.get(candidate.identity)

        if folds is None or len(folds) != 3:
            continue

        try:
            scores = tuple(
                (float(f1), float(balanced))
                for f1, balanced in folds
            )
        except (TypeError, ValueError):
            continue

        if not all(
            isfinite(value) and 0.0 <= value <= 1.0
            for pair in scores
            for value in pair
        ):
            continue

        mean_f1 = fsum(pair[0] for pair in scores) / 3.0
        mean_balanced = fsum(pair[1] for pair in scores) / 3.0

        eligible.append((candidate, mean_f1, mean_balanced))

    if not eligible:
        raise ValueError("No eligible D7 candidate")

    # Step 1: highest inner-fold mean macro-F1.
    best_f1 = max(item[1] for item in eligible)
    finalists = [
        item for item in eligible
        if best_f1 - item[1] <= TOLERANCE
    ]

    # Step 2: highest mean balanced accuracy.
    best_balanced = max(item[2] for item in finalists)
    finalists = [
        item for item in finalists
        if best_balanced - item[2] <= TOLERANCE
    ]

    # Step 3: frozen grid order is the final deterministic tie-break.
    preferred = {item.identity: index for index, item in enumerate(candidates)}
    winner = min(finalists, key=lambda item: preferred[item[0].identity])

    return winner
