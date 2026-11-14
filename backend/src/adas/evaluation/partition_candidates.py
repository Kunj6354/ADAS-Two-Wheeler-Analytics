"""Deterministic diagnostic group-partition search.

This is not a final holdout-selection policy.
"""

from collections.abc import Mapping, Sequence
from itertools import combinations

from adas.evaluation.partition_contract import (
    validate_partition_class_coverage,
)


def first_feasible_candidate(
    recording_counts: Mapping[str, Mapping[str, int]],
    classes: Sequence[str],
    *,
    test_group_count: int = 2,
    validation_group_count: int = 1,
    minimum_test_windows: int = 10,
) -> dict[str, tuple[str, ...]] | None:
    """Return the first lexicographically feasible diagnostic partition.

    Returns None when no candidate satisfies the acceptance contract.
    Does not train models, write files, or freeze a test set.
    """
    if not isinstance(test_group_count, int) or isinstance(
        test_group_count, bool
    ) or test_group_count < 1:
        raise ValueError("Invalid test group count")

    if not isinstance(validation_group_count, int) or isinstance(
        validation_group_count, bool
    ) or validation_group_count < 1:
        raise ValueError("Invalid validation group count")

    names = sorted(recording_counts)

    if len(names) <= test_group_count + validation_group_count:
        return None

    for test in combinations(names, test_group_count):
        remaining = tuple(name for name in names if name not in test)

        for validation in combinations(
            remaining, validation_group_count
        ):
            train = tuple(
                name for name in remaining
                if name not in validation
            )

            candidate = {
                "train": train,
                "validation": validation,
                "test": test,
            }

            try:
                validate_partition_class_coverage(
                    candidate,
                    recording_counts,
                    classes,
                    minimum_test_windows=minimum_test_windows,
                )
            except ValueError:
                continue

            return candidate

    return None
