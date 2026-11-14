"""Recording-group partition acceptance rules. No split generation."""

from collections import Counter
from collections.abc import Mapping, Sequence

from adas.evaluation.group_isolation import validate_group_partitions


def validate_partition_class_coverage(
    partitions: Mapping[str, Sequence[str]],
    recording_counts: Mapping[str, Mapping[str, int]],
    classes: Sequence[str],
    *,
    minimum_test_windows: int = 10,
) -> dict[str, dict[str, int]]:
    """Validate complete group isolation and per-partition class coverage.

    Window counts describe support, not independent riding events.
    This does not establish rider/session independence.
    """
    required = {"train", "validation", "test"}

    if set(partitions) != required:
        raise ValueError("Exactly train, validation and test are required")

    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Classes must be nonempty and unique")

    if any(not isinstance(label, str) or not label.strip()
           for label in classes):
        raise ValueError("Invalid class label")

    if not isinstance(minimum_test_windows, int) or isinstance(
        minimum_test_windows, bool
    ) or minimum_test_windows < 1:
        raise ValueError("Invalid minimum test support")

    if not recording_counts:
        raise ValueError("No recording counts provided")

    for recording, counts in recording_counts.items():
        if not isinstance(recording, str) or not recording.strip():
            raise ValueError("Invalid recording identifier")

        if set(counts) - set(classes):
            raise ValueError("Unknown class in recording counts")

        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in counts.values()
        ):
            raise ValueError("Invalid class count")

        if sum(counts.values()) == 0:
            raise ValueError("Recording has no eligible windows")

    validated = validate_group_partitions(
        partitions,
        expected_groups=recording_counts.keys(),
    )

    result = {}

    for name, groups in validated.items():
        totals = Counter()

        for group in groups:
            totals.update(recording_counts[group])

        result[name] = {
            label: totals[label]
            for label in classes
        }

        if any(totals[label] == 0 for label in classes):
            raise ValueError(
                f"Missing class representation in {name}"
            )

    if any(
        result["test"][label] < minimum_test_windows
        for label in classes
    ):
        raise ValueError("Insufficient test-class window support")

    return result
