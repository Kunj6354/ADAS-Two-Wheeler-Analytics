"""Evaluation grouping contracts. No split generation or ML fitting."""

from collections.abc import Iterable, Mapping


def validate_group_partitions(
    partitions: Mapping[str, Iterable[str]],
    *,
    expected_groups: Iterable[str] | None = None,
) -> dict[str, frozenset[str]]:
    """Validate complete recording-group isolation across partitions.

    A partition contains recording identifiers, not individual windows.
    Supplying expected_groups also requires exact group coverage.
    """
    if not partitions:
        raise ValueError("At least one partition is required")

    seen: set[str] = set()
    validated: dict[str, frozenset[str]] = {}

    for partition_name, group_ids in partitions.items():
        if not isinstance(partition_name, str) or not partition_name.strip():
            raise ValueError("Invalid partition name")

        if isinstance(group_ids, str):
            raise TypeError("Groups must be an iterable of identifiers")

        values = tuple(group_ids)

        if not values:
            raise ValueError(f"Empty partition: {partition_name}")

        if any(
            not isinstance(value, str) or not value.strip()
            for value in values
        ):
            raise ValueError(f"Invalid group in {partition_name}")

        current = frozenset(values)

        if len(current) != len(values):
            raise ValueError(f"Duplicate group in {partition_name}")

        overlap = seen.intersection(current)

        if overlap:
            raise ValueError(
                f"Recording group reused across partitions: {sorted(overlap)}"
            )

        validated[partition_name] = current
        seen.update(current)

    if expected_groups is not None:
        if isinstance(expected_groups, str):
            raise TypeError("Expected groups must be an iterable")

        expected_values = tuple(expected_groups)

        if any(
            not isinstance(value, str) or not value.strip()
            for value in expected_values
        ):
            raise ValueError("Invalid expected recording group")

        expected = set(expected_values)

        if len(expected) != len(expected_values):
            raise ValueError("Duplicate expected recording group")

        if seen != expected:
            missing = sorted(expected - seen)
            unexpected = sorted(seen - expected)
            raise ValueError(
                f"Group coverage mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

    return validated
