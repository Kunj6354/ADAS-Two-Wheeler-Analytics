"""Canonical window-assignment validation; no split generation."""

from collections.abc import Iterable, Mapping
from typing import Any

from adas.evaluation.group_isolation import validate_group_partitions


def window_identity(window: Any) -> tuple:
    """Return the source-scoped identity of one feature window."""
    names = (
        "source_file",
        "parent_segment_id",
        "stretch_id",
        "window_index",
    )
    values = tuple(getattr(window, name) for name in names)

    if not isinstance(values[0], str) or not values[0].strip():
        raise ValueError("Invalid window source_file")

    try:
        hash(values)
    except TypeError as error:
        raise ValueError("Unhashable window identity") from error

    return values


def validate_window_assignments(
    assignments: Mapping[str, Iterable[Any]],
    expected_windows: Iterable[Any],
) -> dict[str, tuple[int, int]]:
    """Validate assignments against trusted canonical window objects.

    Identity alone is insufficient: each assigned window must also
    compare equal to the trusted canonical window with that identity.

    The caller must supply an immutable or independently trusted
    canonical collection. This function does not authenticate the
    underlying raw dataset or verify rider/session independence.
    """
    if not assignments:
        raise ValueError("No window partitions supplied")

    canonical: dict[tuple, Any] = {}

    for window in expected_windows:
        key = window_identity(window)

        if key in canonical:
            raise ValueError(f"Duplicate canonical window: {key}")

        canonical[key] = window

    if not canonical:
        raise ValueError("No canonical windows supplied")

    seen: set[tuple] = set()
    partition_groups: dict[str, set[str]] = {}
    summary: dict[str, tuple[int, int]] = {}

    for name, windows in assignments.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid partition name")

        if isinstance(windows, (str, bytes)):
            raise TypeError("Partition windows must be an iterable")

        items = tuple(windows)

        if not items:
            raise ValueError(f"Empty window partition: {name}")

        groups: set[str] = set()

        for window in items:
            key = window_identity(window)

            if key in seen:
                raise ValueError(f"Repeated window assignment: {key}")

            if key not in canonical:
                raise ValueError(f"Unexpected window identity: {key}")

            if window != canonical[key]:
                raise ValueError(f"Canonical window mismatch: {key}")

            seen.add(key)
            groups.add(key[0])

        partition_groups[name] = groups
        summary[name] = (len(items), len(groups))

    validate_group_partitions(
        partition_groups,
        expected_groups={key[0] for key in canonical},
    )

    if seen != set(canonical):
        missing = set(canonical) - seen
        raise ValueError(
            f"Missing canonical windows: {len(missing)}"
        )

    return summary
