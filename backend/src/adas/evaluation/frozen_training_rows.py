"""Exact frozen-window and recording-partition enforcement.

Pure validation only. No ML fitting, data loading or persistence.
"""

from dataclasses import dataclass
from collections import Counter

from adas.evaluation.d7_selection import CLASS_NAMES


@dataclass(frozen=True)
class FrozenRowPlan:
    identities: tuple[tuple, ...]
    groups: tuple[str, ...]
    labels: tuple[str, ...]
    primary_indices: tuple[int, ...]
    sensitivity_indices: tuple[int, ...]
    outer_test_indices: dict[str, tuple[int, ...]]
    inner_validation_indices: dict[str, dict[str, tuple[int, ...]]]


def validate_frozen_rows(
    identities,
    labels,
    groups,
    rate_compatible,
    inventory,
    freeze,
):
    """Accept only rows matching the inventory and D5 group authority."""

    ids = tuple(tuple(identity) for identity in identities)
    y = tuple(labels)
    g = tuple(groups)
    rates = tuple(rate_compatible)

    if not ids or not (
        len(ids) == len(y) == len(g) == len(rates)
    ):
        raise ValueError("Invalid row metadata lengths")

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate reconstructed identity")

    frozen_rows = inventory["windows"]
    expected = {}

    for item in frozen_rows:
        key = tuple(item["identity"])
        if key in expected:
            raise ValueError("Duplicate inventory identity")
        expected[key] = item

    if set(ids) != set(expected):
        raise ValueError("Frozen identity population mismatch")

    for identity, label, group, compatible in zip(ids, y, g, rates):
        if (
            len(identity) != 4
            or not isinstance(identity[0], str)
            or identity[0] != group
            or label not in CLASS_NAMES
            or label != expected[identity]["label"]
            or type(compatible) is not bool
            or compatible != expected[identity]["rate_compatible"]
        ):
            raise ValueError(f"Frozen metadata mismatch: {identity}")

    if freeze["freeze_status"] != "FROZEN":
        raise ValueError("D5 freeze not active")

    outer = freeze["outer_test_groups"]
    inner = freeze["inner_validation_groups_by_outer"]
    all_groups = set(g)

    if len(all_groups) != 24 or len(outer) != 4:
        raise ValueError("Invalid frozen group coverage")

    if set(inner) != set(outer):
        raise ValueError("Outer/inner keys differ")

    assigned = [group for fold in outer.values() for group in fold]

    if len(assigned) != 24 or set(assigned) != all_groups:
        raise ValueError("Invalid outer group partition")

    if len(set(assigned)) != 24:
        raise ValueError("Repeated outer recording")

    outer_indices = {}
    inner_indices = {}

    for outer_name, test_groups in outer.items():
        test_set = set(test_groups)
        development = all_groups - test_set

        if len(test_groups) != 6 or len(development) != 18:
            raise ValueError("Invalid outer fold dimensions")

        test_rows = tuple(
            index for index, group in enumerate(g)
            if group in test_set
        )

        if not test_rows:
            raise ValueError("Empty outer test fold")

        if set(y[index] for index in test_rows) != set(CLASS_NAMES):
            raise ValueError("Outer test fold missing a true class")

        folds = inner[outer_name]

        if set(folds) != {"fold_1", "fold_2", "fold_3"}:
            raise ValueError("Unexpected inner fold keys")

        inner_indices[outer_name] = {}
        combined = []

        for fold_name in ("fold_1", "fold_2", "fold_3"):
            validation = tuple(folds[fold_name])
            validation_set = set(validation)

            if (
                len(validation) != 6
                or len(validation_set) != 6
                or not validation_set <= development
            ):
                raise ValueError("Invalid inner validation assignment")

            fitting_set = development - validation_set

            if len(fitting_set) != 12:
                raise ValueError("Invalid inner fitting groups")

            validation_rows = tuple(
                index for index, group in enumerate(g)
                if group in validation_set
            )

            fitting_rows = tuple(
                index for index, group in enumerate(g)
                if group in fitting_set
            )

            if (
                not validation_rows
                or not fitting_rows
                or set(y[index] for index in validation_rows)
                   != set(CLASS_NAMES)
                or set(y[index] for index in fitting_rows)
                   != set(CLASS_NAMES)
            ):
                raise ValueError("Inner fold missing a required class")

            inner_indices[outer_name][fold_name] = validation_rows
            combined.extend(validation)

        if len(combined) != 18 or set(combined) != development:
            raise ValueError("Incomplete inner validation coverage")

        if len(set(combined)) != 18:
            raise ValueError("Duplicate inner recording assignment")

        outer_indices[outer_name] = test_rows

    primary = tuple(
        index for index, compatible in enumerate(rates)
        if compatible
    )

    return FrozenRowPlan(
        identities=ids,
        groups=g,
        labels=y,
        primary_indices=primary,
        sensitivity_indices=tuple(range(len(ids))),
        outer_test_indices=outer_indices,
        inner_validation_indices=inner_indices,
    )
