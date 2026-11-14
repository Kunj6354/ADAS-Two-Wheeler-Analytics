"""Feasibility-only MILP fold construction for offline research."""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from adas.evaluation.grouped_folds import (
    FoldLimits,
    validate_grouped_folds,
)


def solve_grouped_folds(
    recording_counts: Mapping[str, Mapping[str, int]],
    classes: Sequence[str],
    limits: FoldLimits = FoldLimits(),
    *,
    time_limit: float = 60.0,
) -> dict[str, tuple[str, ...]] | None:
    """Return independently verified feasible folds.

    None means proven infeasible, not timeout.
    Solver failures or unresolved status raise RuntimeError.
    No file I/O or model training occurs.
    """
    from math import isfinite

    if (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("Invalid solver time limit")

    names = tuple(sorted(recording_counts))
    f_count = limits.n_folds

    # Validate input shapes and limits using the established contract.
    # Construct a temporary complete allocation solely for input
    # validation; numerical quality is not assumed.
    if len(names) != f_count * limits.groups_per_fold:
        raise ValueError("Recording count incompatible with fold sizes")

    placeholder = {
        f"fold_{f + 1}": tuple(
            names[f * limits.groups_per_fold:
                  (f + 1) * limits.groups_per_fold]
        )
        for f in range(f_count)
    }
    validate_grouped_folds(
        placeholder, recording_counts, classes, limits
    )

    g_count = len(names)
    n_vars = g_count * f_count

    def idx(g, f):
        return g * f_count + f

    sizes = [
        sum(recording_counts[name].values())
        for name in names
    ]
    total = sum(sizes)
    rows, lows, highs = [], [], []

    def add(terms, low, high):
        rows.append(terms)
        lows.append(low)
        highs.append(high)

    for g in range(g_count):
        add(
            {idx(g, f): 1 for f in range(f_count)},
            1, 1,
        )

    for f in range(f_count):
        add(
            {idx(g, f): 1 for g in range(g_count)},
            limits.groups_per_fold,
            limits.groups_per_fold,
        )
        add(
            {idx(g, f): sizes[g] for g in range(g_count)},
            limits.minimum_window_share * total,
            limits.maximum_window_share * total,
        )

        for label in classes:
            values = [
                recording_counts[name].get(label, 0)
                for name in names
            ]
            add(
                {
                    idx(g, f): int(values[g] > 0)
                    for g in range(g_count)
                },
                limits.minimum_class_groups,
                np.inf,
            )
            add(
                {idx(g, f): values[g] for g in range(g_count)},
                limits.minimum_class_windows,
                np.inf,
            )

            for g in range(g_count):
                if values[g] == 0:
                    continue

                terms = {
                    idx(j, f):
                        -limits.maximum_class_concentration * values[j]
                    for j in range(g_count)
                    if values[j] > 0
                }
                terms[idx(g, f)] += values[g]
                add(terms, -np.inf, 0)

    matrix = lil_matrix((len(rows), n_vars), dtype=float)
    for r, terms in enumerate(rows):
        for column, coefficient in terms.items():
            matrix[r, column] = coefficient

    result = milp(
        c=np.zeros(n_vars),
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(
            matrix.tocsr(),
            np.asarray(lows, dtype=float),
            np.asarray(highs, dtype=float),
        ),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )

    if result.status == 2:
        return None

    if result.status != 0 or result.x is None:
        raise RuntimeError(
            f"MILP unresolved: status={result.status}, "
            f"message={result.message}"
        )

    rounded = np.rint(result.x)
    if not np.allclose(result.x, rounded, atol=1e-6):
        raise RuntimeError("Non-integral assignment")

    allocation = rounded.astype(int).reshape(g_count, f_count)
    folds = {
        f"fold_{f + 1}": tuple(
            names[g]
            for g in range(g_count)
            if allocation[g, f] == 1
        )
        for f in range(f_count)
    }

    verified = validate_grouped_folds(
        folds, recording_counts, classes, limits
    )
    if not verified["eligible"]:
        raise RuntimeError(
            f"Independent validation failed: {verified['violations']}"
        )

    return folds
