"""Training-group-bound standardization without model fitting."""

from dataclasses import dataclass

import numpy as np


def _matrix(values, expected_columns=None):
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("Expected nonempty 2D feature matrix")

    if array.shape[1] == 0:
        raise ValueError("Feature matrix has zero columns")

    if expected_columns is not None:
        if array.shape[1] != expected_columns:
            raise ValueError("Feature dimension mismatch")

    if not np.isfinite(array).all():
        raise ValueError("Nonfinite feature values are prohibited")

    return array


@dataclass(frozen=True)
class TrainingScaler:
    """Immutable training-fitted standardization parameters."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    training_groups: frozenset[str]

    def transform(self, values):
        """Transform without fitting or changing parameters."""
        matrix = _matrix(values, len(self.feature_names))
        result = (
            matrix - np.asarray(self.means, dtype=np.float64)
        ) / np.asarray(self.scales, dtype=np.float64)

        if not np.isfinite(result).all():
            raise ValueError("Nonfinite standardized values")

        return result


def fit_training_scaler(
    values,
    row_groups,
    *,
    training_groups,
    excluded_groups,
    feature_names,
):
    """Fit statistics only on explicitly authorized training groups."""

    names = tuple(feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Feature names must be unique and nonempty")

    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Invalid feature name")

    matrix = _matrix(values, len(names))
    row_groups = tuple(row_groups)

    if len(row_groups) != len(matrix):
        raise ValueError("Row/group count mismatch")

    training = frozenset(training_groups)
    excluded = frozenset(excluded_groups)

    if not training or not excluded:
        raise ValueError("Explicit training and exclusion groups required")

    if training & excluded:
        raise ValueError("Training/exclusion group leakage")

    if any(not isinstance(g, str) or not g for g in training | excluded):
        raise ValueError("Invalid recording group identity")

    observed = frozenset(row_groups)

    if observed != training:
        raise ValueError("Training rows do not exactly cover training groups")

    if observed & excluded:
        raise ValueError("Excluded group present in fitting data")

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=0)

    # Constant features remain centered and finite.
    scales = np.where(scales == 0.0, 1.0, scales)

    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("Invalid fitted statistics")

    if np.any(scales <= 0):
        raise ValueError("Invalid scaling factor")

    return TrainingScaler(
        feature_names=names,
        means=tuple(float(x) for x in means),
        scales=tuple(float(x) for x in scales),
        training_groups=training,
    )
