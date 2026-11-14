"""D6-approved, recording-isolated classifier execution.

Restricted offline research only. No dataset access or persistence.
"""

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from adas.evaluation.d7_selection import (
    CLASS_NAMES,
    SEED,
    approved_candidates,
)
from adas.evaluation.preprocessing import fit_training_scaler


LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def estimator_for(candidate):
    """Construct one exact approved estimator, without fitting."""

    if candidate not in approved_candidates():
        raise ValueError("Candidate is not approved by D6")

    parameters = dict(candidate.parameters)

    if candidate.family == "dummy":
        return DummyClassifier(strategy="most_frequent")

    if candidate.family == "logistic_regression":
        return LogisticRegression(
            C=parameters["C"],
            l1_ratio=0,
            solver="lbfgs",
            max_iter=2000,
            class_weight=None,
        )

    if candidate.family == "random_forest":
        return RandomForestClassifier(
            n_estimators=parameters["n_estimators"],
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            class_weight=None,
            random_state=SEED,
            n_jobs=1,
        )

    raise ValueError("Unsupported classifier family")


def fit_predict(
    candidate,
    training_values,
    training_labels,
    training_row_groups,
    prediction_values,
    *,
    training_groups,
    excluded_groups,
    feature_names,
):
    """Fit on authorized recordings, then predict excluded recordings.

    The caller remains responsible for passing only excluded-group
    rows as prediction_values and for verifying their identities.
    """

    training = frozenset(training_groups)
    excluded = frozenset(excluded_groups)
    row_groups = tuple(training_row_groups)
    names = tuple(feature_names)

    if not training or not excluded or training & excluded:
        raise ValueError("Invalid group isolation")

    if set(row_groups) != training:
        raise ValueError("Training row groups mismatch")

    x_train = np.asarray(training_values, dtype=np.float64)
    x_predict = np.asarray(prediction_values, dtype=np.float64)

    if (
        x_train.ndim != 2
        or x_predict.ndim != 2
        or x_train.shape[0] == 0
        or x_predict.shape[0] == 0
        or x_train.shape[1] != len(names)
        or x_predict.shape[1] != len(names)
        or not np.isfinite(x_train).all()
        or not np.isfinite(x_predict).all()
    ):
        raise ValueError("Invalid feature matrices")

    labels = tuple(training_labels)

    if len(labels) != x_train.shape[0]:
        raise ValueError("Training label count mismatch")

    if any(label not in LABEL_TO_ID for label in labels):
        raise ValueError("Unknown training class")

    y_train = np.asarray(
        [LABEL_TO_ID[label] for label in labels],
        dtype=np.int64,
    )

    # All classes are required by this restricted evaluation.
    if set(y_train.tolist()) != set(range(len(CLASS_NAMES))):
        raise ValueError("Training split missing an approved class")

    estimator = clone(estimator_for(candidate))

    if candidate.family == "logistic_regression":
        scaler = fit_training_scaler(
            x_train,
            row_groups,
            training_groups=training,
            excluded_groups=excluded,
            feature_names=names,
        )
        x_fit = scaler.transform(x_train)
        x_eval = scaler.transform(x_predict)
    else:
        x_fit = x_train
        x_eval = x_predict

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(x_fit, y_train)

    predictions = np.asarray(estimator.predict(x_eval))

    if predictions.shape != (len(x_predict),):
        raise ValueError("Invalid classifier prediction count")

    if not np.isin(predictions, list(range(len(CLASS_NAMES)))).all():
        raise ValueError("Classifier emitted unexpected label")

    return tuple(CLASS_NAMES[int(value)] for value in predictions)
