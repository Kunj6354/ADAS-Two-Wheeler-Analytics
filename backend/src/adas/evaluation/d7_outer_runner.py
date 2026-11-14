"""One frozen outer-fold refit and held-out prediction collection.

No split generation, file persistence, or automatic training launch.
"""

from dataclasses import dataclass
import numpy as np

from adas.evaluation.d6_execution import fit_predict
from adas.evaluation.d7_inner_runner import evaluate_inner
from adas.evaluation.d7_selection import CLASS_NAMES, approved_candidates


@dataclass(frozen=True)
class OuterPrediction:
    identity: tuple
    outer_fold: str
    true_label: str
    predicted_label: str


@dataclass(frozen=True)
class OuterResult:
    outer_fold: str
    status: str
    selected_candidate: tuple | None
    predictions: tuple[OuterPrediction, ...]
    inner_selection: object | None
    error: str | None


def run_outer_fold(
    plan,
    matrix,
    feature_names,
    population_indices,
    outer_fold,
    outer_test_groups,
    inner_validation_groups,
):
    """Run D7 inner selection, one outer refit, one outer prediction.

    plan must be produced by the frozen-row guard.
    population_indices must equal the full approved primary or
    sensitivity membership, never a caller-selected subset.
    """

    candidates = {
        candidate.identity: candidate
        for candidate in approved_candidates()
    }

    x = np.asarray(matrix, dtype=np.float64)
    names = tuple(feature_names)
    indices = tuple(population_indices)

    if (
        x.ndim != 2
        or x.shape[0] != len(plan.identities)
        or x.shape[1] != len(names)
        or not np.isfinite(x).all()
        or not names
        or len(set(names)) != len(names)
    ):
        raise ValueError("Invalid full feature matrix")

    if indices not in (
        plan.primary_indices,
        plan.sensitivity_indices,
    ):
        raise ValueError("Unapproved population membership")

    if len(indices) != len(set(indices)):
        raise ValueError("Repeated population row")

    if any(
        type(index) is not int
        or not 0 <= index < len(x)
        for index in indices
    ):
        raise ValueError("Invalid population index")

    if outer_fold not in plan.outer_test_indices:
        raise ValueError("Unknown frozen outer fold")

    expected_test = set(plan.outer_test_indices[outer_fold])
    selected = set(indices)
    test_groups = set(outer_test_groups)

    test_indices = tuple(
        index for index in indices if index in expected_test
    )
    development_indices = tuple(
        index for index in indices if index not in expected_test
    )

    if (
        not test_indices
        or not development_indices
        or len(test_groups) != 6
        or {plan.groups[i] for i in test_indices} != test_groups
        or any(plan.groups[i] in test_groups
               for i in development_indices)
        or set(test_indices) | set(development_indices) != selected
        or set(test_indices) & set(development_indices)
    ):
        raise ValueError("Frozen outer partition mismatch")

    expected_inner = plan.inner_validation_indices[outer_fold]
    if set(inner_validation_groups) != set(expected_inner):
        raise ValueError("Frozen inner fold identity mismatch")

    for fold, group_names in inner_validation_groups.items():
        expected_rows = set(expected_inner[fold]) & selected
        actual_rows = {
            i for i in development_indices
            if plan.groups[i] in set(group_names)
        }
        if expected_rows != actual_rows:
            raise ValueError("Frozen inner validation mismatch")

    development_groups = {
        plan.groups[i] for i in development_indices
    }

    if len(development_groups) != 18:
        raise ValueError("Incorrect development group coverage")

    if set(plan.labels[i] for i in test_indices) != set(CLASS_NAMES):
        raise ValueError("Outer test partition missing a class")

    if set(plan.labels[i] for i in development_indices) != set(CLASS_NAMES):
        raise ValueError("Outer development partition missing a class")

    inner = None
    winner_id = None

    try:
        inner = evaluate_inner(
            x[list(development_indices)],
            tuple(plan.labels[i] for i in development_indices),
            tuple(plan.groups[i] for i in development_indices),
            names,
            test_groups,
            inner_validation_groups,
        )

        if inner.status != "PASS" or inner.winner_identity not in candidates:
            raise ValueError("No eligible D7 selected configuration")

        winner_id = inner.winner_identity

        predictions = tuple(fit_predict(
            candidates[winner_id],
            x[list(development_indices)],
            tuple(plan.labels[i] for i in development_indices),
            tuple(plan.groups[i] for i in development_indices),
            x[list(test_indices)],
            training_groups=development_groups,
            excluded_groups=test_groups,
            feature_names=names,
        ))

        if (
            len(predictions) != len(test_indices)
            or any(label not in CLASS_NAMES for label in predictions)
        ):
            raise ValueError("Invalid held-out predictions")

        output = tuple(
            OuterPrediction(
                identity=plan.identities[index],
                outer_fold=outer_fold,
                true_label=plan.labels[index],
                predicted_label=prediction,
            )
            for index, prediction in zip(test_indices, predictions)
        )

        if (
            len({record.identity for record in output}) != len(output)
            or {record.identity for record in output}
               != {plan.identities[i] for i in test_indices}
        ):
            raise ValueError("Incomplete outer prediction identities")

        return OuterResult(
            outer_fold, "PASS", winner_id, output, inner, None,
        )

    except Exception as exc:
        return OuterResult(
            outer_fold,
            "INCOMPLETE",
            winner_id,
            (),
            inner,
            f"{type(exc).__name__}: {exc}",
        )
