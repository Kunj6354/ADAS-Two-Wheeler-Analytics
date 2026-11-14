"""D8 complete out-of-fold accounting for restricted ADAS research.

Consumes frozen row plans and outer-fold prediction records.
Does not train estimators, generate splits or write artifacts.
"""

from dataclasses import dataclass

from adas.evaluation.d7_selection import CLASS_NAMES


@dataclass(frozen=True)
class OOFReport:
    status: str
    expected_count: int
    observed_count: int
    macro_f1: float | None
    balanced_accuracy: float | None
    confusion_matrix: tuple
    class_metrics: tuple
    fold_metrics: tuple
    recording_metrics: tuple
    errors: tuple[str, ...]


def _metrics(records):
    """Compute fixed-order four-class metrics directly from records."""
    lookup = {label: index for index, label in enumerate(CLASS_NAMES)}
    matrix = [[0] * 4 for _ in range(4)]

    for record in records:
        matrix[lookup[record.true_label]][lookup[record.predicted_label]] += 1

    per_class = []
    for i, label in enumerate(CLASS_NAMES):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(matrix[j][i] for j in range(4))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        per_class.append((label, support, precision, recall, f1))

    if any(item[1] == 0 for item in per_class):
        raise ValueError("Required evaluation class missing")

    macro_f1 = sum(item[4] for item in per_class) / 4
    balanced_accuracy = sum(item[3] for item in per_class) / 4

    return (
        macro_f1,
        balanced_accuracy,
        tuple(tuple(row) for row in matrix),
        tuple(per_class),
    )


def collect_oof(plan, population_indices, outer_results):
    """Accept complete, identity-exact frozen four-fold predictions only."""

    selected = tuple(population_indices)

    if selected not in (plan.primary_indices, plan.sensitivity_indices):
        raise ValueError("Unapproved population")

    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate population index")

    expected_folds = set(plan.outer_test_indices)
    if len(expected_folds) != 4:
        raise ValueError("Expected four frozen outer folds")

    if len(outer_results) != 4:
        raise ValueError("Must provide exactly four outer results")

    result_by_fold = {}
    errors = []

    for result in outer_results:
        fold = result.outer_fold
        if fold in result_by_fold:
            raise ValueError(f"Duplicate outer result: {fold}")
        if fold not in expected_folds:
            raise ValueError(f"Unknown outer result: {fold}")
        result_by_fold[fold] = result

    if set(result_by_fold) != expected_folds:
        raise ValueError("Missing outer result")

    expected_by_fold = {
        fold: {
            plan.identities[index]: plan.labels[index]
            for index in selected
            if index in set(indices)
        }
        for fold, indices in plan.outer_test_indices.items()
    }

    observed = {}
    fold_records = {}

    for fold in sorted(expected_folds):
        result = result_by_fold[fold]
        allowed = expected_by_fold[fold]

        if result.status != "PASS":
            errors.append(f"{fold}: {result.status}: {result.error}")
            continue

        if result.error is not None or result.selected_candidate is None:
            errors.append(f"{fold}: inconsistent success metadata")
            continue

        local = {}

        for record in result.predictions:
            key = record.identity

            if record.outer_fold != fold:
                errors.append(f"{fold}: wrong prediction fold")
                continue

            if key not in allowed:
                errors.append(f"{fold}: unexpected identity {key}")
                continue

            if key in local or key in observed:
                errors.append(f"{fold}: duplicate identity {key}")
                continue

            if (
                record.true_label != allowed[key]
                or record.predicted_label not in CLASS_NAMES
            ):
                errors.append(f"{fold}: invalid label for {key}")
                continue

            local[key] = record
            observed[key] = record

        if set(local) != set(allowed):
            errors.append(
                f"{fold}: expected {len(allowed)} predictions, "
                f"received {len(local)} valid unique predictions"
            )

        fold_records[fold] = tuple(local.values())

    # D8 requires all four true classes in every held-out fold.
    # Reject invalid folds as auditable incomplete evaluations,
    # rather than attempting to calculate their metrics.
    for fold in sorted(expected_folds):
        true_classes = set(expected_by_fold[fold].values())
        if true_classes != set(CLASS_NAMES):
            missing = sorted(set(CLASS_NAMES) - true_classes)
            errors.append(
                f"{fold}: required evaluation classes missing: {missing}"
            )

    if errors:
        return OOFReport(
            "INCOMPLETE", len(selected), len(observed),
            None, None, (), (), (), (), tuple(errors),
        )

    if len(observed) != len(selected):
        raise ValueError("Incomplete pooled identity coverage")

    ordered = tuple(observed[plan.identities[i]] for i in selected)
    pooled_f1, pooled_balanced, pooled_cm, pooled_classes = _metrics(ordered)

    folds = []
    for fold in sorted(expected_folds):
        f1, balanced, cm, classes = _metrics(fold_records[fold])
        folds.append((fold, len(fold_records[fold]), f1, balanced, cm))

    recordings = []
    for group in sorted({plan.groups[i] for i in selected}):
        items = tuple(
            record for record in ordered
            if record.identity[0] == group
        )
        correct = sum(
            record.true_label == record.predicted_label
            for record in items
        )
        supports = tuple(
            sum(record.true_label == label for record in items)
            for label in CLASS_NAMES
        )
        recordings.append(
            (group, len(items), supports, correct / len(items))
        )

    return OOFReport(
        "PASS",
        len(selected),
        len(ordered),
        pooled_f1,
        pooled_balanced,
        pooled_cm,
        pooled_classes,
        tuple(folds),
        tuple(recordings),
        (),
    )
