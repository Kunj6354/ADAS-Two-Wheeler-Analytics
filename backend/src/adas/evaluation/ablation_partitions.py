"""Research-only matched ablation partition and scaling checks."""

import numpy as np

from adas.evaluation.feature_ablations import validate_matched_ablations
from adas.evaluation.preprocessing import fit_training_scaler


def prepare_matched_partition(
    variants,
    *,
    training_groups,
    validation_groups,
    outer_evaluation_groups,
):
    """Fit each variant independently on identical training row identities."""

    audit = validate_matched_ablations(variants)
    reference = variants["combined_39"]

    training = frozenset(training_groups)
    validation = frozenset(validation_groups)
    outer = frozenset(outer_evaluation_groups)

    if not training or not validation or not outer:
        raise ValueError("All group partitions must be nonempty")

    if training & validation or training & outer or validation & outer:
        raise ValueError("Partition group leakage")

    if training | validation | outer != set(reference.groups):
        raise ValueError("Incomplete or extra partition groups")

    row_groups = np.asarray(reference.groups, dtype=object)
    train_mask = np.isin(row_groups, tuple(sorted(training)))
    val_mask = np.isin(row_groups, tuple(sorted(validation)))
    outer_mask = np.isin(row_groups, tuple(sorted(outer)))

    if not all(mask.any() for mask in (train_mask, val_mask, outer_mask)):
        raise ValueError("Empty partition rows")

    result = {}

    for name, variant in variants.items():
        if variant.groups != reference.groups:
            raise ValueError("Unmatched variant groups")

        scaler = fit_training_scaler(
            variant.values[train_mask],
            row_groups[train_mask].tolist(),
            training_groups=training,
            excluded_groups=validation | outer,
            feature_names=variant.feature_names,
        )

        transformed = scaler.transform(variant.values[val_mask])

        if not np.isfinite(transformed).all():
            raise ValueError("Nonfinite validation transform")

        result[name] = {
            "training_identities": tuple(
                identity for identity, include
                in zip(reference.identities, train_mask)
                if include
            ),
            "validation_identities": tuple(
                identity for identity, include
                in zip(reference.identities, val_mask)
                if include
            ),
            "training_rows": int(train_mask.sum()),
            "validation_rows": int(val_mask.sum()),
            "feature_count": len(variant.feature_names),
            "scaler": scaler,
        }

    first = next(iter(result.values()))

    for item in result.values():
        if item["training_identities"] != first["training_identities"]:
            raise ValueError("Training identities differ between variants")
        if item["validation_identities"] != first["validation_identities"]:
            raise ValueError("Validation identities differ between variants")

    return {
        "valid": True,
        "variant_count": len(result),
        "source_row_count": audit["row_count"],
        "partitions": result,
    }
