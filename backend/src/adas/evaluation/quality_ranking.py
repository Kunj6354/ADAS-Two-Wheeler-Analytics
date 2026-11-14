"""Deterministic quality ranking of already eligible partitions.

No candidate generation, model evaluation, file I/O or split approval.
"""

from collections.abc import Mapping, Sequence
from math import fsum

from adas.evaluation.quality_contract import (
    QualityLimits,
    evaluate_partition_quality,
)


def quality_rank_key(
    partitions: Mapping[str, Sequence[str]],
    recording_counts: Mapping[str, Mapping[str, int]],
    classes: Sequence[str],
    limits: QualityLimits = QualityLimits(),
) -> tuple:
    """Return a lower-is-better deterministic ranking key.

    Eligibility is mandatory. Ranking is provisional and must not
    be interpreted as scientific evidence of generalization.
    """
    result = evaluate_partition_quality(
        partitions, recording_counts, classes, limits
    )

    if not result["eligible"]:
        raise ValueError(
            "Ineligible candidate: "
            + ", ".join(result["violations"])
        )

    diagnostics = result["diagnostics"]
    labels = tuple(sorted(classes))

    global_counts = {
        label: sum(
            values.get(label, 0)
            for values in recording_counts.values()
        )
        for label in labels
    }
    total = sum(global_counts.values())

    global_shares = {
        label: global_counts[label] / total
        for label in labels
    }

    worst_concentration = max(
        diagnostics[name]["class_concentration"][label]
        for name in ("train", "validation", "test")
        for label in labels
    )

    evaluation_distances = []

    for name in ("validation", "test"):
        item = diagnostics[name]
        distance = fsum(
            abs(
                item["class_counts"][label] / item["windows"]
                - global_shares[label]
            )
            for label in labels
        )
        evaluation_distances.append(distance)

    # A fixed tie-break makes results independent of dict insertion order.
    identity = tuple(
        tuple(sorted(partitions[name]))
        for name in ("train", "validation", "test")
    )

    return (
        worst_concentration,
        max(evaluation_distances),
        -diagnostics["train"]["share"],
        identity,
    )
