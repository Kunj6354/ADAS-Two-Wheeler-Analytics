"""Reconstruct and authenticate frozen D4 training matrices.

Restricted offline research. No estimator fitting or persistence.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from adas.data.loader import load_dataset
from adas.signal.segmentation import segment_records
from adas.signal.adaptive_segmentation import split_timing_stalls
from adas.features.windowing import build_windows
from adas.features.d4_representations import (
    APPROVED_VARIANTS,
    extract_d4_representations,
)
from adas.evaluation.window_assignments import window_identity
from adas.evaluation.frozen_training_rows import validate_frozen_rows


AUTHORITY_HASHES = {
    "M04_D5_COMMON_PARTITION_FREEZE_V1.json":
        "2bf73ab82513ccc8c06d2ef5b947581c3b3d8f30c73dc6555c07ef2ae94a889e",
    "M04_D3_STATISTICAL_INVENTORY_3578_V1_CANDIDATE.json":
        "8fa7d5b78ac141eca29d16ce9ba023cd38cd2296d9327e6a9f7907f2f3a3c721",
}

DIMENSIONS = {
    "accelerometer_only": 15,
    "accelerometer_plus_relative_angles": 30,
    "accelerometer_plus_raw_angles": 30,
}


@dataclass(frozen=True)
class FrozenFeatureDataset:
    plan: object
    matrices: dict
    feature_names: dict


def load_frozen_feature_dataset(root):
    """Read and validate only the approved canonical research inventory."""

    root = Path(root)
    docs = root / "docs"
    dataset = root / "Dataset" / "Own Bike"

    if not dataset.is_dir():
        raise ValueError("Expected historical recording directory missing")

    authorities = {}
    for filename, digest in AUTHORITY_HASHES.items():
        path = docs / filename
        if not path.is_file():
            raise ValueError(f"Missing authority: {filename}")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f"Authority hash mismatch: {filename}")
        authorities[filename] = json.loads(raw.decode("utf-8-sig"))

    inventory = authorities["M04_D3_STATISTICAL_INVENTORY_3578_V1_CANDIDATE.json"]
    freeze = authorities["M04_D5_COMMON_PARTITION_FREEZE_V1.json"]

    records, reports = load_dataset(dataset)
    if len(reports) != 39 or len(records) != 226053:
        raise ValueError("Historical recording baseline mismatch")

    windows = build_windows(
        split_timing_stalls(segment_records(records))
    )

    reconstructed = {}
    for window in windows:
        if window.status != "PURE_LABELED":
            continue
        identity = window_identity(window)
        if identity in reconstructed:
            raise ValueError("Duplicate reconstructed window identity")
        reconstructed[identity] = window

    items = inventory["windows"]
    identities = tuple(tuple(row["identity"]) for row in items)

    if len(identities) != 3578 or set(reconstructed) != set(identities):
        raise ValueError("Frozen reconstruction coverage mismatch")

    labels = tuple(reconstructed[key].label for key in identities)
    groups = tuple(key[0] for key in identities)
    rates = tuple(row["rate_compatible"] for row in items)

    plan = validate_frozen_rows(
        identities,
        labels,
        groups,
        rates,
        inventory,
        freeze,
    )

    values = {name: [] for name in APPROVED_VARIANTS}
    feature_names = {}

    for identity in identities:
        extracted = extract_d4_representations(reconstructed[identity])
        if tuple(row.representation for row in extracted) != APPROVED_VARIANTS:
            raise ValueError("Unexpected feature representation")

        for row in extracted:
            if (
                row.identity != identity
                or row.group != identity[0]
                or row.label != reconstructed[identity].label
            ):
                raise ValueError("Feature metadata mismatch")
            prior = feature_names.setdefault(
                row.representation, row.feature_names
            )
            if prior != row.feature_names:
                raise ValueError("Inconsistent feature order")
            values[row.representation].append(row.values)

    matrices = {}
    for name in APPROVED_VARIANTS:
        matrix = np.asarray(values[name], dtype=np.float64)
        if matrix.shape != (3578, DIMENSIONS[name]):
            raise ValueError(f"Invalid feature shape: {name}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Nonfinite feature matrix: {name}")
        matrix.setflags(write=False)
        matrices[name] = matrix

    accelerator = matrices["accelerometer_only"]
    for name in APPROVED_VARIANTS[1:]:
        if not np.array_equal(matrices[name][:, :15], accelerator):
            raise ValueError("Unmatched accelerometer columns")

    if len(plan.primary_indices) != 3478:
        raise ValueError("Primary population mismatch")

    for filename, digest in AUTHORITY_HASHES.items():
        if sha256((docs / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Authority changed during reconstruction")

    return FrozenFeatureDataset(plan, matrices, feature_names)
