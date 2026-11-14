"""D4-approved window feature representations.

Restricted offline research only. No classifier fitting.
"""

from dataclasses import dataclass
from math import isfinite, sqrt, fsum
from statistics import fmean, pstdev

from adas.features.extraction import FEATURE_NAMES, STATISTICS, extract_window
from adas.features.windowing import FeatureWindow
from adas.evaluation.window_assignments import window_identity


ACCEL_NAMES = tuple(FEATURE_NAMES[:15])
RAW_NAMES = tuple(FEATURE_NAMES)

RELATIVE_NAMES = ACCEL_NAMES + tuple(
    f"{channel}_relative_{stat}"
    for channel in ("gyro_x", "gyro_y", "gyro_z")
    for stat in STATISTICS
)

APPROVED_VARIANTS = (
    "accelerometer_only",
    "accelerometer_plus_relative_angles",
    "accelerometer_plus_raw_angles",
)


@dataclass(frozen=True)
class D4FeatureRow:
    representation: str
    identity: tuple
    group: str
    label: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]


def extract_d4_representations(
    window: FeatureWindow,
) -> tuple[D4FeatureRow, ...]:
    """Extract exactly the three approved D4 representations."""

    raw = extract_window(window)
    identity = window_identity(window)

    accelerometer = tuple(raw.values[:15])
    relative_features = []

    for channel in ("gyro_x", "gyro_y", "gyro_z"):
        first = getattr(window.records[0], channel)

        measurements = tuple(
            getattr(record, channel) - first
            for record in window.records
        )

        if not all(isfinite(value) for value in measurements):
            raise ValueError("Non-finite relative angle")

        statistics = (
            fmean(measurements),
            pstdev(measurements),
            sqrt(fsum(value * value for value in measurements)
                 / len(measurements)),
            min(measurements),
            max(measurements),
        )

        relative_features.extend(statistics)

    representations = (
        ("accelerometer_only", ACCEL_NAMES, accelerometer),
        (
            "accelerometer_plus_relative_angles",
            RELATIVE_NAMES,
            accelerometer + tuple(relative_features),
        ),
        (
            "accelerometer_plus_raw_angles",
            RAW_NAMES,
            tuple(raw.values),
        ),
    )

    rows = tuple(
        D4FeatureRow(
            representation=name,
            identity=identity,
            group=window.source_file,
            label=raw.label,
            feature_names=names,
            values=values,
        )
        for name, names, values in representations
    )

    if tuple(row.representation for row in rows) != APPROVED_VARIANTS:
        raise ValueError("D4 representation mismatch")

    for row in rows:
        if len(row.feature_names) != len(row.values):
            raise ValueError("D4 feature dimension mismatch")
        if not all(isfinite(value) for value in row.values):
            raise ValueError("Non-finite D4 feature")

    return rows
