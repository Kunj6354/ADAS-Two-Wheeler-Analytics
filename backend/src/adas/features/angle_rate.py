"""Research-only rate estimates from logged integrated gyro angles."""

import math
from dataclasses import dataclass
from statistics import fmean, median

from adas.features.windowing import FeatureWindow


AXES = ("gyro_x", "gyro_y", "gyro_z")

RATE_FEATURE_NAMES = tuple(
    f"{axis}_derived_rate_{stat}"
    for axis in AXES
    for stat in ("mean", "rms", "max_abs")
)


@dataclass(frozen=True)
class DerivedRateFeatures:
    source_file: str
    stretch_id: int
    window_index: int
    values: tuple[float, ...]
    approved_for_ml: bool = False

    def as_feature_dict(self) -> dict[str, float]:
        return dict(zip(RATE_FEATURE_NAMES, self.values))


def extract_angle_rate_features(
    window: FeatureWindow,
    max_interval_ratio: float = 3.0,
) -> DerivedRateFeatures:
    """Estimate rates using differences within a single source window."""

    if window.status != "PURE_LABELED":
        raise ValueError("Only pure labeled windows are supported")

    records = window.records

    if len(records) < 3:
        raise ValueError("At least three observations required")

    if (
        not math.isfinite(max_interval_ratio)
        or max_interval_ratio < 1
    ):
        raise ValueError("Invalid interval ratio")

    if any(
        r.source_file != window.source_file
        or r.annotation != window.label
        or not window.start_ms <= r.timestamp_ms < window.end_ms
        for r in records
    ):
        raise ValueError("Inconsistent window records")

    intervals = [
        b.timestamp_ms - a.timestamp_ms
        for a, b in zip(records, records[1:])
    ]

    if (
        not all(math.isfinite(dt) and dt > 0 for dt in intervals)
    ):
        raise ValueError("Invalid timestamp interval")

    typical_dt = median(intervals)

    if max(intervals) > max_interval_ratio * typical_dt:
        raise ValueError("Timing discontinuity inside window")

    features = []

    for axis in AXES:
        angles = [getattr(r, axis) for r in records]

        if not all(math.isfinite(v) for v in angles):
            raise ValueError("Non-finite angle")

        rates = [
            (b - a) * 1000.0 / dt
            for a, b, dt in zip(
                angles,
                angles[1:],
                intervals,
            )
        ]

        if not all(math.isfinite(v) for v in rates):
            raise ValueError("Non-finite derived rate")

        features.extend((
            fmean(rates),
            math.sqrt(math.fsum(v * v for v in rates) / len(rates)),
            max(abs(v) for v in rates),
        ))

    return DerivedRateFeatures(
        source_file=window.source_file,
        stretch_id=window.stretch_id,
        window_index=window.window_index,
        values=tuple(features),
    )
