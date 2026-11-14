"""Deterministic statistical feature extraction from labeled windows."""

import math
from dataclasses import dataclass
from statistics import fmean, pstdev

from adas.data.loader import CLASSES
from adas.features.windowing import FeatureWindow


CHANNELS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)

STATISTICS = ("mean", "std", "rms", "min", "max")

FEATURE_NAMES = tuple(
    f"{channel}_{statistic}"
    for channel in CHANNELS
    for statistic in STATISTICS
)


@dataclass(frozen=True)
class FeatureRow:
    source_file: str
    parent_segment_id: int
    stretch_id: int
    window_index: int
    start_ms: float
    end_ms: float
    source_start_line: int
    source_end_line: int
    sample_count: int
    label: str
    values: tuple[float, ...]

    def as_feature_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values))


def extract_window(window: FeatureWindow) -> FeatureRow:
    """Extract 30 raw-unit statistics from one pure labeled window."""

    if window.status != "PURE_LABELED":
        raise ValueError("Only pure labeled windows are supported")

    if window.label not in CLASSES:
        raise ValueError("Invalid supervised label")

    records = window.records

    if len(records) < 2:
        raise ValueError("At least two samples are required")

    if (
        window.source_start_line is None
        or window.source_end_line is None
    ):
        raise ValueError("Missing provenance")

    if (
        window.source_start_line != records[0].source_line
        or window.source_end_line != records[-1].source_line
    ):
        raise ValueError("Provenance mismatch")

    if any(
        record.source_file != window.source_file
        or record.annotation != window.label
        or not window.start_ms <= record.timestamp_ms < window.end_ms
        for record in records
    ):
        raise ValueError("Inconsistent window contents")

    values = []

    for channel in CHANNELS:
        measurements = tuple(
            getattr(record, channel) for record in records
        )

        if not all(math.isfinite(value) for value in measurements):
            raise ValueError("Non-finite sensor measurement")

        mean = fmean(measurements)
        std = pstdev(measurements)
        rms = math.sqrt(
            math.fsum(value * value for value in measurements)
            / len(measurements)
        )

        values.extend((
            mean,
            std,
            rms,
            min(measurements),
            max(measurements),
        ))

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite extracted feature")

    return FeatureRow(
        source_file=window.source_file,
        parent_segment_id=window.parent_segment_id,
        stretch_id=window.stretch_id,
        window_index=window.window_index,
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        source_start_line=window.source_start_line,
        source_end_line=window.source_end_line,
        sample_count=len(records),
        label=window.label,
        values=tuple(values),
    )


def extract_labeled_windows(
    windows: list[FeatureWindow],
) -> list[FeatureRow]:
    """Extract only pure labeled windows without changing their order."""

    return [
        extract_window(window)
        for window in windows
        if window.status == "PURE_LABELED"
    ]
