"""Deterministic, provenance-aware feature window construction."""

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from adas.data.loader import CLASSES, SensorRecord
from adas.signal.adaptive_segmentation import TimingStretch


WindowStatus = Literal[
    "PURE_LABELED",
    "UNANNOTATED",
    "MIXED_WITH_UNANNOTATED",
    "MIXED_CLASSES",
    "TOO_FEW_SAMPLES",
]


@dataclass(frozen=True)
class FeatureWindow:
    source_file: str
    parent_segment_id: int
    stretch_id: int
    window_index: int
    start_ms: float
    end_ms: float
    source_start_line: int | None
    source_end_line: int | None
    records: tuple[SensorRecord, ...]
    status: WindowStatus
    label: str | None

    @property
    def sample_count(self) -> int:
        return len(self.records)


def classify_window(
    records: Sequence[SensorRecord],
    min_samples: int = 20,
) -> tuple[WindowStatus, str | None]:
    """Classify without inventing or interpolating annotations."""

    if len(records) < min_samples:
        return "TOO_FEW_SAMPLES", None

    annotations = {record.annotation for record in records}

    if annotations == {"NAN"}:
        return "UNANNOTATED", None

    if "NAN" in annotations:
        return "MIXED_WITH_UNANNOTATED", None

    if len(annotations) != 1:
        return "MIXED_CLASSES", None

    label = next(iter(annotations))

    if label not in CLASSES:
        raise ValueError(f"Unexpected label: {label}")

    return "PURE_LABELED", label


def build_windows(
    stretches: Sequence[TimingStretch],
    window_ms: float = 2000.0,
    stride_ms: float = 1000.0,
    min_samples: int = 20,
) -> list[FeatureWindow]:
    """Build half-open, time-based windows within individual stretches."""

    if not math.isfinite(window_ms) or window_ms <= 0:
        raise ValueError("Invalid window duration")

    if not math.isfinite(stride_ms) or stride_ms <= 0:
        raise ValueError("Invalid window stride")

    if (
        isinstance(min_samples, bool)
        or not isinstance(min_samples, int)
        or min_samples < 1
    ):
        raise ValueError("min_samples must be a positive integer")

    result = []

    for stretch in stretches:
        records = stretch.records

        if len(records) < min_samples:
            continue

        if any(r.source_file != stretch.source_file for r in records):
            raise ValueError("Inconsistent source file")

        times = [record.timestamp_ms for record in records]

        if any(not math.isfinite(t) for t in times):
            raise ValueError("Non-finite timestamp")

        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Non-increasing timestamps")

        start_origin = times[0]
        final_timestamp = times[-1]

        window_index = 0

        while start_origin + window_index * stride_ms + window_ms <= final_timestamp:
            start = start_origin + window_index * stride_ms
            end = start + window_ms

            left = bisect_left(times, start)
            right = bisect_left(times, end)

            window_records = tuple(records[left:right])

            status, label = classify_window(
                window_records,
                min_samples=min_samples,
            )

            result.append(
                FeatureWindow(
                    source_file=stretch.source_file,
                    parent_segment_id=stretch.parent_segment_id,
                    stretch_id=stretch.stretch_id,
                    window_index=window_index,
                    start_ms=start,
                    end_ms=end,
                    source_start_line=(
                        window_records[0].source_line
                        if window_records else None
                    ),
                    source_end_line=(
                        window_records[-1].source_line
                        if window_records else None
                    ),
                    records=window_records,
                    status=status,
                    label=label,
                )
            )

            window_index += 1

    return result
