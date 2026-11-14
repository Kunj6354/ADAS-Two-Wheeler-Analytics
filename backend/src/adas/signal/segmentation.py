"""Timestamp-aware segmentation of ADAS sensor recordings."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from adas.data.loader import SensorRecord


BoundaryReason = Literal[
    "recording_start",
    "source_changed",
    "duplicate_timestamp",
    "decreasing_timestamp",
    "large_gap",
]


@dataclass(frozen=True)
class Segment:
    segment_id: int
    source_file: str
    start_line: int
    end_line: int
    start_timestamp_ms: float
    end_timestamp_ms: float
    records: tuple[SensorRecord, ...]
    start_reason: BoundaryReason

    @property
    def sample_count(self) -> int:
        return len(self.records)


def segment_records(
    records: Sequence[SensorRecord],
    gap_threshold_ms: float = 1000.0,
) -> list[Segment]:
    """Split contiguous recordings at source or timestamp boundaries.

    Records must be supplied in their original acquisition order.
    The threshold is an analysis configuration, not proof of data loss.
    """

    if not math.isfinite(gap_threshold_ms) or gap_threshold_ms <= 0:
        raise ValueError("gap_threshold_ms must be finite and positive")

    segments: list[Segment] = []
    current: list[SensorRecord] = []
    current_reason: BoundaryReason = "recording_start"

    def flush() -> None:
        if not current:
            return

        first = current[0]
        last = current[-1]

        segments.append(
            Segment(
                segment_id=len(segments) + 1,
                source_file=first.source_file,
                start_line=first.source_line,
                end_line=last.source_line,
                start_timestamp_ms=first.timestamp_ms,
                end_timestamp_ms=last.timestamp_ms,
                records=tuple(current),
                start_reason=current_reason,
            )
        )

    for record in records:
        if not math.isfinite(record.timestamp_ms):
            raise ValueError("Non-finite timestamp")

        reason: BoundaryReason | None = None

        if current:
            previous = current[-1]

            if record.source_file != previous.source_file:
                reason = "source_changed"
            else:
                delta = record.timestamp_ms - previous.timestamp_ms

                if delta == 0:
                    reason = "duplicate_timestamp"
                elif delta < 0:
                    reason = "decreasing_timestamp"
                elif delta > gap_threshold_ms:
                    reason = "large_gap"

        if reason is not None:
            flush()
            current = []
            current_reason = reason

        current.append(record)

    flush()
    return segments
