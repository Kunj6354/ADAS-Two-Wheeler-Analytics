"""Optional second-stage segmentation at unusual timing intervals."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from adas.data.loader import SensorRecord
from adas.signal.segmentation import Segment


@dataclass(frozen=True)
class TimingStretch:
    stretch_id: int
    parent_segment_id: int
    source_file: str
    start_line: int
    end_line: int
    records: tuple[SensorRecord, ...]
    median_parent_interval_ms: float | None
    start_reason: str

    @property
    def sample_count(self) -> int:
        return len(self.records)


def split_timing_stalls(
    segments: Sequence[Segment],
    interval_ratio: float = 3.0,
    min_parent_samples: int = 2,
) -> list[TimingStretch]:
    """Split each parent segment at intervals above ratio * median.

    Parent segments remain unchanged. Every input sample is preserved.
    The median is calculated once per parent segment.
    """

    if not math.isfinite(interval_ratio) or interval_ratio < 1:
        raise ValueError("interval_ratio must be finite and >= 1")

    if min_parent_samples < 2:
        raise ValueError("min_parent_samples must be >= 2")

    stretches: list[TimingStretch] = []

    for segment in segments:
        records = segment.records

        if not records:
            continue

        intervals = [
            b.timestamp_ms - a.timestamp_ms
            for a, b in zip(records, records[1:])
        ]

        if any(not math.isfinite(dt) or dt <= 0 for dt in intervals):
            raise ValueError("Parent timestamps must strictly increase")

        parent_median = median(intervals) if intervals else None

        if (
            len(records) < min_parent_samples
            or parent_median is None
        ):
            boundaries = []
        else:
            threshold = interval_ratio * parent_median
            boundaries = [
                index + 1
                for index, dt in enumerate(intervals)
                if dt > threshold
            ]

        edges = [0, *boundaries, len(records)]

        for index, (start, end) in enumerate(
            zip(edges, edges[1:])
        ):
            group = tuple(records[start:end])

            if not group:
                raise ValueError("Empty timing stretch")

            stretches.append(
                TimingStretch(
                    stretch_id=len(stretches) + 1,
                    parent_segment_id=segment.segment_id,
                    source_file=segment.source_file,
                    start_line=group[0].source_line,
                    end_line=group[-1].source_line,
                    records=group,
                    median_parent_interval_ms=parent_median,
                    start_reason=(
                        segment.start_reason
                        if index == 0
                        else "timing_stall"
                    ),
                )
            )

    return stretches
