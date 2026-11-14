"""Conservative timestamp-aware resampling for ADAS segments."""

import math
from bisect import bisect_right
from dataclasses import dataclass
from statistics import median

from adas.signal.segmentation import Segment


CHANNELS = (
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
)


@dataclass(frozen=True)
class ResampledSignal:
    source_file: str
    segment_id: int
    times_ms: tuple[float, ...]
    channels: dict[str, tuple[float, ...]]
    source_start_line: int
    source_end_line: int
    target_hz: float
    method: str = "linear_interpolation"


def resample_segment(
    segment: Segment,
    target_hz: float,
    min_samples: int = 100,
    max_interval_ratio: float = 3.0,
) -> ResampledSignal:
    """Interpolate one segment onto a uniform time grid.

    The output is derived data, not original sensor observations.
    No extrapolation or cross-segment interpolation is allowed.
    """

    if not math.isfinite(target_hz) or target_hz <= 0:
        raise ValueError("Target frequency must be finite and positive")

    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")

    if (
        not math.isfinite(max_interval_ratio)
        or max_interval_ratio < 1
    ):
        raise ValueError("Invalid maximum interval ratio")

    records = segment.records

    if len(records) < min_samples:
        raise ValueError("Segment has insufficient samples")

    times = [r.timestamp_ms for r in records]

    if not all(math.isfinite(t) for t in times):
        raise ValueError("Non-finite timestamp")

    intervals = [b - a for a, b in zip(times, times[1:])]

    if any(dt <= 0 for dt in intervals):
        raise ValueError("Timestamps must increase strictly")

    median_interval = median(intervals)

    if max(intervals) > max_interval_ratio * median_interval:
        raise ValueError("Segment contains excessive interval variation")

    observed_hz = 1000.0 / median_interval

    if target_hz > observed_hz:
        raise ValueError("Upsampling is disabled in the initial policy")

    step_ms = 1000.0 / target_hz
    duration = times[-1] - times[0]
    count = math.floor(duration / step_ms) + 1

    if count < 2:
        raise ValueError("Insufficient duration for target grid")

    grid = tuple(times[0] + i * step_ms for i in range(count))

    # Avoid floating-point roundoff beyond the last observation.
    grid = tuple(t for t in grid if t <= times[-1])

    output: dict[str, tuple[float, ...]] = {}

    for channel in CHANNELS:
        original = [getattr(record, channel) for record in records]

        if not all(math.isfinite(v) for v in original):
            raise ValueError("Non-finite channel measurement")

        values = []

        for time in grid:
            right = bisect_right(times, time)

            if right == 0:
                values.append(original[0])
                continue

            if right == len(times):
                values.append(original[-1])
                continue

            left = right - 1
            fraction = (
                (time - times[left]) /
                (times[right] - times[left])
            )

            values.append(
                original[left] +
                fraction * (original[right] - original[left])
            )

        output[channel] = tuple(values)

    return ResampledSignal(
        source_file=segment.source_file,
        segment_id=segment.segment_id,
        times_ms=grid,
        channels=output,
        source_start_line=segment.start_line,
        source_end_line=segment.end_line,
        target_hz=target_hz,
    )
