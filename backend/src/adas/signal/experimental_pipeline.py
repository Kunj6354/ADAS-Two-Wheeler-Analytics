"""Research-only interpolation, FIR filtering, and rate reduction."""

import math
from bisect import bisect_right
from dataclasses import dataclass
from statistics import median

from adas.signal.adaptive_segmentation import TimingStretch
from adas.signal.filtering import apply_fir, design_lowpass_fir


@dataclass(frozen=True)
class ExperimentalSignal:
    source_file: str
    parent_segment_id: int
    stretch_id: int
    source_start_line: int
    source_end_line: int
    estimated_source_hz: float
    target_hz: float
    cutoff_hz: float
    times_ms: tuple[float, ...]
    values: tuple[float, ...]
    reconstruction_method: str = "linear"
    anti_aliasing_experiment: bool = True
    approved_for_ml: bool = False


def _interpolate(times, values, grid):
    result = []

    for point in grid:
        right = bisect_right(times, point)

        if right == 0:
            result.append(values[0])
        elif right >= len(times):
            result.append(values[-1])
        else:
            left = right - 1
            fraction = (
                (point - times[left]) /
                (times[right] - times[left])
            )
            result.append(
                values[left] +
                fraction * (values[right] - values[left])
            )

    return tuple(result)


def process_stretch(
    stretch: TimingStretch,
    channel: str = "acc_x",
    target_hz: float = 20.0,
    cutoff_hz: float = 6.0,
    num_taps: int = 51,
    min_samples: int = 100,
    max_interval_ratio: float = 3.0,
) -> ExperimentalSignal:
    """Process one stretch without using adjacent stretch samples.

    This does not validate spectral reconstruction of irregular data.
    """

    allowed = {
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
    }

    if channel not in allowed:
        raise ValueError("Unknown sensor channel")

    if (
        not math.isfinite(target_hz)
        or target_hz <= 0
        or not math.isfinite(cutoff_hz)
        or cutoff_hz <= 0
        or not math.isfinite(max_interval_ratio)
        or max_interval_ratio < 1
    ):
        raise ValueError("Invalid processing configuration")

    if not isinstance(num_taps, int) or isinstance(num_taps, bool):
        raise ValueError("Invalid FIR length")

    records = stretch.records

    if len(records) < min_samples:
        raise ValueError("Stretch too short")

    if any(r.source_file != stretch.source_file for r in records):
        raise ValueError("Source identity mismatch")

    if (
        records[0].source_line != stretch.start_line
        or records[-1].source_line != stretch.end_line
    ):
        raise ValueError("Source-line mismatch")

    times = [r.timestamp_ms for r in records]
    values = [getattr(r, channel) for r in records]

    if not all(math.isfinite(v) for v in times + values):
        raise ValueError("Non-finite input")

    intervals = [b - a for a, b in zip(times, times[1:])]

    if min(intervals) <= 0:
        raise ValueError("Non-increasing timestamps")

    dt = median(intervals)

    if max(intervals) > max_interval_ratio * dt:
        raise ValueError("Excessive timing variation")

    source_hz = 1000.0 / dt

    if source_hz < target_hz:
        raise ValueError("Target exceeds estimated source rate")

    if cutoff_hz >= min(source_hz, target_hz) / 2:
        raise ValueError("Cutoff violates Nyquist constraint")

    taps = design_lowpass_fir(source_hz, cutoff_hz, num_taps)

    source_count = math.floor(
        (times[-1] - times[0]) / dt
    ) + 1

    source_grid = tuple(
        times[0] + i * dt
        for i in range(source_count)
        if times[0] + i * dt <= times[-1]
    )

    if len(source_grid) <= num_taps:
        raise ValueError("Insufficient source-grid duration")

    reconstructed = _interpolate(times, values, source_grid)
    filtered = apply_fir(reconstructed, taps)

    # Exclude output points within half the FIR length of either edge.
    margin = num_taps // 2
    safe_start = source_grid[margin]
    safe_end = source_grid[-margin - 1]

    target_step = 1000.0 / target_hz
    first_index = math.ceil(
        (safe_start - times[0]) / target_step
    )
    last_index = math.floor(
        (safe_end - times[0]) / target_step
    )

    output_grid = tuple(
        times[0] + i * target_step
        for i in range(first_index, last_index + 1)
        if safe_start <= times[0] + i * target_step <= safe_end
    )

    if len(output_grid) < 2:
        raise ValueError("Too few edge-safe output samples")

    output = _interpolate(source_grid, filtered, output_grid)

    return ExperimentalSignal(
        source_file=stretch.source_file,
        parent_segment_id=stretch.parent_segment_id,
        stretch_id=stretch.stretch_id,
        source_start_line=stretch.start_line,
        source_end_line=stretch.end_line,
        estimated_source_hz=source_hz,
        target_hz=target_hz,
        cutoff_hz=cutoff_hz,
        times_ms=output_grid,
        values=output,
    )
