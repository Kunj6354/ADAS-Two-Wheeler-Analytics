"""Experimental adapter from timing stretches to uniform-grid signals."""

from dataclasses import dataclass

from adas.signal.adaptive_segmentation import TimingStretch
from adas.signal.resampling import ResampledSignal, resample_segment
from adas.signal.segmentation import Segment


@dataclass(frozen=True)
class StretchResamplingResult:
    stretch_id: int
    parent_segment_id: int
    source_file: str
    signal: ResampledSignal
    anti_aliased: bool = False
    approved_for_ml: bool = False


def interpolate_stretch(
    stretch: TimingStretch,
    target_hz: float = 20.0,
    min_samples: int = 100,
    max_interval_ratio: float = 3.0,
) -> StretchResamplingResult:
    """Interpolate one stretch without crossing its boundaries.

    This is not an anti-aliased downsampling pipeline.
    """

    if not stretch.records:
        raise ValueError("Cannot interpolate an empty stretch")

    if any(
        record.source_file != stretch.source_file
        for record in stretch.records
    ):
        raise ValueError("Stretch contains inconsistent source files")

    if (
        stretch.start_line != stretch.records[0].source_line
        or stretch.end_line != stretch.records[-1].source_line
    ):
        raise ValueError("Stretch source-line metadata mismatch")

    if stretch.stretch_id < 1 or stretch.parent_segment_id < 1:
        raise ValueError("Invalid stretch or parent identifier")

    first = stretch.records[0]
    last = stretch.records[-1]

    # Adapt an already-isolated stretch without modifying its records.
    segment = Segment(
        segment_id=stretch.stretch_id,
        source_file=stretch.source_file,
        start_line=stretch.start_line,
        end_line=stretch.end_line,
        start_timestamp_ms=first.timestamp_ms,
        end_timestamp_ms=last.timestamp_ms,
        records=stretch.records,
        start_reason="recording_start",
    )

    signal = resample_segment(
        segment,
        target_hz=target_hz,
        min_samples=min_samples,
        max_interval_ratio=max_interval_ratio,
    )

    return StretchResamplingResult(
        stretch_id=stretch.stretch_id,
        parent_segment_id=stretch.parent_segment_id,
        source_file=stretch.source_file,
        signal=signal,
    )
