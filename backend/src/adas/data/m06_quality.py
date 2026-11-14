"""M06 descriptive, non-destructive CSV quality checks."""

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median


LABELS = frozenset({
    "normal_driving", "rash_driving",
    "normal_turn", "rash_turn", "NAN",
})
HEADER = (
    "Timestamp", "AccX", "AccY", "AccZ",
    "GyroX", "GyroY", "GyroZ", "Annotation",
)


@dataclass(frozen=True)
class QualityReport:
    status: str
    rows: int
    header_count: int
    invalid_rows: tuple
    label_counts: dict
    positive_intervals: int
    min_interval_ms: int | None
    median_interval_ms: float | None
    max_interval_ms: int | None
    short_labeled_runs: int
    short_run_threshold_ms: int
    trailing_header_count: int = 0


def inspect_recording(
    path: Path,
    *,
    short_run_threshold_ms: int = 200,
) -> QualityReport:
    """Inspect a recording without modifying or normalizing its data."""
    if isinstance(short_run_threshold_ms, bool) or (
        not isinstance(short_run_threshold_ms, int)
        or short_run_threshold_ms <= 0
    ):
        raise ValueError("Diagnostic duration must be a positive integer")

    labels = []
    timestamps = []
    errors = []
    header_count = 0
    trailing_header_line = None
    last_content_line = None

    with Path(path).open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row:
                continue

            last_content_line = line_number

            if row[0].strip().lower() == "timestamp":
                if tuple(field.strip() for field in row) != HEADER:
                    errors.append((line_number, "INVALID_HEADER"))
                else:
                    header_count += 1
                    if header_count > 1:
                        errors.append((line_number, "MISPLACED_HEADER"))
                    elif timestamps:
                        trailing_header_line = line_number
                continue

            if len(row) != 8:
                errors.append((line_number, "COLUMN_COUNT"))
                continue

            try:
                timestamp = int(row[0].strip())
            except ValueError:
                errors.append((line_number, "INVALID_TIMESTAMP"))
                continue

            try:
                values = [float(field.strip()) for field in row[1:7]]
            except ValueError:
                errors.append((line_number, "INVALID_SENSOR"))
                continue

            if not all(math.isfinite(value) for value in values):
                errors.append((line_number, "NONFINITE_SENSOR"))
                continue

            label = row[7].strip()
            if label not in LABELS:
                errors.append((line_number, "INVALID_ANNOTATION"))
                continue

            if timestamps and timestamp <= timestamps[-1]:
                errors.append((line_number, "NONINCREASING_TIMESTAMP"))
                continue

            timestamps.append(timestamp)
            labels.append(label)

    if (
        trailing_header_line is not None
        and trailing_header_line != last_content_line
    ):
        errors.append((trailing_header_line, "MISPLACED_HEADER"))

    intervals = [
        right - left
        for left, right in zip(timestamps, timestamps[1:])
    ]

    short_runs = 0
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and labels[end] == labels[start]:
            end += 1

        if labels[start] != "NAN":
            duration = timestamps[end - 1] - timestamps[start]
            if duration < short_run_threshold_ms:
                short_runs += 1

        start = end

    return QualityReport(
        status="STRUCTURAL_PASS" if not errors else "STRUCTURAL_FAIL",
        rows=len(timestamps),
        header_count=header_count,
        invalid_rows=tuple(errors),
        label_counts=dict(Counter(labels)),
        positive_intervals=len(intervals),
        min_interval_ms=min(intervals) if intervals else None,
        median_interval_ms=median(intervals) if intervals else None,
        max_interval_ms=max(intervals) if intervals else None,
        short_labeled_runs=short_runs,
        short_run_threshold_ms=short_run_threshold_ms,
        trailing_header_count=int(
            trailing_header_line is not None
            and trailing_header_line == last_content_line
            and header_count == 1
        ),
    )
