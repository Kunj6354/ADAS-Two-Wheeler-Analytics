"""Deterministic quality summaries for historical ADAS recordings."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from adas.data.loader import CLASSES, FileReport, SensorRecord


@dataclass(frozen=True)
class TimestampStats:
    transitions: int
    duplicate: int
    decreasing: int
    gaps_over_threshold: int
    minimum_positive_interval_ms: float | None
    maximum_positive_interval_ms: float | None


def analyze_timestamps(
    records: Sequence[SensorRecord],
    gap_threshold_ms: float = 1000.0,
) -> TimestampStats:
    """Analyze consecutive timestamps from one recording."""

    if not 0 < gap_threshold_ms < float("inf"):
        raise ValueError("Gap threshold must be finite and positive")

    duplicate = 0
    decreasing = 0
    gaps = 0
    minimum = None
    maximum = None

    for previous, current in zip(records, records[1:]):
        delta = current.timestamp_ms - previous.timestamp_ms

        if delta == 0:
            duplicate += 1
        elif delta < 0:
            decreasing += 1
        else:
            minimum = delta if minimum is None else min(minimum, delta)
            maximum = delta if maximum is None else max(maximum, delta)

            if delta > gap_threshold_ms:
                gaps += 1

    return TimestampStats(
        transitions=max(0, len(records) - 1),
        duplicate=duplicate,
        decreasing=decreasing,
        gaps_over_threshold=gaps,
        minimum_positive_interval_ms=minimum,
        maximum_positive_interval_ms=maximum,
    )


def build_quality_report(
    records: Sequence[SensorRecord],
    reports: Sequence[FileReport],
    gap_threshold_ms: float = 1000.0,
) -> dict[str, Any]:
    """Build a JSON-serializable, per-recording quality report."""

    if not 0 < gap_threshold_ms < float("inf"):
        raise ValueError("Gap threshold must be finite and positive")

    filenames = [report.filename for report in reports]

    if len(filenames) != len(set(filenames)):
        raise ValueError("Duplicate source filenames are ambiguous")

    groups: dict[str, list[SensorRecord]] = {
        name: [] for name in filenames
    }

    for record in records:
        if record.source_file not in groups:
            raise ValueError(
                f"Record has no matching file report: {record.source_file}"
            )
        groups[record.source_file].append(record)

    files: list[dict[str, Any]] = []
    total_labels: Counter[str] = Counter()
    total_duplicate = 0
    total_decreasing = 0
    total_gaps = 0

    for report in reports:
        group = groups[report.filename]

        if len(group) != report.valid:
            raise ValueError(f"Record count mismatch: {report.filename}")

        actual_labels = Counter(record.annotation for record in group)

        if actual_labels != report.labels:
            raise ValueError(f"Label count mismatch: {report.filename}")

        stats = analyze_timestamps(group, gap_threshold_ms)

        total_labels.update(actual_labels)
        total_duplicate += stats.duplicate
        total_decreasing += stats.decreasing
        total_gaps += stats.gaps_over_threshold

        files.append({
            "filename": report.filename,
            "valid": report.valid,
            "headers": report.headers,
            "invalid": report.invalid,
            "unknown_labels": report.unknown_labels,
            "labels": dict(sorted(actual_labels.items())),
            "rejection_lines": list(report.rejection_lines),
            "timestamps": asdict(stats),
        })

    labeled = sum(total_labels[label] for label in CLASSES)

    return {
        "schema_version": 1,
        "gap_threshold_ms": gap_threshold_ms,
        "interpretation": (
            "Gaps are interval observations, not proof of data loss."
        ),
        "summary": {
            "files": len(reports),
            "valid": sum(report.valid for report in reports),
            "headers": sum(report.headers for report in reports),
            "invalid": sum(report.invalid for report in reports),
            "labeled": labeled,
            "unannotated": total_labels["NAN"],
            "duplicate_timestamps": total_duplicate,
            "decreasing_timestamps": total_decreasing,
            "gaps_over_threshold": total_gaps,
            "labels": dict(sorted(total_labels.items())),
        },
        "files": files,
    }
