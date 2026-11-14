"""Read historical ADAS CSV files without modifying source data."""

import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


COLUMNS = (
    "Timestamp", "AccX", "AccY", "AccZ",
    "GyroX", "GyroY", "GyroZ", "Annotation",
)

CLASSES = frozenset({
    "normal_driving",
    "rash_driving",
    "normal_turn",
    "rash_turn",
})

VALID_LABELS = CLASSES | {"NAN"}


@dataclass(frozen=True)
class SensorRecord:
    source_file: str
    source_line: int
    timestamp_ms: float
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    annotation: str


@dataclass
class FileReport:
    filename: str
    valid: int = 0
    headers: int = 0
    invalid: int = 0
    unknown_labels: int = 0
    labels: Counter = field(default_factory=Counter)
    rejection_lines: list[int] = field(default_factory=list)


def load_recording(path: Path) -> tuple[list[SensorRecord], FileReport]:
    """Load one recording and return accepted rows plus diagnostics."""

    path = Path(path)
    report = FileReport(filename=path.name)
    records: list[SensorRecord] = []

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)

        for line_number, row in enumerate(reader, start=1):
            values = [cell.strip() for cell in row]

            if tuple(values) == COLUMNS:
                report.headers += 1
                continue

            if len(values) != 8:
                report.invalid += 1
                report.rejection_lines.append(line_number)
                continue

            try:
                numbers = tuple(float(value) for value in values[:7])

                if not all(math.isfinite(value) for value in numbers):
                    raise ValueError("Non-finite measurement")

            except ValueError:
                report.invalid += 1
                report.rejection_lines.append(line_number)
                continue

            label = values[7]

            if label not in VALID_LABELS:
                report.unknown_labels += 1
                report.invalid += 1
                report.rejection_lines.append(line_number)
                continue

            record = SensorRecord(
                source_file=path.name,
                source_line=line_number,
                timestamp_ms=numbers[0],
                acc_x=numbers[1],
                acc_y=numbers[2],
                acc_z=numbers[3],
                gyro_x=numbers[4],
                gyro_y=numbers[5],
                gyro_z=numbers[6],
                annotation=label,
            )

            records.append(record)
            report.valid += 1
            report.labels[label] += 1

    return records, report


def load_dataset(directory: Path) -> tuple[list[SensorRecord], list[FileReport]]:
    """Read MAIN*.CSV recordings in numeric filename order."""

    directory = Path(directory)

    files = sorted(
        (
            path for path in directory.iterdir()
            if path.is_file()
            and path.name.upper().startswith("MAIN")
            and path.suffix.upper() == ".CSV"
            and path.stem[4:].isdigit()
        ),
        key=lambda path: int(path.stem[4:]),
    )

    records: list[SensorRecord] = []
    reports: list[FileReport] = []

    for path in files:
        file_records, report = load_recording(path)
        records.extend(file_records)
        reports.append(report)

    return records, reports
