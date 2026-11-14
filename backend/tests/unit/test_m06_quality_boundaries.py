"""M06 synthetic structural and diagnostic boundary tests."""

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from adas.data.m06_quality import inspect_recording


HEADER = "Timestamp,AccX,AccY,AccZ,GyroX,GyroY,GyroZ,Annotation\n"


def row(timestamp, label="normal_driving", acc="0"):
    return f"{timestamp},{acc},0,1,0,0,0,{label}\n"


class M06QualityBoundaryTests(unittest.TestCase):

    def inspect(self, content, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            path.write_text(content, encoding="utf-8")
            before = sha256(path.read_bytes()).hexdigest()
            report = inspect_recording(path, **kwargs)
            self.assertEqual(
                sha256(path.read_bytes()).hexdigest(), before
            )
            return report

    def assert_error(self, content, code):
        report = self.inspect(content)
        self.assertEqual(report.status, "STRUCTURAL_FAIL")
        self.assertIn(
            code, {item[1] for item in report.invalid_rows}
        )

    def test_empty_file(self):
        report = self.inspect("")
        self.assertEqual(report.status, "STRUCTURAL_PASS")
        self.assertEqual(report.rows, 0)
        self.assertIsNone(report.median_interval_ms)

    def test_negative_timestamp_is_parseable(self):
        report = self.inspect(row(-10) + row(10))
        self.assertEqual(report.status, "STRUCTURAL_PASS")
        self.assertEqual(report.median_interval_ms, 20)

    def test_decreasing_timestamp_rejected(self):
        self.assert_error(
            row(100) + row(99), "NONINCREASING_TIMESTAMP"
        )

    def test_float_timestamp_rejected(self):
        self.assert_error(row("100.5"), "INVALID_TIMESTAMP")

    def test_missing_sensor_rejected(self):
        self.assert_error(row(100, acc=""), "INVALID_SENSOR")

    def test_infinite_sensor_rejected(self):
        self.assert_error(row(100, acc="inf"), "NONFINITE_SENSOR")

    def test_invalid_header_rejected(self):
        self.assert_error(
            "Timestamp,Wrong,AccY,AccZ,GyroX,GyroY,GyroZ,Annotation\n",
            "INVALID_HEADER",
        )

    def test_embedded_header_rejected(self):
        self.assert_error(
            row(100) + HEADER + row(200),
            "MISPLACED_HEADER",
        )

    def test_repeated_trailing_header_rejected(self):
        self.assert_error(
            row(100) + HEADER + HEADER,
            "MISPLACED_HEADER",
        )

    def test_trailing_header_with_blank_last_line(self):
        report = self.inspect(row(100) + HEADER + "\n")
        self.assertEqual(report.status, "STRUCTURAL_PASS")
        self.assertEqual(report.trailing_header_count, 1)

    def test_exact_duration_is_not_short(self):
        report = self.inspect(
            row(100, "normal_turn")
            + row(300, "normal_turn")
        )
        self.assertEqual(report.short_labeled_runs, 0)

    def test_short_run_is_diagnostic_only(self):
        report = self.inspect(
            row(100, "rash_turn") + row(101, "NAN")
        )
        self.assertEqual(report.status, "STRUCTURAL_PASS")
        self.assertEqual(report.short_labeled_runs, 1)

    def test_invalid_threshold_zero(self):
        with self.assertRaises(ValueError):
            self.inspect(row(100), short_run_threshold_ms=0)

    def test_invalid_threshold_boolean(self):
        with self.assertRaises(ValueError):
            self.inspect(row(100), short_run_threshold_ms=True)


if __name__ == "__main__":
    unittest.main()
