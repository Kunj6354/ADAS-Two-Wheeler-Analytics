"""Synthetic tests for the M06 descriptive quality validator."""

import tempfile
import unittest
from pathlib import Path

from adas.data.m06_quality import inspect_recording


HEADER = "Timestamp,AccX,AccY,AccZ,GyroX,GyroY,GyroZ,Annotation\n"


class M06QualityTests(unittest.TestCase):
    def inspect(self, content, **options):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            path.write_text(content, encoding="utf-8")
            return inspect_recording(path, **options)

    def test_valid_header_and_rows(self):
        report = self.inspect(
            HEADER
            + "100,0,0,1,0,0,0,normal_turn\n"
            + "150,0.1,0,1,0,0,0,normal_turn\n"
            + "200,0,0,1,0,0,0,NAN\n"
        )
        self.assertEqual(report.status, "STRUCTURAL_PASS")
        self.assertEqual(report.rows, 3)
        self.assertEqual(report.header_count, 1)
        self.assertEqual(report.median_interval_ms, 50)
        self.assertEqual(report.short_labeled_runs, 1)

    def test_headerless_csv_is_accepted(self):
        report = self.inspect("100,0,0,1,0,0,0,rash_turn\n")
        self.assertEqual(report.status, "STRUCTURAL_PASS")
        self.assertEqual(report.header_count, 0)

    def test_nonfinite_sensor_rejected(self):
        report = self.inspect(
            "100,nan,0,1,0,0,0,normal_driving\n"
        )
        self.assertEqual(report.status, "STRUCTURAL_FAIL")
        self.assertEqual(report.invalid_rows[0][1], "NONFINITE_SENSOR")

    def test_duplicate_timestamp_rejected(self):
        report = self.inspect(
            "100,0,0,1,0,0,0,normal_driving\n"
            "100,0,0,1,0,0,0,rash_driving\n"
        )
        self.assertEqual(report.status, "STRUCTURAL_FAIL")
        self.assertEqual(
            report.invalid_rows[0][1], "NONINCREASING_TIMESTAMP"
        )

    def test_column_error_rejected(self):
        report = self.inspect("100,0,0,1,NAN\n")
        self.assertEqual(report.status, "STRUCTURAL_FAIL")

    def test_unknown_label_rejected(self):
        report = self.inspect("100,0,0,1,0,0,0,other\n")
        self.assertEqual(
            report.invalid_rows[0][1], "INVALID_ANNOTATION"
        )

    def test_nan_label_is_not_numeric_nan(self):
        report = self.inspect("100,0,0,1,0,0,0,NAN\n")
        self.assertEqual(report.status, "STRUCTURAL_PASS")
        self.assertEqual(report.label_counts["NAN"], 1)
        self.assertEqual(report.short_labeled_runs, 0)

    def test_misplaced_header_rejected(self):
        report = self.inspect(
            "100,0,0,1,0,0,0,normal_turn\n"
            + HEADER
            + "200,0,0,1,0,0,0,normal_turn\n"
        )
        self.assertEqual(report.status, "STRUCTURAL_FAIL")
        self.assertIn(
            (2, "MISPLACED_HEADER"), report.invalid_rows
        )

    def test_single_trailing_header_is_accepted_and_flagged(self):
        report = self.inspect(
            "100,0,0,1,0,0,0,normal_turn\n"
            "150,0,0,1,0,0,0,normal_turn\n"
            + HEADER
        )
        self.assertEqual(report.status, "STRUCTURAL_PASS")
        self.assertEqual(report.rows, 2)
        self.assertEqual(report.header_count, 1)
        self.assertEqual(report.trailing_header_count, 1)
        self.assertEqual(report.invalid_rows, ())

    def test_repeated_headers_rejected(self):
        report = self.inspect(
            HEADER
            + "100,0,0,1,0,0,0,normal_turn\n"
            + HEADER
        )
        self.assertEqual(report.status, "STRUCTURAL_FAIL")
        self.assertTrue(any(
            code == "MISPLACED_HEADER"
            for _, code in report.invalid_rows
        ))

    def test_empty_header_only_recording(self):
        report = self.inspect(HEADER)
        self.assertEqual(report.status, "STRUCTURAL_PASS")
        self.assertEqual(report.rows, 0)

    def test_threshold_is_diagnostic(self):
        content = (
            "100,0,0,1,0,0,0,normal_turn\n"
            "250,0,0,1,0,0,0,normal_turn\n"
        )
        self.assertEqual(
            self.inspect(content, short_run_threshold_ms=200)
            .short_labeled_runs, 1
        )
        self.assertEqual(
            self.inspect(content, short_run_threshold_ms=100)
            .short_labeled_runs, 0
        )


if __name__ == "__main__":
    unittest.main()
