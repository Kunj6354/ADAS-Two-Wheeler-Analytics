"""Unit tests for historical dataset quality reporting."""

import tempfile
import unittest
from pathlib import Path

from adas.data.loader import load_recording
from adas.reporting.quality import (
    analyze_timestamps,
    build_quality_report,
)


class TestQualityReporting(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "MAIN2.CSV"

    def load(self, lines):
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return load_recording(self.path)

    def test_empty_recording(self):
        records, report = self.load([])
        result = build_quality_report(records, [report])
        self.assertEqual(result["summary"]["valid"], 0)
        self.assertEqual(result["summary"]["gaps_over_threshold"], 0)

    def test_normal_intervals(self):
        records, report = self.load([
            "1000,1,2,3,4,5,6,normal_driving",
            "1040,1,2,3,4,5,6,normal_driving",
            "1080,1,2,3,4,5,6,NAN",
        ])
        result = build_quality_report(records, [report])
        self.assertEqual(result["summary"]["valid"], 3)
        self.assertEqual(result["summary"]["labeled"], 2)
        self.assertEqual(result["summary"]["unannotated"], 1)
        self.assertEqual(result["summary"]["gaps_over_threshold"], 0)

    def test_duplicate_and_decreasing(self):
        records, report = self.load([
            "1000,1,2,3,4,5,6,normal_driving",
            "1000,1,2,3,4,5,6,normal_driving",
            "900,1,2,3,4,5,6,normal_driving",
        ])
        result = build_quality_report(records, [report])
        self.assertEqual(result["summary"]["duplicate_timestamps"], 1)
        self.assertEqual(result["summary"]["decreasing_timestamps"], 1)

    def test_gap_threshold_is_strict(self):
        records, report = self.load([
            "1000,1,2,3,4,5,6,normal_driving",
            "2000,1,2,3,4,5,6,normal_driving",
            "3001,1,2,3,4,5,6,normal_driving",
        ])
        result = build_quality_report(records, [report])
        self.assertEqual(result["summary"]["gaps_over_threshold"], 1)

    def test_invalid_gap_threshold(self):
        records, _ = self.load([])
        with self.assertRaises(ValueError):
            analyze_timestamps(records, 0)
        with self.assertRaises(ValueError):
            analyze_timestamps(records, float("nan"))

    def test_report_rejects_missing_source(self):
        records, report = self.load([
            "1000,1,2,3,4,5,6,normal_driving",
        ])
        report.filename = "OTHER.CSV"
        with self.assertRaises(ValueError):
            build_quality_report(records, [report])


if __name__ == "__main__":
    unittest.main()
