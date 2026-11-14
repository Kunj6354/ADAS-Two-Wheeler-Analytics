"""Unit tests for historical ADAS CSV ingestion."""

import tempfile
import unittest
from pathlib import Path

from adas.data.loader import COLUMNS, load_dataset, load_recording


class TestHistoricalCSVLoader(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.header = ",".join(COLUMNS)

    def write_csv(self, name, lines):
        path = self.root / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_header_only_file(self):
        path = self.write_csv("MAIN1.CSV", [self.header])
        records, report = load_recording(path)

        self.assertEqual(len(records), 0)
        self.assertEqual(report.headers, 1)
        self.assertEqual(report.invalid, 0)

    def test_headerless_file(self):
        path = self.write_csv(
            "MAIN39.CSV",
            ["1000,1,2,3,4,5,6,normal_driving"],
        )
        records, report = load_recording(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(report.headers, 0)
        self.assertEqual(records[0].annotation, "normal_driving")

    def test_trailing_header(self):
        path = self.write_csv(
            "MAIN2.CSV",
            [
                "1000,1,2,3,4,5,6,rash_driving",
                self.header,
            ],
        )
        records, report = load_recording(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(report.headers, 1)

    def test_header_between_records(self):
        path = self.write_csv(
            "MAIN3.CSV",
            [
                "1000,1,2,3,4,5,6,normal_turn",
                self.header,
                "1040,1,2,3,4,5,6,rash_turn",
            ],
        )
        records, report = load_recording(path)

        self.assertEqual(len(records), 2)
        self.assertEqual(report.headers, 1)
        self.assertEqual(records[1].source_line, 3)

    def test_nan_is_unannotated_label(self):
        path = self.write_csv(
            "MAIN4.CSV",
            ["1000,1,2,3,4,5,6,NAN"],
        )
        records, report = load_recording(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].annotation, "NAN")
        self.assertEqual(report.labels["NAN"], 1)

    def test_invalid_numeric_is_rejected(self):
        path = self.write_csv(
            "MAIN5.CSV",
            ["1000,1,bad,3,4,5,6,normal_driving"],
        )
        records, report = load_recording(path)

        self.assertEqual(len(records), 0)
        self.assertEqual(report.invalid, 1)
        self.assertEqual(report.rejection_lines, [1])

    def test_nonfinite_numeric_is_rejected(self):
        path = self.write_csv(
            "MAIN6.CSV",
            [
                "1000,nan,2,3,4,5,6,normal_driving",
                "1040,inf,2,3,4,5,6,normal_driving",
            ],
        )
        records, report = load_recording(path)

        self.assertEqual(len(records), 0)
        self.assertEqual(report.invalid, 2)

    def test_incorrect_column_count(self):
        path = self.write_csv(
            "MAIN7.CSV",
            ["1000,1,2,3,normal_driving"],
        )
        records, report = load_recording(path)

        self.assertEqual(len(records), 0)
        self.assertEqual(report.invalid, 1)

    def test_unknown_label_rejected(self):
        path = self.write_csv(
            "MAIN8.CSV",
            ["1000,1,2,3,4,5,6,unexpected_label"],
        )
        records, report = load_recording(path)

        self.assertEqual(len(records), 0)
        self.assertEqual(report.unknown_labels, 1)
        self.assertEqual(report.invalid, 1)

    def test_source_provenance(self):
        path = self.write_csv(
            "MAIN9.CSV",
            [
                self.header,
                "1000,1,2,3,4,5,6,normal_driving",
            ],
        )
        records, report = load_recording(path)

        self.assertEqual(report.valid, 1)
        self.assertEqual(records[0].source_file, "MAIN9.CSV")
        self.assertEqual(records[0].source_line, 2)
        self.assertEqual(records[0].timestamp_ms, 1000.0)

    def test_numeric_filename_order(self):
        self.write_csv(
            "MAIN10.CSV",
            ["1000,1,2,3,4,5,6,normal_driving"],
        )
        self.write_csv(
            "MAIN2.CSV",
            ["2000,1,2,3,4,5,6,rash_turn"],
        )

        records, reports = load_dataset(self.root)

        self.assertEqual(
            [report.filename for report in reports],
            ["MAIN2.CSV", "MAIN10.CSV"],
        )
        self.assertEqual(records[0].source_file, "MAIN2.CSV")

    def test_missing_recording_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            load_recording(self.root / "MISSING.CSV")

    def test_source_file_remains_unchanged(self):
        path = self.write_csv(
            "MAIN11.CSV",
            [
                "1000,1,2,3,4,5,6,normal_driving",
                self.header,
            ],
        )
        before = path.read_bytes()
        load_recording(path)
        after = path.read_bytes()

        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
