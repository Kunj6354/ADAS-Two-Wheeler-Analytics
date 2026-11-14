"""Unit tests for timestamp-aware segmentation."""

import unittest

from adas.data.loader import SensorRecord
from adas.signal.segmentation import segment_records


def sample(timestamp, source="MAIN2.CSV", line=1):
    return SensorRecord(
        source_file=source,
        source_line=line,
        timestamp_ms=timestamp,
        acc_x=1.0,
        acc_y=2.0,
        acc_z=3.0,
        gyro_x=4.0,
        gyro_y=5.0,
        gyro_z=6.0,
        annotation="normal_driving",
    )


class TestSegmentation(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(segment_records([]), [])

    def test_continuous_recording(self):
        segments = segment_records([
            sample(1000, line=1),
            sample(1040, line=2),
            sample(1080, line=3),
        ])
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].sample_count, 3)
        self.assertEqual(segments[0].start_reason, "recording_start")

    def test_large_gap(self):
        segments = segment_records([
            sample(1000),
            sample(2040),
        ])
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1].start_reason, "large_gap")

    def test_exact_threshold_does_not_split(self):
        segments = segment_records([
            sample(1000),
            sample(2000),
        ])
        self.assertEqual(len(segments), 1)

    def test_duplicate_timestamp(self):
        segments = segment_records([
            sample(1000),
            sample(1000),
        ])
        self.assertEqual(len(segments), 2)
        self.assertEqual(
            segments[1].start_reason,
            "duplicate_timestamp",
        )

    def test_decreasing_timestamp(self):
        segments = segment_records([
            sample(1000),
            sample(900),
        ])
        self.assertEqual(len(segments), 2)
        self.assertEqual(
            segments[1].start_reason,
            "decreasing_timestamp",
        )

    def test_source_change(self):
        segments = segment_records([
            sample(1000, "MAIN2.CSV"),
            sample(1040, "MAIN3.CSV"),
        ])
        self.assertEqual(len(segments), 2)
        self.assertEqual(
            segments[1].start_reason,
            "source_changed",
        )

    def test_preserves_record_provenance(self):
        original = [
            sample(1000, line=7),
            sample(1040, line=8),
        ]
        segments = segment_records(original)
        self.assertEqual(segments[0].start_line, 7)
        self.assertEqual(segments[0].end_line, 8)
        self.assertEqual(
            segments[0].records,
            tuple(original),
        )

    def test_invalid_threshold(self):
        for value in [0, -1, float("nan"), float("inf")]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    segment_records([], value)

    def test_all_samples_preserved(self):
        original = [
            sample(1000, line=1),
            sample(1040, line=2),
            sample(3000, line=3),
            sample(3040, line=4),
        ]
        segments = segment_records(original)
        restored = [
            record
            for segment in segments
            for record in segment.records
        ]
        self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()
