"""Tests for optional timing-stall segmentation."""

import unittest

from adas.data.loader import SensorRecord
from adas.signal.adaptive_segmentation import split_timing_stalls
from adas.signal.segmentation import segment_records


def sample(t, line):
    return SensorRecord(
        source_file="MAIN17.CSV",
        source_line=line,
        timestamp_ms=float(t),
        acc_x=1.0,
        acc_y=2.0,
        acc_z=3.0,
        gyro_x=4.0,
        gyro_y=5.0,
        gyro_z=6.0,
        annotation="normal_driving",
    )


class TestAdaptiveSegmentation(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(split_timing_stalls([]), [])

    def test_uniform_intervals_not_split(self):
        parent = segment_records([
            sample(0, 1),
            sample(40, 2),
            sample(80, 3),
            sample(120, 4),
        ])
        result = split_timing_stalls(parent)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].sample_count, 4)

    def test_isolated_timing_stall(self):
        original = [
            sample(0, 1),
            sample(40, 2),
            sample(80, 3),
            sample(300, 4),
            sample(340, 5),
            sample(380, 6),
        ]
        parent = segment_records(original)
        result = split_timing_stalls(parent)

        self.assertEqual([s.sample_count for s in result], [3, 3])
        self.assertEqual(result[1].start_reason, "timing_stall")
        self.assertEqual(
            [r for s in result for r in s.records],
            original,
        )

    def test_strict_threshold(self):
        parent = segment_records([
            sample(0, 1),
            sample(40, 2),
            sample(160, 3),
            sample(200, 4),
        ])
        result = split_timing_stalls(parent, interval_ratio=3)
        self.assertEqual(len(result), 1)

    def test_preserves_source_lines(self):
        parent = segment_records([
            sample(0, 10),
            sample(40, 11),
            sample(80, 12),
        ])
        result = split_timing_stalls(parent)
        self.assertEqual(result[0].start_line, 10)
        self.assertEqual(result[0].end_line, 12)

    def test_rejects_invalid_ratio(self):
        parent = segment_records([sample(0, 1), sample(40, 2)])
        for value in [0, -1, float("nan"), float("inf")]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    split_timing_stalls(parent, interval_ratio=value)


if __name__ == "__main__":
    unittest.main()
