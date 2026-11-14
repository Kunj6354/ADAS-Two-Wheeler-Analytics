"""Unit tests for timestamp-aware interpolation."""

import unittest

from adas.data.loader import SensorRecord
from adas.signal.segmentation import segment_records
from adas.signal.resampling import resample_segment


def make_segment(count=100, dt=50):
    records = [
        SensorRecord(
            source_file="MAIN9.CSV",
            source_line=i + 1,
            timestamp_ms=i * dt,
            acc_x=float(i),
            acc_y=2.0,
            acc_z=3.0,
            gyro_x=4.0,
            gyro_y=5.0,
            gyro_z=6.0,
            annotation="normal_driving",
        )
        for i in range(count)
    ]
    return segment_records(records)[0]


class TestResampling(unittest.TestCase):

    def test_uniform_grid(self):
        result = resample_segment(make_segment(), 20.0)
        self.assertEqual(len(result.times_ms), 100)
        self.assertAlmostEqual(result.times_ms[1], 50)
        self.assertEqual(result.channels["acc_x"][0], 0.0)
        self.assertEqual(result.channels["acc_x"][-1], 99.0)

    def test_linear_interpolation(self):
        segment = make_segment()
        result = resample_segment(segment, 10.0)
        self.assertAlmostEqual(result.channels["acc_x"][1], 2.0)

    def test_preserves_source_metadata(self):
        result = resample_segment(make_segment(), 20)
        self.assertEqual(result.source_file, "MAIN9.CSV")
        self.assertEqual(result.source_start_line, 1)
        self.assertEqual(result.source_end_line, 100)

    def test_rejects_short_segment(self):
        with self.assertRaises(ValueError):
            resample_segment(make_segment(count=10), 20)

    def test_rejects_upsampling(self):
        with self.assertRaises(ValueError):
            resample_segment(make_segment(), 30)

    def test_rejects_invalid_rate(self):
        for rate in [0, -1, float("nan"), float("inf")]:
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    resample_segment(make_segment(), rate)

    def test_rejects_irregular_large_interval(self):
        segment = make_segment(count=100)
        records = list(segment.records)
        changed = [
            SensorRecord(
                source_file=r.source_file,
                source_line=r.source_line,
                timestamp_ms=(
                    r.timestamp_ms + 400
                    if i >= 50 else r.timestamp_ms
                ),
                acc_x=r.acc_x,
                acc_y=r.acc_y,
                acc_z=r.acc_z,
                gyro_x=r.gyro_x,
                gyro_y=r.gyro_y,
                gyro_z=r.gyro_z,
                annotation=r.annotation,
            )
            for i, r in enumerate(records)
        ]
        altered = segment_records(changed)[0]
        with self.assertRaises(ValueError):
            resample_segment(altered, 20)


if __name__ == "__main__":
    unittest.main()
