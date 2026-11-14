"""Tests for the adaptive stretch interpolation adapter."""

import unittest
from dataclasses import replace

from adas.data.loader import SensorRecord
from adas.signal.adaptive_segmentation import split_timing_stalls
from adas.signal.segmentation import segment_records
from adas.signal.stretch_resampling import interpolate_stretch


def make_stretch(count=120):
    records = [
        SensorRecord(
            source_file="MAIN17.CSV",
            source_line=i + 10,
            timestamp_ms=float(i * 50),
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
    parents = segment_records(records)
    return split_timing_stalls(parents)[0]


class TestStretchResampling(unittest.TestCase):

    def test_adapter_produces_uniform_grid(self):
        result = interpolate_stretch(make_stretch(), 20.0)
        self.assertEqual(len(result.signal.times_ms), 120)
        self.assertAlmostEqual(result.signal.times_ms[1], 50.0)

    def test_preserves_parent_and_stretch_identity(self):
        stretch = make_stretch()
        result = interpolate_stretch(stretch, 20.0)

        self.assertEqual(result.stretch_id, stretch.stretch_id)
        self.assertEqual(
            result.parent_segment_id, stretch.parent_segment_id
        )
        self.assertEqual(result.signal.source_start_line, 10)
        self.assertEqual(result.signal.source_end_line, 129)

    def test_explicitly_not_ml_approved(self):
        result = interpolate_stretch(make_stretch(), 20.0)
        self.assertFalse(result.anti_aliased)
        self.assertFalse(result.approved_for_ml)

    def test_rejects_short_stretch(self):
        with self.assertRaises(ValueError):
            interpolate_stretch(make_stretch(20), 20.0)

    def test_rejects_inconsistent_metadata(self):
        stretch = make_stretch()
        broken = replace(stretch, source_file="OTHER.CSV")
        with self.assertRaises(ValueError):
            interpolate_stretch(broken, 20.0)


if __name__ == "__main__":
    unittest.main()
