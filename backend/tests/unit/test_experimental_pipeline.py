"""Synthetic tests for the experimental preprocessing pipeline."""

import math
import unittest

from adas.data.loader import SensorRecord
from adas.signal.segmentation import segment_records
from adas.signal.adaptive_segmentation import split_timing_stalls
from adas.signal.experimental_pipeline import process_stretch


def make_stretch(frequency_hz=3.0, source_hz=40.0, count=400):
    records = []

    for i in range(count):
        time_s = i / source_hz
        records.append(SensorRecord(
            source_file="SYNTHETIC.CSV",
            source_line=i + 1,
            timestamp_ms=time_s * 1000,
            acc_x=math.cos(2 * math.pi * frequency_hz * time_s),
            acc_y=4.0,
            acc_z=0.0,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            annotation="NAN",
        ))

    return split_timing_stalls(segment_records(records))[0]


def rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


class TestExperimentalPipeline(unittest.TestCase):

    def test_constant_channel_preserved(self):
        stretch = make_stretch()
        result = process_stretch(stretch, channel="acc_y")
        self.assertTrue(
            all(abs(value - 4.0) < 1e-8 for value in result.values)
        )

    def test_high_frequency_suppressed(self):
        stretch = make_stretch(frequency_hz=15.0)
        result = process_stretch(stretch, num_taps=101)
        self.assertLess(rms(result.values), 0.05)

    def test_low_frequency_preserved(self):
        stretch = make_stretch(frequency_hz=3.0)
        result = process_stretch(stretch, num_taps=101)
        self.assertGreater(rms(result.values), 0.60)

    def test_metadata_and_ml_restriction(self):
        stretch = make_stretch()
        result = process_stretch(stretch)
        self.assertEqual(result.stretch_id, stretch.stretch_id)
        self.assertEqual(
            result.parent_segment_id, stretch.parent_segment_id
        )
        self.assertEqual(result.source_file, stretch.source_file)
        self.assertFalse(result.approved_for_ml)

    def test_rejects_invalid_cutoff(self):
        with self.assertRaises(ValueError):
            process_stretch(make_stretch(), cutoff_hz=10.0)

    def test_rejects_short_stretch(self):
        with self.assertRaises(ValueError):
            process_stretch(make_stretch(count=40))


if __name__ == "__main__":
    unittest.main()
