"""Unit tests for six-channel statistical feature extraction."""

import math
import unittest
from dataclasses import replace

from adas.data.loader import SensorRecord
from adas.features.extraction import (
    CHANNELS,
    FEATURE_NAMES,
    extract_labeled_windows,
    extract_window,
)
from adas.features.windowing import build_windows
from adas.signal.adaptive_segmentation import split_timing_stalls
from adas.signal.segmentation import segment_records


def make_window(label="normal_driving", count=100):
    records = [
        SensorRecord(
            source_file="MAIN17.CSV",
            source_line=i + 1,
            timestamp_ms=float(i * 40),
            acc_x=float(i % 2),
            acc_y=2.0,
            acc_z=3.0,
            gyro_x=4.0,
            gyro_y=5.0,
            gyro_z=6.0,
            annotation=label,
        )
        for i in range(count)
    ]

    stretches = split_timing_stalls(segment_records(records))
    return build_windows(stretches)[0]


class TestFeatureExtraction(unittest.TestCase):

    def test_feature_schema(self):
        self.assertEqual(len(CHANNELS), 6)
        self.assertEqual(len(FEATURE_NAMES), 30)
        self.assertEqual(len(set(FEATURE_NAMES)), 30)

    def test_output_feature_count(self):
        row = extract_window(make_window())
        self.assertEqual(len(row.values), 30)
        self.assertEqual(len(row.as_feature_dict()), 30)

    def test_binary_channel_statistics(self):
        features = extract_window(make_window()).as_feature_dict()
        self.assertAlmostEqual(features["acc_x_mean"], 0.5)
        self.assertAlmostEqual(features["acc_x_std"], 0.5)
        self.assertAlmostEqual(
            features["acc_x_rms"], math.sqrt(0.5)
        )
        self.assertEqual(features["acc_x_min"], 0.0)
        self.assertEqual(features["acc_x_max"], 1.0)

    def test_constant_channel_statistics(self):
        features = extract_window(make_window()).as_feature_dict()
        self.assertEqual(features["acc_y_mean"], 2.0)
        self.assertEqual(features["acc_y_std"], 0.0)
        self.assertEqual(features["acc_y_rms"], 2.0)

    def test_all_six_channels_present(self):
        features = extract_window(make_window()).as_feature_dict()
        for channel in CHANNELS:
            self.assertIn(f"{channel}_mean", features)
            self.assertIn(f"{channel}_std", features)

    def test_provenance_preserved(self):
        window = make_window()
        row = extract_window(window)
        self.assertEqual(row.source_file, window.source_file)
        self.assertEqual(row.stretch_id, window.stretch_id)
        self.assertEqual(row.window_index, window.window_index)
        self.assertEqual(row.source_start_line, 1)
        self.assertEqual(row.source_end_line, 50)

    def test_unannotated_rejected(self):
        window = make_window("NAN")
        with self.assertRaises(ValueError):
            extract_window(window)

    def test_invalid_sensor_value_rejected(self):
        window = make_window()
        records = list(window.records)
        records[0] = replace(records[0], acc_x=float("nan"))
        broken = replace(window, records=tuple(records))

        with self.assertRaises(ValueError):
            extract_window(broken)

    def test_extraction_is_deterministic(self):
        window = make_window()
        self.assertEqual(
            extract_window(window),
            extract_window(window),
        )

    def test_batch_excludes_unannotated(self):
        labeled = make_window()
        unannotated = make_window("NAN")
        rows = extract_labeled_windows([unannotated, labeled])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].label, "normal_driving")


if __name__ == "__main__":
    unittest.main()
