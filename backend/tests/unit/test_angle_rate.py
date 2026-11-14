"""Tests for rates derived from integrated gyroscope angles."""

import unittest
from dataclasses import replace

from adas.data.loader import SensorRecord
from adas.features.angle_rate import (
    RATE_FEATURE_NAMES,
    extract_angle_rate_features,
)
from adas.features.windowing import build_windows
from adas.signal.segmentation import segment_records
from adas.signal.adaptive_segmentation import split_timing_stalls


def make_window(label="normal_driving"):
    records = []

    for i in range(100):
        seconds = i * 0.04

        records.append(SensorRecord(
            source_file="SYNTHETIC.CSV",
            source_line=i + 1,
            timestamp_ms=seconds * 1000,
            acc_x=0.0,
            acc_y=0.0,
            acc_z=1.0,
            gyro_x=10.0 * seconds,
            gyro_y=-20.0 * seconds,
            gyro_z=5.0,
            annotation=label,
        ))

    stretch = split_timing_stalls(
        segment_records(records)
    )[0]

    return build_windows([stretch])[0]


class TestAngleRate(unittest.TestCase):

    def test_schema(self):
        self.assertEqual(len(RATE_FEATURE_NAMES), 9)
        self.assertEqual(len(set(RATE_FEATURE_NAMES)), 9)

    def test_known_constant_rates(self):
        row = extract_angle_rate_features(make_window())
        features = row.as_feature_dict()

        self.assertAlmostEqual(
            features["gyro_x_derived_rate_mean"], 10.0
        )
        self.assertAlmostEqual(
            features["gyro_y_derived_rate_mean"], -20.0
        )
        self.assertAlmostEqual(
            features["gyro_z_derived_rate_mean"], 0.0
        )

    def test_rms_and_absolute_peak(self):
        features = extract_angle_rate_features(
            make_window()
        ).as_feature_dict()

        self.assertAlmostEqual(
            features["gyro_x_derived_rate_rms"], 10.0
        )
        self.assertAlmostEqual(
            features["gyro_y_derived_rate_max_abs"], 20.0
        )

    def test_unannotated_rejected(self):
        with self.assertRaises(ValueError):
            extract_angle_rate_features(
                make_window("NAN")
            )

    def test_zero_interval_rejected(self):
        window = make_window()
        records = list(window.records)
        records[1] = replace(
            records[1],
            timestamp_ms=records[0].timestamp_ms,
        )
        broken = replace(window, records=tuple(records))

        with self.assertRaises(ValueError):
            extract_angle_rate_features(broken)

    def test_large_gap_rejected(self):
        window = make_window()
        records = list(window.records)

        # Shift later timestamps to create a genuine 200 ms gap.
        # Normal sample spacing is 40 ms; threshold is 120 ms.
        records = [
            replace(
                record,
                timestamp_ms=record.timestamp_ms + 160.0,
            )
            if index >= 25 else record
            for index, record in enumerate(records)
        ]
        broken = replace(window, records=tuple(records))

        with self.assertRaises(ValueError):
            extract_angle_rate_features(broken)

    def test_provenance_and_ml_guard(self):
        window = make_window()
        row = extract_angle_rate_features(window)

        self.assertEqual(row.source_file, window.source_file)
        self.assertEqual(row.stretch_id, window.stretch_id)
        self.assertEqual(row.window_index, window.window_index)
        self.assertFalse(row.approved_for_ml)

    def test_deterministic(self):
        window = make_window()
        self.assertEqual(
            extract_angle_rate_features(window),
            extract_angle_rate_features(window),
        )


if __name__ == "__main__":
    unittest.main()
