"""Synthetic verification of the approved D4 representations."""

import math
import unittest
from dataclasses import replace

from adas.data.loader import SensorRecord
from adas.features.windowing import FeatureWindow
from adas.features.d4_representations import (
    APPROVED_VARIANTS,
    extract_d4_representations,
)


class D4RepresentationTests(unittest.TestCase):
    def make_window(self):
        rows = []

        for index, gyro in enumerate((10.0, 11.0, 13.0)):
            rows.append(SensorRecord(
                source_file="MAIN11.CSV",
                source_line=index + 1,
                timestamp_ms=float(index * 100),
                acc_x=float(index + 1),
                acc_y=2.0,
                acc_z=3.0,
                gyro_x=gyro,
                gyro_y=gyro + 20,
                gyro_z=gyro - 5,
                annotation="normal_driving",
            ))

        return FeatureWindow(
            source_file="MAIN11.CSV",
            parent_segment_id=636,
            stretch_id=648,
            window_index=1,
            start_ms=0.0,
            end_ms=1000.0,
            source_start_line=1,
            source_end_line=3,
            records=tuple(rows),
            status="PURE_LABELED",
            label="normal_driving",
        )

    def test_exact_approved_variant_set(self):
        rows = extract_d4_representations(self.make_window())

        self.assertEqual(
            tuple(row.representation for row in rows),
            APPROVED_VARIANTS,
        )
        self.assertEqual(
            tuple(len(row.values) for row in rows),
            (15, 30, 30),
        )

        for row in rows:
            self.assertEqual(
                row.identity,
                ("MAIN11.CSV", 636, 648, 1),
            )
            self.assertEqual(row.group, "MAIN11.CSV")
            self.assertEqual(row.label, "normal_driving")

    def test_relative_angle_calculation(self):
        rows = extract_d4_representations(self.make_window())
        accel, relative, raw = rows

        self.assertEqual(relative.values[:15], accel.values)
        self.assertEqual(raw.values[:15], accel.values)

        # Gyro X: [10, 11, 13] -> [0, 1, 3].
        expected_mean = 4.0 / 3.0
        expected_rms = math.sqrt(10.0 / 3.0)

        self.assertAlmostEqual(relative.values[15], expected_mean)
        self.assertAlmostEqual(relative.values[17], expected_rms)
        self.assertEqual(relative.values[18:20], (0.0, 3.0))
        self.assertAlmostEqual(raw.values[15], 34.0 / 3.0)

        # Relative Y and Z also start from zero.
        self.assertEqual(relative.values[23], 0.0)
        self.assertEqual(relative.values[28], 0.0)

    def test_repeated_extraction_and_no_mutation(self):
        window = self.make_window()
        original_records = window.records

        first = extract_d4_representations(window)
        second = extract_d4_representations(window)

        self.assertEqual(first, second)
        self.assertEqual(window.records, original_records)

    def test_reject_non_pure_window(self):
        window = replace(
            self.make_window(),
            status="MIXED_CLASSES",
            label=None,
        )

        with self.assertRaises(ValueError):
            extract_d4_representations(window)


if __name__ == "__main__":
    unittest.main()
