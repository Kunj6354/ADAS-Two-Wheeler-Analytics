"""Tests for provenance-aware feature windowing."""

import unittest
from dataclasses import replace

from adas.data.loader import SensorRecord
from adas.features.windowing import build_windows, classify_window
from adas.signal.adaptive_segmentation import split_timing_stalls
from adas.signal.segmentation import segment_records


def make_stretch(labels=None, count=100):
    if labels is None:
        labels = ["normal_driving"] * count

    records = [
        SensorRecord(
            source_file="MAIN17.CSV",
            source_line=i + 1,
            timestamp_ms=float(i * 40),
            acc_x=float(i),
            acc_y=2.0,
            acc_z=3.0,
            gyro_x=4.0,
            gyro_y=5.0,
            gyro_z=6.0,
            annotation=labels[i],
        )
        for i in range(count)
    ]

    return split_timing_stalls(segment_records(records))[0]


class TestWindowing(unittest.TestCase):

    def test_pure_supervised_window(self):
        windows = build_windows([make_stretch()])
        self.assertEqual(len(windows), 2)
        self.assertTrue(
            all(w.status == "PURE_LABELED" for w in windows)
        )
        self.assertTrue(
            all(w.label == "normal_driving" for w in windows)
        )

    def test_unannotated_is_not_supervised(self):
        stretch = make_stretch(["NAN"] * 100)
        windows = build_windows([stretch])
        self.assertTrue(
            all(w.status == "UNANNOTATED" for w in windows)
        )
        self.assertTrue(all(w.label is None for w in windows))

    def test_mixed_classes(self):
        labels = ["normal_driving"] * 25 + ["rash_driving"] * 75
        status, label = classify_window(
            make_stretch(labels).records[:50]
        )
        self.assertEqual(status, "MIXED_CLASSES")
        self.assertIsNone(label)

    def test_mixed_with_unannotated(self):
        labels = ["normal_driving"] * 25 + ["NAN"] * 75
        status, label = classify_window(
            make_stretch(labels).records[:50]
        )
        self.assertEqual(status, "MIXED_WITH_UNANNOTATED")
        self.assertIsNone(label)

    def test_too_few_samples(self):
        status, label = classify_window(
            make_stretch().records[:10]
        )
        self.assertEqual(status, "TOO_FEW_SAMPLES")
        self.assertIsNone(label)

    def test_half_open_window_boundaries(self):
        windows = build_windows([make_stretch()])
        self.assertEqual(windows[0].sample_count, 50)
        self.assertEqual(windows[0].start_ms, 0.0)
        self.assertEqual(windows[0].end_ms, 2000.0)
        self.assertEqual(windows[0].source_end_line, 50)

    def test_provenance(self):
        stretch = make_stretch()
        window = build_windows([stretch])[0]
        self.assertEqual(window.source_file, stretch.source_file)
        self.assertEqual(window.stretch_id, stretch.stretch_id)
        self.assertEqual(
            window.parent_segment_id, stretch.parent_segment_id
        )

    def test_no_cross_stretch_window(self):
        first = make_stretch()
        second = replace(
            first,
            stretch_id=2,
            source_file="MAIN18.CSV",
            records=tuple(
                replace(r, source_file="MAIN18.CSV")
                for r in first.records
            ),
        )
        windows = build_windows([first, second])
        self.assertEqual(len(windows), 4)
        self.assertEqual(
            {w.source_file for w in windows},
            {"MAIN17.CSV", "MAIN18.CSV"},
        )

    def test_invalid_configuration(self):
        stretch = make_stretch()
        with self.assertRaises(ValueError):
            build_windows([stretch], window_ms=0)
        with self.assertRaises(ValueError):
            build_windows([stretch], stride_ms=0)
        with self.assertRaises(ValueError):
            build_windows([stretch], min_samples=0)

    def test_deterministic_results(self):
        stretch = make_stretch()
        self.assertEqual(
            build_windows([stretch]),
            build_windows([stretch]),
        )


if __name__ == "__main__":
    unittest.main()
