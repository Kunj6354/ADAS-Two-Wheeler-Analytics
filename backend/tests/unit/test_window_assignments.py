"""Tests for canonical recording-group window assignment."""

from dataclasses import dataclass, replace
import unittest

from adas.evaluation.window_assignments import (
    validate_window_assignments,
    window_identity,
)


@dataclass(frozen=True)
class FakeWindow:
    source_file: str
    parent_segment_id: str
    stretch_id: str
    window_index: int
    source_start_line: int
    source_end_line: int
    label: str


class WindowAssignmentTests(unittest.TestCase):

    def setUp(self):
        self.a1 = FakeWindow(
            "MAIN9.CSV", "p1", "s1", 0, 10, 60, "normal_driving"
        )
        self.a2 = FakeWindow(
            "MAIN9.CSV", "p1", "s1", 1, 35, 85, "normal_driving"
        )
        self.b1 = FakeWindow(
            "MAIN17.CSV", "p2", "s2", 0, 20, 70, "rash_driving"
        )
        self.canonical = [self.a1, self.a2, self.b1]

    def test_valid_complete_partition(self):
        result = validate_window_assignments(
            {
                "train": [self.a1, self.a2],
                "test": [self.b1],
            },
            self.canonical,
        )
        self.assertEqual(result["train"], (2, 1))
        self.assertEqual(result["test"], (1, 1))

    def test_identity_contains_recording(self):
        self.assertEqual(window_identity(self.a1)[0], "MAIN9.CSV")

    def test_rejects_cross_partition_recording(self):
        with self.assertRaises(ValueError):
            validate_window_assignments(
                {
                    "train": [self.a1, self.b1],
                    "test": [self.a2],
                },
                self.canonical,
            )

    def test_rejects_duplicate_window(self):
        with self.assertRaises(ValueError):
            validate_window_assignments(
                {
                    "train": [self.a1, self.a2, self.a1],
                    "test": [self.b1],
                },
                self.canonical,
            )

    def test_rejects_missing_window(self):
        with self.assertRaises(ValueError):
            validate_window_assignments(
                {"train": [self.a1], "test": [self.b1]},
                self.canonical,
            )

    def test_rejects_unexpected_window(self):
        extra = replace(self.a1, window_index=999)
        with self.assertRaises(ValueError):
            validate_window_assignments(
                {"train": [self.a1, self.a2, extra],
                 "test": [self.b1]},
                self.canonical,
            )

    def test_rejects_forged_source_lines(self):
        forged = replace(self.a2, source_start_line=999)
        with self.assertRaises(ValueError):
            validate_window_assignments(
                {"train": [self.a1, forged], "test": [self.b1]},
                self.canonical,
            )

    def test_rejects_forged_label(self):
        forged = replace(self.a2, label="rash_turn")
        with self.assertRaises(ValueError):
            validate_window_assignments(
                {"train": [self.a1, forged], "test": [self.b1]},
                self.canonical,
            )

    def test_rejects_duplicate_canonical_identity(self):
        with self.assertRaises(ValueError):
            validate_window_assignments(
                {"train": [self.a1, self.a2], "test": [self.b1]},
                self.canonical + [self.a1],
            )

    def test_rejects_empty_partition(self):
        with self.assertRaises(ValueError):
            validate_window_assignments(
                {"train": self.canonical, "test": []},
                self.canonical,
            )

    def test_rejects_empty_canonical_collection(self):
        with self.assertRaises(ValueError):
            validate_window_assignments(
                {"train": [self.a1]}, [],
            )

    def test_rejects_invalid_source(self):
        invalid = replace(self.a1, source_file="")
        with self.assertRaises(ValueError):
            validate_window_assignments(
                {"train": [invalid, self.a2],
                 "test": [self.b1]},
                self.canonical,
            )


if __name__ == "__main__":
    unittest.main()
