"""Tests for grouped evaluation partition acceptance."""

import unittest

from adas.evaluation.partition_contract import (
    validate_partition_class_coverage,
)


class PartitionContractTests(unittest.TestCase):

    def setUp(self):
        self.classes = ("normal", "rash")
        self.counts = {
            "A": {"normal": 20, "rash": 10},
            "B": {"normal": 15, "rash": 12},
            "C": {"normal": 18, "rash": 14},
        }
        self.parts = {
            "train": ["A"],
            "validation": ["B"],
            "test": ["C"],
        }

    def check(self, partitions=None, counts=None, minimum=10):
        return validate_partition_class_coverage(
            self.parts if partitions is None else partitions,
            self.counts if counts is None else counts,
            self.classes,
            minimum_test_windows=minimum,
        )

    def test_accepts_valid_partition(self):
        result = self.check()
        self.assertEqual(result["test"]["rash"], 14)

    def test_deterministic_result(self):
        self.assertEqual(self.check(), self.check())

    def test_rejects_group_overlap(self):
        parts = dict(self.parts, test=["A"])
        with self.assertRaises(ValueError):
            self.check(partitions=parts)

    def test_rejects_missing_group(self):
        parts = dict(self.parts, validation=["A"])
        with self.assertRaises(ValueError):
            self.check(partitions=parts)

    def test_rejects_missing_class(self):
        counts = dict(self.counts, B={"normal": 15})
        with self.assertRaises(ValueError):
            self.check(counts=counts)

    def test_rejects_insufficient_test_support(self):
        with self.assertRaises(ValueError):
            self.check(minimum=15)

    def test_rejects_negative_counts(self):
        counts = dict(self.counts, A={"normal": -1, "rash": 10})
        with self.assertRaises(ValueError):
            self.check(counts=counts)

    def test_rejects_boolean_counts(self):
        counts = dict(self.counts, A={"normal": True, "rash": 10})
        with self.assertRaises(ValueError):
            self.check(counts=counts)

    def test_rejects_unknown_class(self):
        counts = dict(
            self.counts,
            A={"normal": 20, "rash": 10, "unknown": 1},
        )
        with self.assertRaises(ValueError):
            self.check(counts=counts)

    def test_rejects_missing_partition(self):
        parts = {"train": ["A"], "test": ["B", "C"]}
        with self.assertRaises(ValueError):
            self.check(partitions=parts)

    def test_rejects_invalid_minimum(self):
        with self.assertRaises(ValueError):
            self.check(minimum=0)

    def test_rejects_zero_window_recording(self):
        counts = dict(self.counts, A={"normal": 0, "rash": 0})
        with self.assertRaises(ValueError):
            self.check(counts=counts)


if __name__ == "__main__":
    unittest.main()
