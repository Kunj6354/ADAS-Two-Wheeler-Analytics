"""Synthetic tests for independent grouped-fold validation."""

import unittest
from dataclasses import replace

from adas.evaluation.grouped_folds import (
    FoldLimits,
    validate_grouped_folds,
)


class GroupedFoldTests(unittest.TestCase):

    def setUp(self):
        self.classes = ("normal", "rash")
        self.counts = {
            name: {"normal": 20, "rash": 20}
            for name in "ABCDEFGH"
        }
        self.folds = {
            "fold_1": ("A", "B"),
            "fold_2": ("C", "D"),
            "fold_3": ("E", "F"),
            "fold_4": ("G", "H"),
        }
        self.limits = FoldLimits(groups_per_fold=2)

    def evaluate(self, folds=None, counts=None, limits=None):
        return validate_grouped_folds(
            self.folds if folds is None else folds,
            self.counts if counts is None else counts,
            self.classes,
            self.limits if limits is None else limits,
        )

    def test_valid(self):
        self.assertTrue(self.evaluate()["eligible"])

    def test_repeatable(self):
        self.assertEqual(self.evaluate(), self.evaluate())

    def test_group_order_independent(self):
        reversed_folds = {
            key: tuple(reversed(value))
            for key, value in self.folds.items()
        }
        self.assertEqual(
            self.evaluate()["eligible"],
            self.evaluate(folds=reversed_folds)["eligible"],
        )

    def test_rejects_cross_fold_reuse(self):
        folds = dict(self.folds, fold_4=("A", "H"))
        with self.assertRaises(ValueError):
            self.evaluate(folds=folds)

    def test_rejects_missing_recording(self):
        folds = dict(self.folds, fold_4=("G",))
        with self.assertRaises(ValueError):
            self.evaluate(folds=folds)

    def test_rejects_unknown_recording(self):
        folds = dict(self.folds, fold_4=("G", "Z"))
        with self.assertRaises(ValueError):
            self.evaluate(folds=folds)

    def test_rejects_duplicate_within_fold(self):
        folds = dict(self.folds, fold_4=("G", "G"))
        with self.assertRaises(ValueError):
            self.evaluate(folds=folds)

    def test_rejects_wrong_fold_count(self):
        folds = dict(self.folds)
        del folds["fold_4"]
        with self.assertRaises(ValueError):
            self.evaluate(folds=folds)

    def test_detects_wrong_group_count(self):
        limits = replace(self.limits, groups_per_fold=3)
        result = self.evaluate(limits=limits)
        self.assertFalse(result["eligible"])
        self.assertIn("fold_1_group_count", result["violations"])

    def test_detects_concentration(self):
        counts = dict(self.counts)
        counts["A"] = {"normal": 100, "rash": 100}
        result = self.evaluate(counts=counts)
        self.assertFalse(result["eligible"])
        self.assertIn(
            "fold_1_class_concentration", result["violations"]
        )

    def test_detects_class_support(self):
        counts = dict(self.counts)
        counts["A"] = {"normal": 40, "rash": 0}
        result = self.evaluate(counts=counts)
        self.assertIn("fold_1_class_groups", result["violations"])

    def test_detects_class_window_minimum(self):
        limits = replace(self.limits, minimum_class_windows=50)
        result = self.evaluate(limits=limits)
        self.assertIn("fold_1_class_windows", result["violations"])

    def test_rejects_boolean_count(self):
        counts = dict(self.counts)
        counts["A"] = {"normal": True, "rash": 20}
        with self.assertRaises(ValueError):
            self.evaluate(counts=counts)

    def test_rejects_invalid_fraction(self):
        limits = replace(
            self.limits, maximum_class_concentration=float("nan")
        )
        with self.assertRaises(ValueError):
            self.evaluate(limits=limits)


if __name__ == "__main__":
    unittest.main()
