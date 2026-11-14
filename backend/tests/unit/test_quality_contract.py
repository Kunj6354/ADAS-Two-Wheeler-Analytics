"""Synthetic checks for exploratory partition quality."""

import unittest
from dataclasses import replace

from adas.evaluation.quality_contract import (
    QualityLimits,
    evaluate_partition_quality,
)


class QualityContractTests(unittest.TestCase):

    def setUp(self):
        self.classes = ("normal", "rash")
        self.counts = {
            name: {"normal": 20, "rash": 20}
            for name in "ABCDEFGH"
        }
        self.partitions = {
            "train": ("A", "B", "C", "D"),
            "validation": ("E", "F"),
            "test": ("G", "H"),
        }
        self.limits = QualityLimits(
            minimum_train_class_groups=3,
            minimum_evaluation_class_groups=2,
        )

    def evaluate(self, **kwargs):
        return evaluate_partition_quality(
            kwargs.get("partitions", self.partitions),
            kwargs.get("counts", self.counts),
            self.classes,
            kwargs.get("limits", self.limits),
        )

    def test_valid(self):
        self.assertTrue(self.evaluate()["eligible"])

    def test_deterministic(self):
        self.assertEqual(self.evaluate(), self.evaluate())

    def test_train_share(self):
        limits = replace(self.limits, train_min_share=0.60)
        self.assertIn("train_share", self.evaluate(limits=limits)["violations"])

    def test_evaluation_share(self):
        limits = replace(self.limits, evaluation_min_share=0.30)
        result = self.evaluate(limits=limits)
        self.assertIn("validation_share", result["violations"])
        self.assertIn("test_share", result["violations"])

    def test_validation_class_support(self):
        limits = replace(self.limits, minimum_validation_windows=50)
        self.assertIn(
            "validation_class_windows",
            self.evaluate(limits=limits)["violations"],
        )

    def test_test_class_support(self):
        limits = replace(self.limits, minimum_test_windows=50)
        self.assertIn(
            "test_class_windows",
            self.evaluate(limits=limits)["violations"],
        )

    def test_group_support(self):
        limits = replace(self.limits, minimum_train_class_groups=5)
        self.assertIn(
            "train_class_groups",
            self.evaluate(limits=limits)["violations"],
        )

    def test_concentration(self):
        counts = dict(self.counts)
        counts["E"] = {"normal": 100, "rash": 100}
        result = self.evaluate(counts=counts)
        self.assertIn(
            "validation_class_concentration", result["violations"]
        )

    def test_overlap_is_error(self):
        partitions = dict(self.partitions, test=("A", "H"))
        with self.assertRaises(ValueError):
            self.evaluate(partitions=partitions)

    def test_invalid_fraction_is_error(self):
        limits = replace(
            self.limits, max_class_concentration=float("nan")
        )
        with self.assertRaises(ValueError):
            self.evaluate(limits=limits)

    def test_invalid_integer_is_error(self):
        limits = replace(self.limits, minimum_test_windows=True)
        with self.assertRaises(ValueError):
            self.evaluate(limits=limits)


if __name__ == "__main__":
    unittest.main()
