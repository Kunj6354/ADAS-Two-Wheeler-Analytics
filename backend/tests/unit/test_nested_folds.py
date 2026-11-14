"""Synthetic nested-fold leakage and training-support tests."""

from copy import deepcopy
import unittest

from adas.evaluation.nested_folds import validate_nested_folds
from adas.evaluation.grouped_folds import FoldLimits


class NestedFoldTests(unittest.TestCase):
    def setUp(self):
        self.classes = ("normal", "rash")
        self.counts = {
            f"G{i:02}": {"normal": 20, "rash": 20}
            for i in range(24)
        }
        names = tuple(sorted(self.counts))
        self.outer = {
            f"outer_{i + 1}": names[i * 6:(i + 1) * 6]
            for i in range(4)
        }
        self.inner = {}

        for outer_name, held_out in self.outer.items():
            development = tuple(
                name for name in names if name not in held_out
            )
            self.inner[outer_name] = {
                f"inner_{i + 1}": development[i * 6:(i + 1) * 6]
                for i in range(3)
            }

    def evaluate(self, outer=None, inner=None, counts=None, **kwargs):
        return validate_nested_folds(
            self.outer if outer is None else outer,
            self.inner if inner is None else inner,
            self.counts if counts is None else counts,
            self.classes,
            outer_limits=FoldLimits(),
            inner_limits=FoldLimits(n_folds=3, groups_per_fold=6),
            **kwargs,
        )

    def test_valid_complete_hierarchy(self):
        result = self.evaluate()
        self.assertTrue(result["valid"])
        self.assertEqual(result["inner_evaluations"], 12)

    def test_repeatability(self):
        self.assertEqual(self.evaluate(), self.evaluate())

    def test_rejects_missing_outer_entry(self):
        inner = deepcopy(self.inner)
        del inner["outer_4"]
        with self.assertRaises(ValueError):
            self.evaluate(inner=inner)

    def test_rejects_outer_evaluation_leakage(self):
        inner = deepcopy(self.inner)
        inner["outer_1"]["inner_1"] = (
            self.outer["outer_1"][0],
            *inner["outer_1"]["inner_1"][1:],
        )
        with self.assertRaises(ValueError):
            self.evaluate(inner=inner)

    def test_rejects_duplicate_inner_group(self):
        inner = deepcopy(self.inner)
        first = inner["outer_1"]["inner_1"][0]
        inner["outer_1"]["inner_2"] = (
            first,
            *inner["outer_1"]["inner_2"][1:],
        )
        with self.assertRaises(ValueError):
            self.evaluate(inner=inner)

    def test_rejects_missing_inner_group(self):
        inner = deepcopy(self.inner)
        inner["outer_1"]["inner_3"] = (
            inner["outer_1"]["inner_3"][:-1]
        )
        with self.assertRaises(ValueError):
            self.evaluate(inner=inner)

    def test_rejects_invalid_outer_quality(self):
        outer = deepcopy(self.outer)
        outer["outer_1"] = outer["outer_1"][:-1]
        with self.assertRaises(ValueError):
            self.evaluate(outer=outer)

    def test_rejects_insufficient_training_windows(self):
        with self.assertRaises(ValueError):
            self.evaluate(minimum_training_class_windows=300)

    def test_rejects_insufficient_training_groups(self):
        with self.assertRaises(ValueError):
            self.evaluate(minimum_training_class_groups=13)

    def test_rejects_boolean_training_minimum(self):
        with self.assertRaises(ValueError):
            self.evaluate(minimum_training_class_groups=True)


if __name__ == "__main__":
    unittest.main()
