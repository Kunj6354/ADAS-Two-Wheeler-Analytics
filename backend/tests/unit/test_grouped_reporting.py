"""Synthetic outer-fold and recording-level reporting tests."""

import unittest

from adas.evaluation.classification_metrics import CLASS_NAMES
from adas.evaluation.grouped_reporting import (
    evaluate_grouped_predictions,
)


class GroupedReportingTests(unittest.TestCase):

    def setUp(self):
        self.outer = {
            "fold_1": ("A", "B"),
            "fold_2": ("C", "D"),
        }
        self.groups = tuple(
            group for group in ("A", "B", "C", "D")
            for _ in CLASS_NAMES
        )
        self.truth = CLASS_NAMES * 4
        self.predicted = self.truth

    def report(self, **overrides):
        args = {
            "y_true": self.truth,
            "y_pred": self.predicted,
            "row_groups": self.groups,
            "outer_folds": self.outer,
        }
        args.update(overrides)
        return evaluate_grouped_predictions(**args)

    def test_perfect_grouped_report(self):
        result = self.report()
        self.assertTrue(result["valid"])
        self.assertEqual(result["overall_accuracy"], 1.0)

    def test_two_fold_reports(self):
        result = self.report()
        self.assertEqual(result["outer_fold_count"], 2)
        self.assertEqual(len(result["per_fold"]), 2)

    def test_four_recording_reports(self):
        result = self.report()
        self.assertEqual(result["recording_count"], 4)
        self.assertEqual(result["per_recording"]["A"]["sample_count"], 4)

    def test_pooled_confusion_reconciles(self):
        result = self.report()
        self.assertEqual(
            sum(map(sum, result["pooled_confusion_matrix"])), 16
        )

    def test_sparse_recordings_have_no_four_class_macro_f1(self):
        truth = (
            CLASS_NAMES[0], CLASS_NAMES[0],
            CLASS_NAMES[1], CLASS_NAMES[1],
            CLASS_NAMES[2], CLASS_NAMES[2],
            CLASS_NAMES[3], CLASS_NAMES[3],
        )
        groups = ("A", "A", "B", "B", "C", "C", "D", "D")
        result = self.report(
            y_true=truth,
            y_pred=truth,
            row_groups=groups,
            outer_folds={"fold_1": ("A", "B", "C", "D")},
        )
        self.assertIsNone(
            result["per_recording"]["A"]["four_class_macro_f1"]
        )
        self.assertFalse(
            result["per_recording"]["A"]["four_class_coverage"]
        )

    def test_rejects_fold_missing_true_class(self):
        truth = list(self.truth)
        truth[0] = CLASS_NAMES[1]
        truth[4] = CLASS_NAMES[1]
        with self.assertRaises(ValueError):
            self.report(y_true=truth)

    def test_rejects_recording_in_two_folds(self):
        with self.assertRaises(ValueError):
            self.report(
                outer_folds={
                    "fold_1": ("A", "B"),
                    "fold_2": ("B", "C", "D"),
                }
            )

    def test_rejects_missing_recording(self):
        with self.assertRaises(ValueError):
            self.report(
                outer_folds={
                    "fold_1": ("A", "B"),
                    "fold_2": ("C",),
                }
            )

    def test_rejects_unknown_prediction(self):
        predictions = self.predicted[:-1] + ("UNKNOWN",)
        with self.assertRaises(ValueError):
            self.report(y_pred=predictions)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            self.report(y_pred=self.predicted[:-1])

    def test_repeated_report_is_deterministic(self):
        self.assertEqual(self.report(), self.report())


if __name__ == "__main__":
    unittest.main()
