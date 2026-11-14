"""Synthetic four-class metric and coverage tests."""

import unittest

from adas.evaluation.classification_metrics import (
    CLASS_NAMES,
    evaluate_classification,
)


class ClassificationMetricsTests(unittest.TestCase):

    def test_perfect_predictions(self):
        result = evaluate_classification(CLASS_NAMES, CLASS_NAMES)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(result["accuracy"], 1.0)

    def test_perfect_confusion_matrix(self):
        result = evaluate_classification(CLASS_NAMES, CLASS_NAMES)
        self.assertEqual(
            result["confusion_matrix"],
            ((1, 0, 0, 0), (0, 1, 0, 0),
             (0, 0, 1, 0), (0, 0, 0, 1)),
        )

    def test_all_predictions_one_class(self):
        predicted = (CLASS_NAMES[0],) * 4
        result = evaluate_classification(CLASS_NAMES, predicted)
        self.assertAlmostEqual(result["accuracy"], 0.25)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.25)
        self.assertAlmostEqual(result["macro_f1"], 0.1)

    def test_zero_predicted_support(self):
        predicted = (CLASS_NAMES[0],) * 4
        result = evaluate_classification(CLASS_NAMES, predicted)
        self.assertEqual(
            result["per_class"][CLASS_NAMES[1]]["precision"], 0.0
        )
        self.assertEqual(
            result["per_class"][CLASS_NAMES[1]]["f1"], 0.0
        )

    def test_unbalanced_class_support(self):
        true = (CLASS_NAMES[0],) * 10 + CLASS_NAMES[1:]
        result = evaluate_classification(true, true)
        self.assertEqual(result["sample_count"], 13)
        self.assertEqual(
            result["per_class"][CLASS_NAMES[0]]["support"], 10
        )
        self.assertEqual(result["macro_f1"], 1.0)

    def test_rejects_missing_true_class(self):
        with self.assertRaises(ValueError):
            evaluate_classification(CLASS_NAMES[:3], CLASS_NAMES[:3])

    def test_rejects_unknown_true_label(self):
        with self.assertRaises(ValueError):
            evaluate_classification(
                CLASS_NAMES[:3] + ("unknown",), CLASS_NAMES
            )

    def test_rejects_unknown_prediction(self):
        with self.assertRaises(ValueError):
            evaluate_classification(
                CLASS_NAMES, CLASS_NAMES[:3] + ("unknown",)
            )

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            evaluate_classification(CLASS_NAMES, CLASS_NAMES[:3])

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            evaluate_classification((), ())

    def test_rejects_wrong_class_order(self):
        with self.assertRaises(ValueError):
            evaluate_classification(
                CLASS_NAMES, CLASS_NAMES,
                classes=tuple(reversed(CLASS_NAMES)),
            )

    def test_rejects_string_instead_of_sequence(self):
        with self.assertRaises(ValueError):
            evaluate_classification("normal_driving", CLASS_NAMES)

    def test_counts_all_confusion_cells(self):
        predicted = (
            CLASS_NAMES[1], CLASS_NAMES[0],
            CLASS_NAMES[3], CLASS_NAMES[2],
        )
        result = evaluate_classification(CLASS_NAMES, predicted)
        self.assertEqual(
            sum(map(sum, result["confusion_matrix"])), 4
        )
        self.assertEqual(result["accuracy"], 0.0)

    def test_repeated_evaluation_is_deterministic(self):
        first = evaluate_classification(CLASS_NAMES, CLASS_NAMES)
        second = evaluate_classification(CLASS_NAMES, CLASS_NAMES)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
