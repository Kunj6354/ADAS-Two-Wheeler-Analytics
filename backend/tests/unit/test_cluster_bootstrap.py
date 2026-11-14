"""Synthetic recording-cluster bootstrap contract tests."""

import unittest

from adas.evaluation.classification_metrics import CLASS_NAMES
from adas.evaluation.cluster_bootstrap import (
    recording_cluster_bootstrap,
)


class ClusterBootstrapTests(unittest.TestCase):

    def setUp(self):
        self.groups = tuple(
            group
            for group in ("A", "B", "C", "D")
            for _ in CLASS_NAMES
        )
        self.truth = CLASS_NAMES * 4
        self.predictions = self.truth

    def audit(self, **overrides):
        args = {
            "y_true": self.truth,
            "y_pred": self.predictions,
            "recording_groups": self.groups,
            "repetitions": 100,
            "seed": 17,
        }
        args.update(overrides)
        return recording_cluster_bootstrap(**args)

    def test_perfect_predictions(self):
        result = self.audit()
        self.assertEqual(result["macro_f1"]["point_estimate"], 1.0)
        self.assertEqual(
            result["macro_f1"]["percentile_interval"], (1.0, 1.0)
        )
        self.assertEqual(
            result["balanced_accuracy"]["percentile_interval"],
            (1.0, 1.0),
        )

    def test_fixed_seed_is_deterministic(self):
        self.assertEqual(self.audit(), self.audit())

    def test_all_draws_are_counted(self):
        result = self.audit()
        self.assertEqual(
            result["valid_repetitions"]
            + result["rejected_class_coverage"],
            100,
        )

    def test_preserves_source_group_count(self):
        result = self.audit()
        self.assertEqual(result["recording_count"], 4)
        self.assertEqual(result["window_count"], 16)

    def test_rejects_missing_original_true_class(self):
        # Remove every instance of one true class.
        mask = tuple(
            i for i, label in enumerate(self.truth)
            if label != CLASS_NAMES[-1]
        )
        with self.assertRaises(ValueError):
            self.audit(
                y_true=tuple(self.truth[i] for i in mask),
                y_pred=tuple(self.predictions[i] for i in mask),
                recording_groups=tuple(self.groups[i] for i in mask),
            )

    def test_rejects_unknown_prediction(self):
        with self.assertRaises(ValueError):
            self.audit(
                y_pred=self.predictions[:-1] + ("UNKNOWN",)
            )

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            self.audit(y_pred=self.predictions[:-1])

    def test_rejects_single_group(self):
        with self.assertRaises(ValueError):
            self.audit(recording_groups=("A",) * 16)

    def test_rejects_zero_repetitions(self):
        with self.assertRaises(ValueError):
            self.audit(repetitions=0)

    def test_rejects_boolean_repetitions(self):
        with self.assertRaises(ValueError):
            self.audit(repetitions=True)

    def test_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            self.audit(confidence_level=1.0)

    def test_rejects_negative_seed(self):
        with self.assertRaises(ValueError):
            self.audit(seed=-1)

    def test_sparse_group_resamples_accounted_for(self):
        truth = CLASS_NAMES
        groups = ("A", "B", "C", "D")
        result = recording_cluster_bootstrap(
            truth, truth, groups,
            repetitions=2000, seed=17,
        )
        self.assertGreater(result["valid_repetitions"], 0)
        self.assertGreater(result["rejected_class_coverage"], 0)
        self.assertEqual(
            result["valid_repetitions"]
            + result["rejected_class_coverage"],
            2000,
        )

    def test_all_one_class_predictions_are_finite(self):
        predicted = (CLASS_NAMES[0],) * 16
        result = self.audit(y_pred=predicted)
        self.assertAlmostEqual(
            result["macro_f1"]["point_estimate"], 0.1
        )
        for metric in ("macro_f1", "balanced_accuracy"):
            low, high = result[metric]["percentile_interval"]
            self.assertTrue(0 <= low <= high <= 1)


if __name__ == "__main__":
    unittest.main()
