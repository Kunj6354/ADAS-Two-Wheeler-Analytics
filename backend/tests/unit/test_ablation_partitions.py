"""Synthetic fold-consistent research ablation tests."""

import unittest
import numpy as np

from adas.evaluation.feature_ablations import build_matched_ablations
from adas.evaluation.ablation_partitions import prepare_matched_partition


class AblationPartitionTests(unittest.TestCase):

    def setUp(self):
        self.groups = ("A", "A", "B", "B", "C", "C", "D", "D")
        self.identities = tuple(
            (group, 1, 1, index)
            for index, group in enumerate(self.groups)
        )
        self.labels = ("normal_driving",) * 8
        self.matrix = np.arange(8 * 39, dtype=float).reshape(8, 39)

    def build(self, matrix=None):
        return build_matched_ablations(
            self.matrix if matrix is None else matrix,
            self.identities,
            self.groups,
            self.labels,
        )

    def prepare(self, variants=None, **overrides):
        settings = {
            "training_groups": ("A", "B"),
            "validation_groups": ("C",),
            "outer_evaluation_groups": ("D",),
        }
        settings.update(overrides)
        return prepare_matched_partition(
            self.build() if variants is None else variants,
            **settings,
        )

    def test_four_variants_share_training_and_validation_rows(self):
        audit = self.prepare()
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["variant_count"], 4)
        for item in audit["partitions"].values():
            self.assertEqual(item["training_rows"], 4)
            self.assertEqual(item["validation_rows"], 2)

    def test_each_variant_uses_correct_feature_dimension(self):
        audit = self.prepare()
        expected = {
            "accelerometer_only": 15,
            "accelerometer_plus_rates": 24,
            "statistical_baseline": 30,
            "combined_39": 39,
        }
        for name, width in expected.items():
            self.assertEqual(
                audit["partitions"][name]["feature_count"], width
            )

    def test_scalers_fit_only_on_training_groups(self):
        audit = self.prepare()
        for item in audit["partitions"].values():
            self.assertEqual(
                item["scaler"].training_groups, frozenset(("A", "B"))
            )

    def test_validation_perturbation_does_not_change_fit(self):
        baseline = self.prepare()
        changed = self.matrix.copy()
        changed[4:6] += 1e8
        altered = self.prepare(self.build(changed))
        for name in baseline["partitions"]:
            first = baseline["partitions"][name]["scaler"]
            second = altered["partitions"][name]["scaler"]
            self.assertEqual(first.means, second.means)
            self.assertEqual(first.scales, second.scales)

    def test_outer_perturbation_does_not_change_fit(self):
        baseline = self.prepare()
        changed = self.matrix.copy()
        changed[6:8] -= 1e8
        altered = self.prepare(self.build(changed))
        for name in baseline["partitions"]:
            first = baseline["partitions"][name]["scaler"]
            second = altered["partitions"][name]["scaler"]
            self.assertEqual(first.means, second.means)
            self.assertEqual(first.scales, second.scales)

    def test_rejects_overlapping_groups(self):
        with self.assertRaises(ValueError):
            self.prepare(validation_groups=("B", "C"))

    def test_rejects_missing_groups(self):
        with self.assertRaises(ValueError):
            self.prepare(outer_evaluation_groups=("C",))

    def test_rejects_empty_partition(self):
        with self.assertRaises(ValueError):
            self.prepare(validation_groups=())

    def test_training_change_affects_scaler(self):
        baseline = self.prepare()
        changed = self.matrix.copy()
        changed[0, 0] += 1000
        altered = self.prepare(self.build(changed))
        first = baseline["partitions"]["combined_39"]["scaler"]
        second = altered["partitions"]["combined_39"]["scaler"]
        self.assertNotEqual(first.means, second.means)


if __name__ == "__main__":
    unittest.main()
