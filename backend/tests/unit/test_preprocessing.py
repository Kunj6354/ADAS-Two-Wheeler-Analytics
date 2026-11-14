"""Synthetic leakage tests for training-only standardization."""

import unittest

import numpy as np

from adas.evaluation.preprocessing import fit_training_scaler


class TrainingOnlyPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.values = np.asarray(
            [[1.0, 4.0], [3.0, 4.0], [5.0, 4.0], [7.0, 4.0]]
        )
        self.groups = ("A", "A", "B", "B")
        self.names = ("acc_mean", "acc_constant")

    def fit(self, values=None, groups=None, **overrides):
        kwargs = {
            "training_groups": ("A", "B"),
            "excluded_groups": ("C", "D"),
            "feature_names": self.names,
        }
        kwargs.update(overrides)
        return fit_training_scaler(
            self.values if values is None else values,
            self.groups if groups is None else groups,
            **kwargs,
        )

    def test_means_from_training_only(self):
        scaler = self.fit()
        np.testing.assert_allclose(scaler.means, [4.0, 4.0])

    def test_constant_feature_safe(self):
        scaler = self.fit()
        self.assertEqual(scaler.scales[1], 1.0)
        transformed = scaler.transform(self.values)
        np.testing.assert_allclose(transformed[:, 1], 0.0)

    def test_training_values_are_centered(self):
        transformed = self.fit().transform(self.values)
        np.testing.assert_allclose(
            transformed.mean(axis=0), [0.0, 0.0], atol=1e-12
        )

    def test_validation_extremes_cannot_change_fit(self):
        scaler = self.fit()
        before = (scaler.means, scaler.scales)
        extreme_validation = [[1e9, -1e9]]
        transformed = scaler.transform(extreme_validation)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(before, (scaler.means, scaler.scales))

    def test_rejects_excluded_training_row(self):
        with self.assertRaises(ValueError):
            self.fit(groups=("A", "A", "B", "C"))

    def test_rejects_training_exclusion_overlap(self):
        with self.assertRaises(ValueError):
            self.fit(excluded_groups=("B", "C"))

    def test_rejects_missing_training_group(self):
        with self.assertRaises(ValueError):
            self.fit(training_groups=("A", "B", "E"))

    def test_rejects_nonfinite_training_input(self):
        values = self.values.copy()
        values[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.fit(values=values)

    def test_rejects_infinite_input(self):
        values = self.values.copy()
        values[0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.fit(values=values)

    def test_rejects_duplicate_feature_names(self):
        with self.assertRaises(ValueError):
            self.fit(feature_names=("duplicate", "duplicate"))

    def test_rejects_wrong_transform_dimensions(self):
        scaler = self.fit()
        with self.assertRaises(ValueError):
            scaler.transform([[1.0, 2.0, 3.0]])

    def test_repeatability(self):
        first = self.fit()
        second = self.fit()
        self.assertEqual(first, second)

    def test_rejects_incorrect_group_row_count(self):
        with self.assertRaises(ValueError):
            self.fit(groups=("A", "B"))

    def test_rejects_empty_exclusion_boundary(self):
        with self.assertRaises(ValueError):
            self.fit(excluded_groups=())


if __name__ == "__main__":
    unittest.main()
