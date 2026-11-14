"""Synthetic tests for matched-row feature ablations."""

from dataclasses import replace
import unittest

import numpy as np

from adas.evaluation.feature_ablations import (
    ALL_FEATURES,
    VARIANT_COLUMNS,
    build_matched_ablations,
    validate_matched_ablations,
)


class MatchedFeatureAblationTests(unittest.TestCase):

    def setUp(self):
        self.matrix = np.arange(
            4 * 39, dtype=float
        ).reshape(4, 39)

        self.identities = (
            ("A", 1, 1, 0),
            ("A", 1, 1, 1),
            ("B", 1, 1, 0),
            ("B", 1, 1, 1),
        )

        self.groups = ("A", "A", "B", "B")

        self.labels = (
            "normal_driving",
            "rash_driving",
            "normal_turn",
            "rash_turn",
        )

    def build(self, **overrides):
        inputs = {
            "values": self.matrix,
            "identities": self.identities,
            "groups": self.groups,
            "labels": self.labels,
            "feature_names": ALL_FEATURES,
        }
        inputs.update(overrides)
        return build_matched_ablations(**inputs)

    def test_all_four_variants_validate(self):
        result = validate_matched_ablations(self.build())
        self.assertTrue(result["valid"])
        self.assertEqual(result["row_count"], 4)

    def test_expected_dimensions(self):
        variants = self.build()
        expected = {
            "accelerometer_only": 15,
            "accelerometer_plus_rates": 24,
            "statistical_baseline": 30,
            "combined_39": 39,
        }
        for name, width in expected.items():
            self.assertEqual(variants[name].values.shape, (4, width))

    def test_same_identity_label_and_group_order(self):
        variants = self.build()
        for item in variants.values():
            self.assertEqual(item.identities, self.identities)
            self.assertEqual(item.labels, self.labels)
            self.assertEqual(item.groups, self.groups)

    def test_column_selection_exact(self):
        variants = self.build()
        for name, columns in VARIANT_COLUMNS.items():
            np.testing.assert_array_equal(
                variants[name].values,
                self.matrix[:, columns],
            )

    def test_rejects_duplicate_identity(self):
        with self.assertRaises(ValueError):
            self.build(identities=(
                self.identities[0],
                self.identities[0],
                *self.identities[2:],
            ))

    def test_rejects_metadata_length_mismatch(self):
        with self.assertRaises(ValueError):
            self.build(labels=self.labels[:-1])

    def test_rejects_identity_group_mismatch(self):
        with self.assertRaises(ValueError):
            self.build(groups=("B", "A", "B", "B"))

    def test_rejects_incorrect_feature_order(self):
        changed = list(ALL_FEATURES)
        changed[0], changed[1] = changed[1], changed[0]
        with self.assertRaises(ValueError):
            self.build(feature_names=changed)

    def test_rejects_nonfinite_feature(self):
        changed = self.matrix.copy()
        changed[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.build(values=changed)

    def test_rejects_unknown_class(self):
        with self.assertRaises(ValueError):
            self.build(labels=(
                "normal_driving",
                "rash_driving",
                "UNKNOWN",
                "rash_turn",
            ))

    def test_rejects_missing_variant(self):
        variants = self.build()
        del variants["accelerometer_only"]
        with self.assertRaises(ValueError):
            validate_matched_ablations(variants)

    def test_rejects_reordered_variant_rows(self):
        variants = self.build()
        variant = variants["accelerometer_only"]
        reordered = variant.values[::-1].copy()
        variants["accelerometer_only"] = replace(
            variant, values=reordered
        )
        with self.assertRaises(ValueError):
            validate_matched_ablations(variants)

    def test_rejects_variant_metadata_changes(self):
        variants = self.build()
        variant = variants["statistical_baseline"]
        variants["statistical_baseline"] = replace(
            variant, labels=tuple(reversed(variant.labels))
        )
        with self.assertRaises(ValueError):
            validate_matched_ablations(variants)

    def test_output_does_not_alias_input(self):
        variants = self.build()
        initial = variants["combined_39"].values[0, 0]
        self.matrix[0, 0] = -999.0
        self.assertEqual(
            variants["combined_39"].values[0, 0],
            initial,
        )

    def test_variant_matrix_read_only(self):
        variants = self.build()
        self.assertFalse(
            variants["combined_39"].values.flags.writeable
        )


if __name__ == "__main__":
    unittest.main()
