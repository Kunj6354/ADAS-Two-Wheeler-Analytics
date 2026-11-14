"""Nested-CV preprocessing boundary tests using synthetic recordings."""

import unittest

import numpy as np

from adas.evaluation.grouped_folds import FoldLimits
from adas.evaluation.nested_folds import validate_nested_folds
from adas.evaluation.preprocessing import fit_training_scaler


class NestedPreprocessingTests(unittest.TestCase):

    def setUp(self):
        self.classes = ("normal", "rash")
        self.names = ("feature_a", "feature_b")

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
                f"inner_{i + 1}":
                    development[i * 6:(i + 1) * 6]
                for i in range(3)
            }

        self.hierarchy = validate_nested_folds(
            self.outer,
            self.inner,
            self.counts,
            self.classes,
            outer_limits=FoldLimits(),
            inner_limits=FoldLimits(
                n_folds=3, groups_per_fold=6
            ),
        )

        # Two deterministic synthetic observations per recording.
        self.row_groups = tuple(
            name
            for name in names
            for _ in range(2)
        )

        self.values = np.asarray(
            [
                [float(i * 2 + j), float(i + j + 10)]
                for i in range(24)
                for j in range(2)
            ],
            dtype=np.float64,
        )

    def partitions(self, outer_name, inner_name):
        held_out = set(self.outer[outer_name])
        validation = set(self.inner[outer_name][inner_name])
        development = set(self.counts) - held_out
        training = development - validation

        return training, validation, held_out

    def fit_partition(self, values, training, excluded):
        selected = [
            i for i, group in enumerate(self.row_groups)
            if group in training
        ]

        return fit_training_scaler(
            values[selected],
            tuple(self.row_groups[i] for i in selected),
            training_groups=training,
            excluded_groups=excluded,
            feature_names=self.names,
        )

    def test_complete_hierarchy(self):
        self.assertTrue(self.hierarchy["valid"])
        self.assertEqual(self.hierarchy["inner_evaluations"], 12)

    def test_all_twelve_training_boundaries(self):
        audited = 0

        for outer_name in sorted(self.outer):
            for inner_name in sorted(self.inner[outer_name]):
                training, validation, held_out = self.partitions(
                    outer_name, inner_name
                )

                self.assertEqual(len(training), 12)
                self.assertEqual(len(validation), 6)
                self.assertEqual(len(held_out), 6)

                self.assertFalse(training & validation)
                self.assertFalse(training & held_out)
                self.assertFalse(validation & held_out)

                scaler = self.fit_partition(
                    self.values,
                    training,
                    validation | held_out,
                )

                self.assertEqual(scaler.training_groups, training)
                audited += 1

        self.assertEqual(audited, 12)

    def test_validation_perturbation_invariance(self):
        audited = 0

        for outer_name in sorted(self.outer):
            for inner_name in sorted(self.inner[outer_name]):
                training, validation, held_out = self.partitions(
                    outer_name, inner_name
                )

                original = self.fit_partition(
                    self.values, training, validation | held_out
                )

                changed = self.values.copy()

                for i, group in enumerate(self.row_groups):
                    if group in validation:
                        changed[i] = [1e8, -1e8]

                repeated = self.fit_partition(
                    changed, training, validation | held_out
                )

                self.assertEqual(original.means, repeated.means)
                self.assertEqual(original.scales, repeated.scales)
                audited += 1

        self.assertEqual(audited, 12)

    def test_outer_evaluation_perturbation_invariance(self):
        audited = 0

        for outer_name in sorted(self.outer):
            for inner_name in sorted(self.inner[outer_name]):
                training, validation, held_out = self.partitions(
                    outer_name, inner_name
                )

                original = self.fit_partition(
                    self.values, training, validation | held_out
                )

                changed = self.values.copy()

                for i, group in enumerate(self.row_groups):
                    if group in held_out:
                        changed[i] = [-1e9, 1e9]

                repeated = self.fit_partition(
                    changed, training, validation | held_out
                )

                self.assertEqual(original.means, repeated.means)
                self.assertEqual(original.scales, repeated.scales)
                audited += 1

        self.assertEqual(audited, 12)

    def test_excluded_rows_cannot_be_fitted(self):
        training, validation, held_out = self.partitions(
            "outer_1", "inner_1"
        )

        forbidden = next(
            name for name in self.row_groups
            if name in validation
        )

        selected = [
            i for i, group in enumerate(self.row_groups)
            if group in training
        ]

        contaminated_values = np.vstack(
            (self.values[selected], [[999.0, -999.0]])
        )

        contaminated_groups = (
            tuple(self.row_groups[i] for i in selected)
            + (forbidden,)
        )

        with self.assertRaises(ValueError):
            fit_training_scaler(
                contaminated_values,
                contaminated_groups,
                training_groups=training,
                excluded_groups=validation | held_out,
                feature_names=self.names,
            )

    def test_training_change_affects_fitted_statistics(self):
        training, validation, held_out = self.partitions(
            "outer_1", "inner_1"
        )

        original = self.fit_partition(
            self.values, training, validation | held_out
        )

        changed = self.values.copy()
        index = next(
            i for i, group in enumerate(self.row_groups)
            if group in training
        )
        changed[index, 0] += 1000.0

        repeated = self.fit_partition(
            changed, training, validation | held_out
        )

        self.assertNotEqual(original.means, repeated.means)


if __name__ == "__main__":
    unittest.main()
