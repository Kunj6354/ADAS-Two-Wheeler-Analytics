"""Synthetic-only D6 classifier execution tests."""

import unittest

import numpy as np

try:
    import sklearn
except ModuleNotFoundError as exc:
    if exc.name != "sklearn":
        raise
    sklearn = None

from adas.evaluation.d7_selection import (
    CLASS_NAMES,
    approved_candidates,
)

if sklearn is not None:
    from adas.evaluation.d6_execution import estimator_for, fit_predict


@unittest.skipIf(
    sklearn is None,
    "Requires approved isolated ML environment",
)
class D6ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = approved_candidates()
        rng = np.random.default_rng(20261025)

        self.train = rng.normal(size=(48, 15))
        self.predict = rng.normal(size=(8, 15))

        self.labels = tuple(
            CLASS_NAMES[index % 4]
            for index in range(48)
        )
        self.groups = ("train_a",) * 24 + ("train_b",) * 24
        self.names = tuple(f"feature_{i}" for i in range(15))

    def execute(self, candidate):
        return fit_predict(
            candidate,
            self.train,
            self.labels,
            self.groups,
            self.predict,
            training_groups={"train_a", "train_b"},
            excluded_groups={"validation_a"},
            feature_names=self.names,
        )

    def test_exact_16_estimators_construct(self):
        for candidate in self.candidates:
            with self.subTest(candidate=candidate.identity):
                self.assertIsNotNone(estimator_for(candidate))

    def test_dummy_uses_fixed_class_order_for_tie(self):
        result = self.execute(self.candidates[0])
        self.assertEqual(result, ("normal_driving",) * 8)

    def test_logistic_deterministic(self):
        first = self.execute(self.candidates[1])
        second = self.execute(self.candidates[1])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)

    def test_random_forest_deterministic(self):
        first = self.execute(self.candidates[4])
        second = self.execute(self.candidates[4])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)

    def test_reject_training_exclusion_overlap(self):
        with self.assertRaises(ValueError):
            fit_predict(
                self.candidates[0],
                self.train,
                self.labels,
                self.groups,
                self.predict,
                training_groups={"train_a", "train_b"},
                excluded_groups={"train_a"},
                feature_names=self.names,
            )

    def test_reject_missing_training_group(self):
        with self.assertRaises(ValueError):
            fit_predict(
                self.candidates[0],
                self.train,
                self.labels,
                self.groups,
                self.predict,
                training_groups={"train_a"},
                excluded_groups={"validation_a"},
                feature_names=self.names,
            )

    def test_reject_nonfinite_input(self):
        invalid = self.train.copy()
        invalid[0, 0] = float("nan")

        with self.assertRaises(ValueError):
            fit_predict(
                self.candidates[0],
                invalid,
                self.labels,
                self.groups,
                self.predict,
                training_groups={"train_a", "train_b"},
                excluded_groups={"validation_a"},
                feature_names=self.names,
            )

    def test_reject_unapproved_candidate(self):
        from adas.evaluation.d7_selection import Candidate

        with self.assertRaises(ValueError):
            estimator_for(Candidate("unapproved", ()))

    def test_no_mutation_of_feature_arrays(self):
        before_train = self.train.copy()
        before_predict = self.predict.copy()

        self.execute(self.candidates[1])

        np.testing.assert_array_equal(self.train, before_train)
        np.testing.assert_array_equal(self.predict, before_predict)


if __name__ == "__main__":
    unittest.main()
