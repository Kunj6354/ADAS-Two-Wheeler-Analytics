"""Synthetic orchestration tests without fitting real classifiers."""

import unittest
from unittest.mock import patch

import numpy as np

try:
    import sklearn
except ModuleNotFoundError as exc:
    if exc.name != "sklearn":
        raise
    sklearn = None

from adas.evaluation.d7_selection import CLASS_NAMES

if sklearn is not None:
    from adas.evaluation.d7_inner_runner import evaluate_inner


@unittest.skipIf(
    sklearn is None,
    "Requires approved isolated ML environment",
)
class D7InnerRunnerTests(unittest.TestCase):
    def setUp(self):
        self.groups = tuple(
            f"group_{i}"
            for i in range(18)
            for _ in range(4)
        )
        self.labels = CLASS_NAMES * 18
        self.values = np.arange(72 * 3, dtype=float).reshape(72, 3)
        self.names = ("a", "b", "c")
        self.outer = {f"heldout_{i}" for i in range(6)}
        self.folds = {
            f"fold_{i+1}": [
                f"group_{j}" for j in range(i*6, (i+1)*6)
            ]
            for i in range(3)
        }

    def run_inner(self):
        return evaluate_inner(
            self.values,
            self.labels,
            self.groups,
            self.names,
            self.outer,
            self.folds,
        )

    def test_all_candidates_and_folds_audited(self):
        def predict(candidate, values, labels, groups,
                    prediction_values, **kwargs):
            return CLASS_NAMES * (len(prediction_values) // 4)

        with patch(
            "adas.evaluation.d7_inner_runner.fit_predict",
            side_effect=predict,
        ):
            result = self.run_inner()

        self.assertEqual(result.status, "PASS")
        self.assertEqual(len(result.outcomes), 48)
        self.assertTrue(all(x.status == "PASS" for x in result.outcomes))
        self.assertEqual(result.mean_macro_f1, 1.0)

    def test_candidate_failures_are_preserved(self):
        def predict(*args, **kwargs):
            raise RuntimeError("synthetic fit failure")

        with patch(
            "adas.evaluation.d7_inner_runner.fit_predict",
            side_effect=predict,
        ):
            result = self.run_inner()

        self.assertEqual(result.status, "INCOMPLETE")
        self.assertIsNone(result.winner_identity)
        self.assertEqual(len(result.outcomes), 48)
        self.assertTrue(all(x.status == "FAILED" for x in result.outcomes))
        self.assertTrue(all("synthetic fit failure" in x.error
                            for x in result.outcomes))

    def test_reject_outer_test_overlap(self):
        corrupted = self.outer | {"group_0"}
        with self.assertRaises(ValueError):
            evaluate_inner(
                self.values, self.labels, self.groups,
                self.names, corrupted, self.folds,
            )

    def test_reject_duplicate_inner_assignment(self):
        corrupted = dict(self.folds)
        corrupted["fold_2"] = list(corrupted["fold_1"])
        with self.assertRaises(ValueError):
            evaluate_inner(
                self.values, self.labels, self.groups,
                self.names, self.outer, corrupted,
            )


if __name__ == "__main__":
    unittest.main()
