"""Synthetic tests for D6 candidate identities and D7 selection."""

import unittest

from adas.evaluation.d7_selection import (
    approved_candidates,
    select_candidate,
)


class D7SelectionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = approved_candidates()

    def scores(self, value, balanced=None):
        second = value if balanced is None else balanced
        return ((value, second),) * 3

    def test_exact_grid_and_order(self):
        candidates = self.candidates
        self.assertEqual(len(candidates), 16)
        self.assertEqual(
            [item.family for item in candidates[:4]],
            ["dummy", "logistic_regression",
             "logistic_regression", "logistic_regression"],
        )
        self.assertEqual(
            [dict(item.parameters)["C"] for item in candidates[1:4]],
            [0.1, 1.0, 10.0],
        )
        self.assertEqual(
            dict(candidates[4].parameters),
            {"n_estimators": 100, "max_depth": 5,
             "min_samples_leaf": 1},
        )
        self.assertEqual(
            dict(candidates[-1].parameters),
            {"n_estimators": 200, "max_depth": None,
             "min_samples_leaf": 5},
        )

    def test_primary_macro_f1_priority(self):
        scores = {
            self.candidates[0].identity: self.scores(0.4, 0.9),
            self.candidates[1].identity: self.scores(0.6, 0.3),
        }
        winner, _, _ = select_candidate(scores)
        self.assertEqual(winner, self.candidates[1])

    def test_secondary_balanced_accuracy(self):
        scores = {
            self.candidates[0].identity: self.scores(0.5, 0.6),
            self.candidates[1].identity: self.scores(0.5, 0.7),
        }
        winner, _, _ = select_candidate(scores)
        self.assertEqual(winner, self.candidates[1])

    def test_deterministic_family_order(self):
        scores = {
            candidate.identity: self.scores(0.5, 0.5)
            for candidate in self.candidates
        }
        winner, _, _ = select_candidate(scores)
        self.assertEqual(winner, self.candidates[0])

    def test_incomplete_candidate_is_ineligible(self):
        scores = {
            self.candidates[0].identity: self.scores(0.3),
            self.candidates[1].identity: ((0.9, 0.9),),
        }
        winner, _, _ = select_candidate(scores)
        self.assertEqual(winner, self.candidates[0])

    def test_nonfinite_candidate_is_ineligible(self):
        scores = {
            self.candidates[0].identity: self.scores(0.3),
            self.candidates[1].identity: self.scores(float("nan")),
        }
        winner, _, _ = select_candidate(scores)
        self.assertEqual(winner, self.candidates[0])

    def test_no_eligible_candidate_is_error(self):
        with self.assertRaises(ValueError):
            select_candidate({})

    def test_unknown_candidate_is_error(self):
        with self.assertRaises(ValueError):
            select_candidate({("unapproved", ()): self.scores(0.9)})

    def test_numerical_tolerance(self):
        scores = {
            self.candidates[0].identity: self.scores(0.5, 0.5),
            self.candidates[1].identity: self.scores(
                0.5 + 5e-13, 0.5
            ),
        }
        winner, _, _ = select_candidate(scores)
        self.assertEqual(winner, self.candidates[0])


if __name__ == "__main__":
    unittest.main()
