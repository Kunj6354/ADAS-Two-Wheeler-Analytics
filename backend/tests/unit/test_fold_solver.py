"""Synthetic MILP grouped-fold generator tests."""

import unittest

from adas.evaluation.fold_solver import solve_grouped_folds
from adas.evaluation.grouped_folds import (
    FoldLimits,
    validate_grouped_folds,
)


class FoldSolverTests(unittest.TestCase):

    def setUp(self):
        self.classes = ("normal", "rash")
        self.counts = {
            name: {"normal": 20, "rash": 20}
            for name in "ABCDEFGH"
        }
        self.limits = FoldLimits(
            n_folds=4,
            groups_per_fold=2,
        )

    def solve(self, counts=None, limits=None, **kwargs):
        return solve_grouped_folds(
            self.counts if counts is None else counts,
            self.classes,
            self.limits if limits is None else limits,
            **kwargs,
        )

    def test_finds_verified_allocation(self):
        folds = self.solve()
        self.assertIsNotNone(folds)
        self.assertTrue(
            validate_grouped_folds(
                folds, self.counts, self.classes, self.limits
            )["eligible"]
        )

    def test_complete_group_coverage(self):
        folds = self.solve()
        assigned = [g for fold in folds.values() for g in fold]
        self.assertEqual(len(assigned), 8)
        self.assertEqual(set(assigned), set(self.counts))

    def test_repeatable_in_same_environment(self):
        self.assertEqual(self.solve(), self.solve())

    def test_input_order_independent(self):
        reversed_counts = dict(reversed(list(self.counts.items())))
        self.assertEqual(self.solve(), self.solve(reversed_counts))

    def test_proven_infeasible(self):
        counts = dict(self.counts)
        counts["A"] = {"normal": 1000, "rash": 1000}
        self.assertIsNone(self.solve(counts))

    def test_rejects_incompatible_group_count(self):
        with self.assertRaises(ValueError):
            self.solve(dict(list(self.counts.items())[:-1]))

    def test_rejects_invalid_time_limit(self):
        with self.assertRaises(ValueError):
            self.solve(time_limit=0)


if __name__ == "__main__":
    unittest.main()
