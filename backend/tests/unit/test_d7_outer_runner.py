"""Synthetic outer-fold boundary and failure-accounting tests."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

try:
    import sklearn
except ModuleNotFoundError as exc:
    if exc.name != "sklearn":
        raise
    sklearn = None

from adas.evaluation.d7_selection import CLASS_NAMES, approved_candidates

if sklearn is not None:
    from adas.evaluation.d7_outer_runner import run_outer_fold


@unittest.skipIf(
    sklearn is None,
    "Requires approved isolated ML environment",
)
class D7OuterRunnerTests(unittest.TestCase):
    def setUp(self):
        groups = [f"MAIN{i}.CSV" for i in range(1, 25)]
        self.plan = SimpleNamespace(
            identities=tuple(
                (group, 1, 1, j)
                for group in groups for j in range(4)
            ),
            labels=CLASS_NAMES * 24,
            groups=tuple(group for group in groups for _ in range(4)),
            primary_indices=tuple(range(96)),
            sensitivity_indices=tuple(range(96)),
            outer_test_indices={
                f"outer_{i+1}": tuple(
                    index for index in range(96)
                    if groups[index // 4] in groups[i*6:(i+1)*6]
                )
                for i in range(4)
            },
            inner_validation_indices={},
        )
        self.outer_groups = set(groups[:6])
        self.inner_groups = {
            f"fold_{i+1}": groups[6+i*6:12+i*6]
            for i in range(3)
        }
        self.plan.inner_validation_indices["outer_1"] = {
            name: tuple(
                index for index in range(96)
                if self.plan.groups[index] in group_names
            )
            for name, group_names in self.inner_groups.items()
        }
        self.x = np.arange(96 * 3, dtype=float).reshape(96, 3)

    def run_case(self, **overrides):
        args = dict(
            plan=self.plan,
            matrix=self.x,
            feature_names=("a", "b", "c"),
            population_indices=self.plan.primary_indices,
            outer_fold="outer_1",
            outer_test_groups=self.outer_groups,
            inner_validation_groups=self.inner_groups,
        )
        args.update(overrides)
        return run_outer_fold(**args)

    def test_exact_outer_predictions(self):
        chosen = approved_candidates()[0]
        seen = []

        def predict(candidate, training_values, training_labels,
                    training_row_groups, prediction_values, **kwargs):
            seen.append((
                len(training_values),
                len(prediction_values),
                set(training_row_groups),
                set(kwargs["excluded_groups"]),
            ))
            return ("normal_driving",) * len(prediction_values)

        with patch(
            "adas.evaluation.d7_outer_runner.evaluate_inner",
            return_value=SimpleNamespace(
                status="PASS", winner_identity=chosen.identity,
                outcomes=(),
            ),
        ), patch(
            "adas.evaluation.d7_outer_runner.fit_predict",
            side_effect=predict,
        ):
            result = self.run_case()

        self.assertEqual(result.status, "PASS")
        self.assertEqual(len(result.predictions), 24)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0:2], (72, 24))
        self.assertEqual(seen[0][3], self.outer_groups)
        self.assertEqual(
            {record.identity for record in result.predictions},
            {self.plan.identities[i] for i in range(24)},
        )

    def test_refit_failure_kept_incomplete(self):
        chosen = approved_candidates()[0]
        with patch(
            "adas.evaluation.d7_outer_runner.evaluate_inner",
            return_value=SimpleNamespace(
                status="PASS", winner_identity=chosen.identity,
            ),
        ), patch(
            "adas.evaluation.d7_outer_runner.fit_predict",
            side_effect=RuntimeError("synthetic outer failure"),
        ):
            result = self.run_case()

        self.assertEqual(result.status, "INCOMPLETE")
        self.assertEqual(result.predictions, ())
        self.assertIn("synthetic outer failure", result.error)

    def test_reject_partial_population(self):
        with self.assertRaises(ValueError):
            self.run_case(population_indices=tuple(range(95)))

    def test_reject_frozen_inner_mismatch(self):
        changed = dict(self.inner_groups)
        changed["fold_1"] = changed["fold_2"]
        with self.assertRaises(ValueError):
            self.run_case(inner_validation_groups=changed)

    def test_no_inner_winner_is_incomplete(self):
        with patch(
            "adas.evaluation.d7_outer_runner.evaluate_inner",
            return_value=SimpleNamespace(
                status="INCOMPLETE", winner_identity=None,
            ),
        ):
            result = self.run_case()

        self.assertEqual(result.status, "INCOMPLETE")
        self.assertEqual(result.predictions, ())


if __name__ == "__main__":
    unittest.main()
