"""Synthetic D8 out-of-fold accounting and failure tests."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

from adas.evaluation.d8_oof_collector import collect_oof
try:
    import sklearn
except ModuleNotFoundError as exc:
    if exc.name != "sklearn":
        raise
    sklearn = None

if sklearn is not None:
    from adas.evaluation.d7_outer_runner import OuterPrediction
from adas.evaluation.d7_selection import CLASS_NAMES


@unittest.skipIf(
    sklearn is None,
    "Requires approved isolated ML environment",
)
class D8CollectorTests(unittest.TestCase):
    def setUp(self):
        groups = [f"MAIN{i}.CSV" for i in range(1, 25)]
        ids = tuple(
            (group, 1, 1, j)
            for group in groups for j in range(4)
        )
        labels = CLASS_NAMES * 24
        outer = {
            f"outer_{i + 1}": tuple(range(i * 24, (i + 1) * 24))
            for i in range(4)
        }

        self.plan = SimpleNamespace(
            identities=ids,
            labels=labels,
            groups=tuple(group for group in groups for _ in range(4)),
            primary_indices=tuple(range(72)),
            sensitivity_indices=tuple(range(96)),
            outer_test_indices=outer,
        )
        self.results = []
        for fold, indices in outer.items():
            predictions = tuple(
                OuterPrediction(ids[i], fold, labels[i], labels[i])
                for i in indices
            )
            self.results.append(SimpleNamespace(
                outer_fold=fold,
                status="PASS",
                selected_candidate=("dummy", ()),
                predictions=predictions,
                error=None,
            ))

    def collect(self, results=None, population=None):
        return collect_oof(
            self.plan,
            self.plan.sensitivity_indices if population is None else population,
            self.results if results is None else results,
        )

    def test_perfect_full_coverage(self):
        report = self.collect()
        self.assertEqual(report.status, "PASS")
        self.assertEqual(report.observed_count, 96)
        self.assertEqual(report.macro_f1, 1.0)
        self.assertEqual(report.balanced_accuracy, 1.0)
        self.assertEqual(len(report.fold_metrics), 4)
        self.assertEqual(len(report.recording_metrics), 24)

    def test_primary_subset(self):
        report = self.collect(
            results=[
                replace_result(
                    result,
                    predictions=tuple(
                        p for p in result.predictions
                        if p.identity in set(self.plan.identities[:72])
                    ),
                )
                for result in self.results
            ],
            population=self.plan.primary_indices,
        )
        self.assertEqual(report.status, "INCOMPLETE")
        # Synthetic primary membership excludes the entire fourth fold;
        # the collector correctly rejects the missing true classes.

    def test_valid_primary_subset_across_all_four_folds(self):
        # Remove one complete recording from each outer fold.
        # Each remaining outer fold still has all four classes.
        removed_groups = {0, 6, 12, 18}
        primary = tuple(
            index for index in range(96)
            if index // 4 not in removed_groups
        )
        self.plan.primary_indices = primary
        allowed = {self.plan.identities[index] for index in primary}

        adjusted_results = [
            replace_result(
                result,
                predictions=tuple(
                    record for record in result.predictions
                    if record.identity in allowed
                ),
            )
            for result in self.results
        ]

        report = self.collect(
            results=adjusted_results,
            population=primary,
        )
        self.assertEqual(report.status, "PASS")
        self.assertEqual(report.expected_count, 80)
        self.assertEqual(report.observed_count, 80)
        self.assertEqual(report.macro_f1, 1.0)
        self.assertEqual(len(report.fold_metrics), 4)

    def test_missing_prediction_is_incomplete(self):
        changed = list(self.results)
        changed[0] = replace_result(
            changed[0],
            predictions=changed[0].predictions[:-1],
        )
        report = self.collect(changed)
        self.assertEqual(report.status, "INCOMPLETE")
        self.assertIsNone(report.macro_f1)

    def test_duplicate_prediction_is_incomplete(self):
        changed = list(self.results)
        changed[0] = replace_result(
            changed[0],
            predictions=changed[0].predictions
                        + (changed[0].predictions[0],),
        )
        self.assertEqual(self.collect(changed).status, "INCOMPLETE")

    def test_wrong_true_label_is_incomplete(self):
        changed = list(self.results)
        first = changed[0].predictions[0]
        bad = replace(first, true_label="rash_turn")
        changed[0] = replace_result(
            changed[0],
            predictions=(bad,) + changed[0].predictions[1:],
        )
        self.assertEqual(self.collect(changed).status, "INCOMPLETE")

    def test_outer_failure_is_incomplete(self):
        changed = list(self.results)
        changed[0] = replace_result(
            changed[0],
            status="INCOMPLETE",
            predictions=(),
            error="synthetic failure",
        )
        self.assertEqual(self.collect(changed).status, "INCOMPLETE")

    def test_reject_unapproved_subset(self):
        with self.assertRaises(ValueError):
            self.collect(population=tuple(range(95)))

    def test_reject_duplicate_outer_result(self):
        changed = list(self.results)
        changed[3] = changed[0]
        with self.assertRaises(ValueError):
            self.collect(changed)


def replace_result(result, **updates):
    fields = vars(result).copy()
    fields.update(updates)
    return SimpleNamespace(**fields)


if __name__ == "__main__":
    unittest.main()
