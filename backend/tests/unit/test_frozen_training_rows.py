"""Synthetic tests for exact frozen-window enforcement."""

import copy
import unittest

from adas.evaluation.d7_selection import CLASS_NAMES
from adas.evaluation.frozen_training_rows import validate_frozen_rows


class FrozenTrainingRowTests(unittest.TestCase):

    def setUp(self):
        names = [f"MAIN{i}.CSV" for i in range(1, 25)]
        rows = []
        for group in names:
            for index, label in enumerate(CLASS_NAMES):
                rows.append({
                    "identity": [group, 1, 1, index],
                    "label": label,
                    "rate_compatible": index != 3,
                })

        self.inventory = {"windows": rows}
        self.identities = tuple(
            tuple(row["identity"]) for row in rows
        )
        self.labels = tuple(row["label"] for row in rows)
        self.groups = tuple(row["identity"][0] for row in rows)
        self.rates = tuple(row["rate_compatible"] for row in rows)

        outer = {
            f"outer_{i + 1}": names[i*6:(i+1)*6]
            for i in range(4)
        }

        inner = {}
        for outer_name, excluded in outer.items():
            development = [
                group for group in names if group not in excluded
            ]
            inner[outer_name] = {
                f"fold_{i + 1}": development[i*6:(i+1)*6]
                for i in range(3)
            }

        self.freeze = {
            "freeze_status": "FROZEN",
            "outer_test_groups": outer,
            "inner_validation_groups_by_outer": inner,
        }

    def validate(self, **overrides):
        inputs = {
            "identities": self.identities,
            "labels": self.labels,
            "groups": self.groups,
            "rate_compatible": self.rates,
            "inventory": self.inventory,
            "freeze": self.freeze,
        }
        inputs.update(overrides)
        return validate_frozen_rows(**inputs)

    def test_complete_population_and_folds(self):
        plan = self.validate()
        self.assertEqual(len(plan.identities), 96)
        self.assertEqual(len(plan.primary_indices), 72)
        self.assertEqual(len(plan.sensitivity_indices), 96)
        self.assertEqual(
            [len(x) for x in plan.outer_test_indices.values()],
            [24] * 4,
        )
        self.assertTrue(all(
            len(folds) == 3
            for folds in plan.inner_validation_indices.values()
        ))

    def test_reject_missing_identity(self):
        with self.assertRaises(ValueError):
            self.validate(
                identities=self.identities[:-1],
                labels=self.labels[:-1],
                groups=self.groups[:-1],
                rate_compatible=self.rates[:-1],
            )

    def test_reject_duplicate_identity(self):
        corrupted = list(self.identities)
        corrupted[-1] = corrupted[0]
        with self.assertRaises(ValueError):
            self.validate(identities=corrupted)

    def test_reject_relabeling(self):
        corrupted = list(self.labels)
        corrupted[0] = "rash_driving"
        with self.assertRaises(ValueError):
            self.validate(labels=corrupted)

    def test_reject_changed_population_membership(self):
        corrupted = list(self.rates)
        corrupted[0] = False
        with self.assertRaises(ValueError):
            self.validate(rate_compatible=corrupted)

    def test_reject_outer_inner_leakage(self):
        corrupted = copy.deepcopy(self.freeze)
        corrupted["inner_validation_groups_by_outer"]["outer_1"][
            "fold_1"
        ][0] = corrupted["outer_test_groups"]["outer_1"][0]

        with self.assertRaises(ValueError):
            self.validate(freeze=corrupted)


if __name__ == "__main__":
    unittest.main()
