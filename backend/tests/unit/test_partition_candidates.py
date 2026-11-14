"""Synthetic tests for deterministic diagnostic partition search."""

import unittest

from adas.evaluation.partition_candidates import (
    first_feasible_candidate,
)
from adas.evaluation.partition_contract import (
    validate_partition_class_coverage,
)


class PartitionCandidateTests(unittest.TestCase):

    def setUp(self):
        self.classes = ("normal", "rash")
        self.counts = {
            name: {"normal": 20, "rash": 15}
            for name in ("A", "B", "C", "D", "E")
        }

    def generate(self, counts=None, **kwargs):
        return first_feasible_candidate(
            self.counts if counts is None else counts,
            self.classes,
            **kwargs,
        )

    def test_returns_valid_candidate(self):
        result = self.generate()
        self.assertIsNotNone(result)
        validate_partition_class_coverage(
            result, self.counts, self.classes
        )

    def test_reproducible(self):
        self.assertEqual(self.generate(), self.generate())

    def test_input_order_independent(self):
        reversed_counts = dict(reversed(list(self.counts.items())))
        self.assertEqual(
            self.generate(),
            self.generate(reversed_counts),
        )

    def test_exact_group_coverage(self):
        result = self.generate()
        assigned = [
            group for groups in result.values() for group in groups
        ]
        self.assertEqual(set(assigned), set(self.counts))
        self.assertEqual(len(assigned), len(self.counts))

    def test_requested_group_sizes(self):
        result = self.generate(
            test_group_count=2, validation_group_count=1
        )
        self.assertEqual(len(result["test"]), 2)
        self.assertEqual(len(result["validation"]), 1)
        self.assertEqual(len(result["train"]), 2)

    def test_insufficient_groups(self):
        self.assertIsNone(
            self.generate(
                {"A": self.counts["A"], "B": self.counts["B"]},
            )
        )

    def test_no_feasible_class_coverage(self):
        counts = {
            "A": {"normal": 20},
            "B": {"normal": 20},
            "C": {"normal": 20},
            "D": {"rash": 20},
            "E": {"rash": 20},
        }
        self.assertIsNone(self.generate(counts))

    def test_rejects_invalid_group_count(self):
        with self.assertRaises(ValueError):
            self.generate(test_group_count=0)

    def test_rejects_boolean_group_count(self):
        with self.assertRaises(ValueError):
            self.generate(validation_group_count=True)


if __name__ == "__main__":
    unittest.main()
