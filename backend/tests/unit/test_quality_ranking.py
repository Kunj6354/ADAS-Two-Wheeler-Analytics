"""Tests for deterministic quality-ranking keys."""

import unittest
from dataclasses import replace

from adas.evaluation.quality_contract import QualityLimits
from adas.evaluation.quality_ranking import quality_rank_key


class QualityRankingTests(unittest.TestCase):

    def setUp(self):
        self.classes = ("normal", "rash")
        self.counts = {
            name: {"normal": 20, "rash": 20}
            for name in "ABCDEFGH"
        }
        self.candidate = {
            "train": ("A", "B", "C", "D"),
            "validation": ("E", "F"),
            "test": ("G", "H"),
        }

    def rank(self, candidate=None, counts=None, classes=None, limits=None):
        return quality_rank_key(
            self.candidate if candidate is None else candidate,
            self.counts if counts is None else counts,
            self.classes if classes is None else classes,
            QualityLimits() if limits is None else limits,
        )

    def test_valid_rank_key(self):
        key = self.rank()
        self.assertEqual(len(key), 4)
        self.assertEqual(key[0], 0.5)

    def test_deterministic(self):
        self.assertEqual(self.rank(), self.rank())

    def test_recording_count_order_independent(self):
        reordered = dict(reversed(list(self.counts.items())))
        self.assertEqual(self.rank(), self.rank(counts=reordered))

    def test_partition_group_order_independent(self):
        reordered = {
            key: tuple(reversed(value))
            for key, value in self.candidate.items()
        }
        self.assertEqual(self.rank(), self.rank(candidate=reordered))

    def test_class_order_independent(self):
        self.assertEqual(
            self.rank(),
            self.rank(classes=tuple(reversed(self.classes))),
        )

    def test_fixed_lexical_tie_break(self):
        alternative = {
            "train": ("A", "B", "C", "D"),
            "validation": ("G", "H"),
            "test": ("E", "F"),
        }
        first = self.rank()
        second = self.rank(candidate=alternative)

        self.assertEqual(first[:3], second[:3])
        self.assertNotEqual(first[3], second[3])
        self.assertLess(first, second)

    def test_rejects_ineligible_candidate(self):
        candidate = {
            "train": ("A", "B", "C", "D", "E", "F"),
            "validation": ("G",),
            "test": ("H",),
        }
        with self.assertRaises(ValueError):
            self.rank(candidate=candidate)

    def test_rejects_overlap(self):
        candidate = dict(self.candidate, test=("A", "H"))
        with self.assertRaises(ValueError):
            self.rank(candidate=candidate)

    def test_rejects_invalid_limits(self):
        limits = replace(
            QualityLimits(),
            max_class_concentration=float("nan"),
        )
        with self.assertRaises(ValueError):
            self.rank(limits=limits)

    def test_rejects_unapproved_training_share(self):
        limits = replace(QualityLimits(), train_min_share=0.60)
        with self.assertRaises(ValueError):
            self.rank(limits=limits)


if __name__ == "__main__":
    unittest.main()
