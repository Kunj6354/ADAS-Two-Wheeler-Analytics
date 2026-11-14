"""Synthetic malformed-type hardening tests for M06-C4A."""

import copy
import unittest

from adas.data.m06_ground_truth import (
    validate_review,
    validate_review_pair,
)


def fixture():
    return {
        "recording_id": "SYNTHETIC_ONLY",
        "annotation_policy_version": "draft",
        "reference_source_id": "SYNTHETIC_REFERENCE",
        "reference_source_sha256": "a" * 64,
        "timebase_id": "SYNTHETIC_CLOCK",
        "event_start_ms": 100,
        "event_end_ms": 500,
        "reviewer_id": "reviewer_1",
        "observed_maneuver": "normal_turn",
        "reviewer_decision": "SUPPORTED",
        "uncertainty_reason": None,
        "provenance_notes": "Synthetic evidence only",
    }


class GroundTruthHardeningTests(unittest.TestCase):

    def test_reference_record_valid(self):
        result = validate_review(fixture())
        self.assertTrue(result.valid)
        self.assertFalse(result.ground_truth_confirmed)

    def test_malformed_maneuver_types(self):
        for value in ([], {}, ["normal_turn"], {"x": 1}, 123):
            with self.subTest(value=value):
                record = fixture()
                record["observed_maneuver"] = value
                before = copy.deepcopy(record)

                result = validate_review(record)

                self.assertFalse(result.valid)
                self.assertIn("INVALID_MANEUVER", result.errors)
                self.assertFalse(result.ground_truth_confirmed)
                self.assertEqual(record, before)

    def test_malformed_uncertainty_types(self):
        for value in ([], {}, ["INSUFFICIENT_EVIDENCE"], 123):
            with self.subTest(value=value):
                record = fixture()
                record["uncertainty_reason"] = value
                result = validate_review(record)

                self.assertFalse(result.valid)
                self.assertIn(
                    "INVALID_UNCERTAINTY_REASON",
                    result.errors,
                )

    def test_malformed_both_fields(self):
        record = fixture()
        record["observed_maneuver"] = []
        record["uncertainty_reason"] = {}

        result = validate_review(record)

        self.assertEqual(
            set(result.errors),
            {"INVALID_MANEUVER", "INVALID_UNCERTAINTY_REASON"},
        )

    def test_pair_with_invalid_maneuver(self):
        first = fixture()
        second = fixture()
        second["reviewer_id"] = "reviewer_2"
        second["observed_maneuver"] = []

        result = validate_review_pair(first, second)

        self.assertFalse(result.valid)
        self.assertIn("INVALID_MANEUVER", result.errors)
        self.assertFalse(result.ground_truth_confirmed)

    def test_pair_with_invalid_uncertainty(self):
        first = fixture()
        second = fixture()
        second["reviewer_id"] = "reviewer_2"
        second["uncertainty_reason"] = []

        result = validate_review_pair(first, second)

        self.assertFalse(result.valid)
        self.assertIn("INVALID_UNCERTAINTY_REASON", result.errors)

    def test_distinct_reviewers_still_required(self):
        record = fixture()
        result = validate_review_pair(record, copy.deepcopy(record))
        self.assertIn("REVIEWERS_NOT_DISTINCT", result.errors)

    def test_agreement_cannot_confirm_ground_truth(self):
        first = fixture()
        second = fixture()
        second["reviewer_id"] = "reviewer_2"

        result = validate_review_pair(first, second)

        self.assertTrue(result.valid)
        self.assertFalse(result.ground_truth_confirmed)


if __name__ == "__main__":
    unittest.main()
