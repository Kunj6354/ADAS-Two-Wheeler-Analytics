"""Synthetic tests of the draft M06 ground-truth record contract."""

import copy
import unittest

from adas.data.m06_ground_truth import (
    validate_review,
    validate_review_pair,
)


def example():
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
        "provenance_notes": "Synthetic fixture only",
    }


class GroundTruthContractTests(unittest.TestCase):

    def test_draft_supported_review_not_ground_truth(self):
        result = validate_review(example())
        self.assertTrue(result.valid)
        self.assertFalse(result.ground_truth_confirmed)

    def test_missing_field(self):
        record = example()
        del record["timebase_id"]
        self.assertIn(
            "MISSING_TIMEBASE_ID", validate_review(record).errors
        )

    def test_invalid_hash(self):
        record = example()
        record["reference_source_sha256"] = "invalid"
        self.assertIn(
            "INVALID_REFERENCE_SHA256",
            validate_review(record).errors,
        )

    def test_reversed_interval(self):
        record = example()
        record["event_end_ms"] = 99
        self.assertIn(
            "INVALID_EVENT_INTERVAL",
            validate_review(record).errors,
        )

    def test_boolean_timestamp_rejected(self):
        record = example()
        record["event_start_ms"] = True
        self.assertIn(
            "INVALID_EVENT_INTERVAL",
            validate_review(record).errors,
        )

    def test_unknown_maneuver_rejected(self):
        record = example()
        record["observed_maneuver"] = "unknown_class"
        self.assertIn(
            "INVALID_MANEUVER",
            validate_review(record).errors,
        )

    def test_unresolved_requires_reason(self):
        record = example()
        record["reviewer_decision"] = "UNRESOLVED"
        record["observed_maneuver"] = None
        self.assertIn(
            "UNRESOLVED_REQUIRES_REASON",
            validate_review(record).errors,
        )

    def test_unresolved_with_reason(self):
        record = example()
        record["reviewer_decision"] = "UNRESOLVED"
        record["observed_maneuver"] = None
        record["uncertainty_reason"] = "INSUFFICIENT_EVIDENCE"
        self.assertTrue(validate_review(record).valid)

    def test_duplicate_reviewers_rejected(self):
        first = example()
        second = copy.deepcopy(first)
        result = validate_review_pair(first, second)
        self.assertIn("REVIEWERS_NOT_DISTINCT", result.errors)

    def test_reference_mismatch_rejected(self):
        first = example()
        second = copy.deepcopy(first)
        second["reviewer_id"] = "reviewer_2"
        second["reference_source_sha256"] = "b" * 64
        result = validate_review_pair(first, second)
        self.assertIn(
            "REFERENCE_MISMATCH_REFERENCE_SOURCE_SHA256",
            result.errors,
        )

    def test_disagreement_reported(self):
        first = example()
        second = copy.deepcopy(first)
        second["reviewer_id"] = "reviewer_2"
        second["observed_maneuver"] = "rash_turn"
        self.assertIn(
            "REVIEWER_DISAGREEMENT",
            validate_review_pair(first, second).errors,
        )

    def test_agreement_does_not_confirm_ground_truth(self):
        first = example()
        second = copy.deepcopy(first)
        second["reviewer_id"] = "reviewer_2"
        result = validate_review_pair(first, second)
        self.assertTrue(result.valid)
        self.assertFalse(result.ground_truth_confirmed)

    def test_input_is_not_modified(self):
        record = example()
        before = copy.deepcopy(record)
        validate_review(record)
        self.assertEqual(record, before)


if __name__ == "__main__":
    unittest.main()
