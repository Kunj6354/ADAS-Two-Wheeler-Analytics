"""Synthetic checks for draft M06 adjudication transitions."""

import copy
import unittest

from adas.data.m06_adjudication import (
    validate_transition, validate_history,
)


def event(previous="PENDING", current="UNRESOLVED", event_id="event_1"):
    return {
        "recording_id": "SYNTHETIC",
        "reference_source_id": "SYNTHETIC_REFERENCE",
        "reference_source_sha256": "a" * 64,
        "annotation_policy_version": "draft",
        "event_start_ms": 100,
        "event_end_ms": 500,
        "previous_state": previous,
        "current_state": current,
        "transition_reason": "Synthetic unresolved evidence",
        "decision_actor_id": "reviewer_1",
        "decision_timestamp": "2026-10-25T12:00:00Z",
        "audit_event_id": event_id,
    }


class AdjudicationTests(unittest.TestCase):

    def test_pending_to_unresolved(self):
        result = validate_transition(event())
        self.assertTrue(result.valid)
        self.assertFalse(result.ground_truth_confirmed)

    def test_confirmation_blocked(self):
        record = event(current="CONFIRMED")
        result = validate_transition(record)
        self.assertIn("CONFIRMATION_DISABLED", result.errors)
        self.assertFalse(result.ground_truth_confirmed)

    def test_reopening_requires_evidence(self):
        result = validate_transition(
            event("UNRESOLVED", "PENDING")
        )
        self.assertIn("REOPENING_EVIDENCE_REQUIRED", result.errors)

    def test_reopening_with_reference(self):
        record = event("UNRESOLVED", "PENDING")
        record["evidence_or_review_reference"] = "review_note_2"
        self.assertTrue(validate_transition(record).valid)

    def test_self_transition_rejected(self):
        result = validate_transition(event("PENDING", "PENDING"))
        self.assertIn("TRANSITION_NOT_ALLOWED", result.errors)

    def test_missing_audit_id(self):
        record = event()
        del record["audit_event_id"]
        self.assertIn(
            "MISSING_AUDIT_EVENT_ID", validate_transition(record).errors
        )

    def test_invalid_interval(self):
        record = event()
        record["event_start_ms"] = True
        self.assertIn(
            "INVALID_EVENT_INTERVAL", validate_transition(record).errors
        )

    def test_history_valid(self):
        first = event()
        second = event("UNRESOLVED", "PENDING", "event_2")
        second["evidence_or_review_reference"] = "new_review"
        self.assertTrue(validate_history([first, second]).valid)

    def test_duplicate_audit_id(self):
        first = event()
        second = event("UNRESOLVED", "PENDING")
        second["evidence_or_review_reference"] = "new_review"
        self.assertIn(
            "EVENT_1:DUPLICATE_AUDIT_ID",
            validate_history([first, second]).errors
        )

    def test_broken_history_chain(self):
        self.assertIn(
            "EVENT_1:BROKEN_STATE_CHAIN",
            validate_history([
                event(),
                event("PENDING", "UNRESOLVED", "event_2")
            ]).errors
        )

    def test_event_identity_change(self):
        first = event()
        second = event("UNRESOLVED", "PENDING", "event_2")
        second["evidence_or_review_reference"] = "new_review"
        second["recording_id"] = "OTHER"
        self.assertIn(
            "EVENT_1:EVENT_IDENTITY_CHANGED_RECORDING_ID",
            validate_history([first, second]).errors
        )

    def test_invalid_input_does_not_raise(self):
        self.assertFalse(validate_transition([]).valid)
        self.assertFalse(validate_history([None]).valid)

    def test_no_mutation(self):
        record = event()
        original = copy.deepcopy(record)
        validate_transition(record)
        validate_history([record])
        self.assertEqual(record, original)


if __name__ == "__main__":
    unittest.main()
