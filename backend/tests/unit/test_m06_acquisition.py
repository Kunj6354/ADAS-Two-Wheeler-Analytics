"""Synthetic tests of M06 acquisition metadata completeness."""

import copy
import unittest

from adas.data.m06_acquisition import (
    REQUIRED_FIELDS, validate_session_metadata,
)


def fixture():
    return {
        "session_id": "SYNTHETIC_SESSION",
        "sensor_identity": {"model": "MPU6050"},
        "firmware_sha256": "a" * 64,
        "mounting_record": {"description": "synthetic fixture"},
        "calibration_record": {"description": "synthetic fixture"},
        "timebase_record": {"description": "synthetic fixture"},
        "reference_evidence": {"description": "synthetic fixture"},
        "annotation_policy_version": "draft",
        "collection_approval_reference": "SYNTHETIC_NOT_APPROVED",
    }


class AcquisitionMetadataTests(unittest.TestCase):
    def test_all_contract_fields_present(self):
        self.assertEqual(len(REQUIRED_FIELDS), 9)

    def test_complete_draft_not_authorized(self):
        result = validate_session_metadata(fixture())
        self.assertTrue(result.structurally_complete)
        self.assertFalse(result.acquisition_authorized)
        self.assertFalse(result.ground_truth_verified)

    def test_non_mapping_rejected(self):
        result = validate_session_metadata([])
        self.assertIn("NOT_A_MAPPING", result.errors)

    def test_missing_fields_rejected(self):
        for field in REQUIRED_FIELDS:
            with self.subTest(field=field):
                record = fixture()
                del record[field]
                result = validate_session_metadata(record)
                self.assertIn(
                    f"MISSING_{field.upper()}", result.errors
                )

    def test_bad_firmware_hash_rejected(self):
        for value in ("bad", "z" * 64, [], None):
            with self.subTest(value=value):
                record = fixture()
                record["firmware_sha256"] = value
                self.assertIn(
                    "INVALID_FIRMWARE_SHA256",
                    validate_session_metadata(record).errors,
                )

    def test_missing_mounting_details(self):
        record = fixture()
        record["mounting_record"] = {}
        self.assertIn(
            "INCOMPLETE_MOUNTING_RECORD",
            validate_session_metadata(record).errors,
        )

    def test_missing_reference_details(self):
        record = fixture()
        record["reference_evidence"] = None
        self.assertIn(
            "INCOMPLETE_REFERENCE_EVIDENCE",
            validate_session_metadata(record).errors,
        )

    def test_empty_calibration_description(self):
        record = fixture()
        record["calibration_record"] = {"description": " "}
        self.assertIn(
            "INCOMPLETE_CALIBRATION_RECORD",
            validate_session_metadata(record).errors,
        )

    def test_text_field_rejected(self):
        record = fixture()
        record["session_id"] = []
        self.assertIn(
            "INVALID_SESSION_ID",
            validate_session_metadata(record).errors,
        )

    def test_no_input_mutation(self):
        record = fixture()
        before = copy.deepcopy(record)
        validate_session_metadata(record)
        self.assertEqual(record, before)

    def test_fake_approval_never_enables_collection(self):
        record = fixture()
        record["collection_approval_reference"] = "APPROVED"
        result = validate_session_metadata(record)
        self.assertTrue(result.structurally_complete)
        self.assertFalse(result.acquisition_authorized)


if __name__ == "__main__":
    unittest.main()
