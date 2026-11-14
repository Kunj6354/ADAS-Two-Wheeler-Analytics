"""Synthetic nested metadata regression for M06-E1D."""

import copy
import unittest

from adas.data.m06_acquisition import validate_session_metadata


def fixture():
    return {
        "session_id": "SYNTHETIC",
        "sensor_identity": {"model": "MPU6050"},
        "firmware_sha256": "a" * 64,
        "mounting_record": {"description": "synthetic"},
        "calibration_record": {"description": "synthetic"},
        "timebase_record": {"description": "synthetic"},
        "reference_evidence": {"description": "synthetic"},
        "annotation_policy_version": "draft",
        "collection_approval_reference": "NOT_APPROVED",
    }


class NestedMetadataTests(unittest.TestCase):

    def check_invalid(self, field, value):
        record = fixture()
        record[field] = value
        before = copy.deepcopy(record)
        result = validate_session_metadata(record)
        self.assertFalse(result.structurally_complete)
        self.assertIn(f"INCOMPLETE_{field.upper()}", result.errors)
        self.assertFalse(result.acquisition_authorized)
        self.assertFalse(result.ground_truth_verified)
        self.assertEqual(record, before)

    def test_nested_empty_mounting(self):
        self.check_invalid(
            "mounting_record",
            {"position": "front", "axes": {}},
        )

    def test_nested_empty_calibration(self):
        self.check_invalid(
            "calibration_record", {"offsets": []}
        )

    def test_invalid_reference_digest(self):
        self.check_invalid(
            "reference_evidence", {"sha256": "bad"}
        )

    def test_boolean_sensor_model(self):
        self.check_invalid(
            "sensor_identity", {"model": True}
        )

    def test_nonfinite_numeric_calibration(self):
        self.check_invalid(
            "calibration_record", {"offsets": [float("nan")]}
        )

    def test_nested_valid_shape_not_authorized(self):
        record = fixture()
        record["mounting_record"] = {
            "position": "synthetic",
            "axes": {"x": "forward", "y": "left"},
        }
        record["calibration_record"] = {"offsets": [0.0, 0.0]}
        record["reference_evidence"] = {"sha256": "b" * 64}
        result = validate_session_metadata(record)
        self.assertTrue(result.structurally_complete)
        self.assertFalse(result.acquisition_authorized)
        self.assertFalse(result.ground_truth_verified)

    def test_fake_approval_still_not_authorized(self):
        record = fixture()
        record["collection_approval_reference"] = "APPROVED"
        result = validate_session_metadata(record)
        self.assertFalse(result.acquisition_authorized)

    def test_excessive_nesting_rejected(self):
        nested = "leaf"
        for _ in range(20):
            nested = {"nested": nested}
        self.check_invalid("timebase_record", nested)


if __name__ == "__main__":
    unittest.main()
