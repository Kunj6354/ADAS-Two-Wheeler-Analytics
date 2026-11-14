"""Synthetic tests for proposed M06-E2 semantic field checks."""

import copy
import json
import unittest
from pathlib import Path

from adas.data.m06_semantic import validate_semantic_metadata


ROOT = Path(__file__).resolve().parents[3]
PARENT = json.loads(
    (ROOT / "docs/M06_E2C_FIELD_DEFINITION_CONTRACT.json")
    .read_text(encoding="utf-8")
)
NESTED = json.loads(
    (ROOT / "docs/M06_E2E_NESTED_METADATA_CONTRACT.json")
    .read_text(encoding="utf-8")
)


def fixture():
    return {
        "sensor_configuration": {
            "sensor_model": "MPU6050",
            "device_identifier": "SYNTHETIC_DEVICE",
            "accelerometer_range_g": 2,
            "gyroscope_range_dps": 250,
            "accelerometer_output_unit": "g",
            "gyro_output_semantics": "integrated_angle",
            "firmware_sha256": "a" * 64,
            "library_sha256": "b" * 64,
        },
        "physical_mounting": {
            "mount_location": "synthetic_mount",
            "attachment_description": "synthetic_attachment",
            "axis_mapping": {
                "sensor_frame_id": "sensor",
                "vehicle_frame_id": "vehicle",
                "axis_correspondence": {"x": "forward"},
                "orientation_convention": "synthetic_convention",
            },
            "orientation_reference": "synthetic",
            "mounting_revision": "draft",
            "mounting_evidence_id": "synthetic_evidence",
        },
        "calibration": {
            "calibration_method": "synthetic_method",
            "stationary_condition_evidence": "synthetic_evidence",
            "offset_values": {
                "channel_values": {"x": 0.0, "y": 0.1},
                "calibration_run_id": "synthetic_run",
            },
            "offset_units": {
                "channel_units": {"x": "g", "y": "g"}
            },
            "calibration_outcome": "synthetic_outcome",
            "calibration_evidence_id": "synthetic_evidence",
        },
        "timing": {
            "sensor_clock_id": "sensor_clock",
            "timestamp_unit": "ms",
            "timestamp_origin": "synthetic_origin",
            "sampling_interval_observations": {
                "interval_count": 3,
                "minimum": 40.0,
                "median": 45.0,
                "maximum": 60.0,
                "unit": "ms",
            },
            "reference_clock_id": "reference_clock",
            "clock_alignment_method": "synthetic_method",
            "alignment_uncertainty_ms": 1.0,
        },
        "independent_reference": {
            "reference_source_id": "synthetic_reference",
            "reference_source_sha256": "c" * 64,
            "reference_acquisition_method": "synthetic_method",
            "recording_association": {
                "sensor_recording_id": "synthetic_recording",
                "association_method": "synthetic_method",
                "association_evidence_id": "synthetic_evidence",
            },
            "alignment_evidence_id": "synthetic_alignment",
            "reference_provenance": {
                "source_identity": "synthetic_source",
                "acquisition_record_id": "synthetic_acquisition",
                "custody_reference": "synthetic_custody",
            },
        },
    }


def check(record, parent=PARENT, nested=NESTED):
    return validate_semantic_metadata(record, parent, nested)


class SemanticMetadataTests(unittest.TestCase):

    def test_valid_shape_has_no_authority(self):
        result = check(fixture())
        self.assertTrue(result.structurally_valid, result.errors)
        self.assertFalse(result.scientific_evidence_verified)
        self.assertFalse(result.acquisition_authorized)
        self.assertFalse(result.ground_truth_verified)

    def test_missing_group(self):
        record = fixture()
        del record["timing"]
        self.assertFalse(check(record).structurally_valid)

    def test_missing_field(self):
        record = fixture()
        del record["sensor_configuration"]["sensor_model"]
        self.assertIn(
            "MISSING:sensor_configuration.sensor_model",
            check(record).errors,
        )

    def test_boolean_numeric_rejected(self):
        record = fixture()
        record["sensor_configuration"]["accelerometer_range_g"] = True
        self.assertFalse(check(record).structurally_valid)

    def test_invalid_sha_rejected(self):
        record = fixture()
        record["independent_reference"]["reference_source_sha256"] = "bad"
        self.assertFalse(check(record).structurally_valid)

    def test_missing_nested_field(self):
        record = fixture()
        del record["physical_mounting"]["axis_mapping"]["sensor_frame_id"]
        self.assertFalse(check(record).structurally_valid)

    def test_empty_axis_correspondence_rejected(self):
        record = fixture()
        record["physical_mounting"]["axis_mapping"]["axis_correspondence"] = {}
        self.assertFalse(check(record).structurally_valid)

    def test_mismatched_calibration_channels(self):
        record = fixture()
        del record["calibration"]["offset_units"]["channel_units"]["y"]
        self.assertIn(
            "CALIBRATION_CHANNEL_UNIT_MISMATCH",
            check(record).errors,
        )

    def test_reversed_sampling_intervals(self):
        record = fixture()
        record["timing"]["sampling_interval_observations"]["minimum"] = 80
        self.assertIn(
            "INCOHERENT_SAMPLING_INTERVALS",
            check(record).errors,
        )

    def test_zero_interval_count_rejected(self):
        record = fixture()
        record["timing"]["sampling_interval_observations"]["interval_count"] = 0
        self.assertIn(
            "INCOHERENT_SAMPLING_INTERVALS",
            check(record).errors,
        )

    def test_approved_contract_not_accepted(self):
        parent = copy.deepcopy(PARENT)
        parent["policy"]["ground_truth_verified"] = True
        self.assertFalse(check(fixture(), parent=parent).structurally_valid)

    def test_no_input_mutation(self):
        record = fixture()
        before = copy.deepcopy(record)
        check(record)
        self.assertEqual(record, before)

    def test_invalid_input_rejected(self):
        self.assertFalse(check([]).structurally_valid)


if __name__ == "__main__":
    unittest.main()
