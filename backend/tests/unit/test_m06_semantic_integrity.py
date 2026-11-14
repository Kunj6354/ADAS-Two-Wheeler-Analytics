"""M06-E2I frozen contract and exact-field boundary regression."""

import copy
import unittest

from adas.data.m06_semantic import validate_semantic_metadata
from test_m06_semantic import fixture, PARENT, NESTED


def check(record, parent=None, nested=None):
    return validate_semantic_metadata(
        record,
        copy.deepcopy(PARENT) if parent is None else parent,
        copy.deepcopy(NESTED) if nested is None else nested,
    )


class SemanticIntegrityTests(unittest.TestCase):

    def test_original_contracts_and_fixture(self):
        result = check(fixture())
        self.assertTrue(result.structurally_valid, result.errors)
        self.assertFalse(result.scientific_evidence_verified)
        self.assertFalse(result.acquisition_authorized)
        self.assertFalse(result.ground_truth_verified)

    def test_removed_parent_field_rejected(self):
        parent = copy.deepcopy(PARENT)
        del parent["field_definitions"]["sensor_configuration"][
            "sensor_model"
        ]
        self.assertIn(
            "CONTRACT_INTEGRITY_MISMATCH",
            check(fixture(), parent=parent).errors,
        )

    def test_altered_parent_type_rejected(self):
        parent = copy.deepcopy(PARENT)
        parent["field_definitions"]["sensor_configuration"][
            "sensor_model"
        ]["proposed_type"] = "sha256"
        self.assertIn(
            "CONTRACT_INTEGRITY_MISMATCH",
            check(fixture(), parent=parent).errors,
        )

    def test_removed_nested_definition_rejected(self):
        nested = copy.deepcopy(NESTED)
        del nested["nested_definitions"][
            "calibration.offset_values"
        ]["fields"]["calibration_run_id"]
        self.assertIn(
            "CONTRACT_INTEGRITY_MISMATCH",
            check(fixture(), nested=nested).errors,
        )

    def test_changed_approval_flag_rejected(self):
        parent = copy.deepcopy(PARENT)
        parent["policy"]["ground_truth_verified"] = True
        self.assertFalse(
            check(fixture(), parent=parent).structurally_valid
        )

    def test_malformed_contract_rejected(self):
        parent = copy.deepcopy(PARENT)
        parent["field_definitions"] = object()
        result = check(fixture(), parent=parent)
        self.assertFalse(result.structurally_valid)
        self.assertIn(
            "INVALID_CONTRACT_SERIALIZATION", result.errors
        )

    def test_unknown_top_level_group(self):
        record = fixture()
        record["unexpected_group"] = {"test": "synthetic"}
        self.assertIn(
            "UNKNOWN_GROUP:unexpected_group",
            check(record).errors,
        )

    def test_unknown_group_field(self):
        record = fixture()
        record["sensor_configuration"]["unexpected"] = "synthetic"
        self.assertIn(
            "UNKNOWN_FIELD:sensor_configuration.unexpected",
            check(record).errors,
        )

    def test_unknown_nested_field(self):
        record = fixture()
        record["physical_mounting"]["axis_mapping"][
            "unexpected"
        ] = "synthetic"
        self.assertIn(
            "UNKNOWN_NESTED_FIELD:physical_mounting.axis_mapping.unexpected",
            check(record).errors,
        )

    def test_negative_uncertainty(self):
        record = fixture()
        record["timing"]["alignment_uncertainty_ms"] = -5.0
        self.assertIn(
            "NEGATIVE_ALIGNMENT_UNCERTAINTY",
            check(record).errors,
        )

    def test_zero_uncertainty_is_structurally_valid(self):
        record = fixture()
        record["timing"]["alignment_uncertainty_ms"] = 0.0
        self.assertTrue(check(record).structurally_valid)

    def test_nonfinite_uncertainty_rejected(self):
        record = fixture()
        record["timing"]["alignment_uncertainty_ms"] = float("inf")
        self.assertFalse(check(record).structurally_valid)

    def test_no_input_mutation(self):
        record = fixture()
        parent = copy.deepcopy(PARENT)
        nested = copy.deepcopy(NESTED)
        original = copy.deepcopy((record, parent, nested))

        check(record, parent, nested)

        self.assertEqual((record, parent, nested), original)

    def test_all_authorities_remain_false(self):
        record = fixture()
        result = check(record)
        self.assertFalse(result.scientific_evidence_verified)
        self.assertFalse(result.acquisition_authorized)
        self.assertFalse(result.ground_truth_verified)


if __name__ == "__main__":
    unittest.main()
