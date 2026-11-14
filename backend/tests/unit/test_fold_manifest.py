"""Adversarial draft fold-manifest validation tests."""

from copy import deepcopy
from dataclasses import asdict
import unittest

from adas.evaluation.fold_manifest import (
    SCHEMA_VERSION,
    canonical_partition_sha256,
    validate_fold_manifest,
)
from adas.evaluation.grouped_folds import FoldLimits


class FoldManifestTests(unittest.TestCase):

    def setUp(self):
        self.classes = ("normal", "rash")
        self.limits = FoldLimits(groups_per_fold=2)

        self.counts = {
            name: {"normal": 20, "rash": 20}
            for name in "ABCDEFGH"
        }

        self.folds = {
            "fold_1": ("A", "B"),
            "fold_2": ("C", "D"),
            "fold_3": ("E", "F"),
            "fold_4": ("G", "H"),
        }

        self.dataset_hash = "a" * 64
        self.component_hashes = {
            "loader": "b" * 64,
            "fold_solver": "c" * 64,
            "grouped_folds": "d" * 64,
        }

        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "approval_status": "DRAFT",
            "dataset_manifest_sha256": self.dataset_hash,
            "component_sha256": dict(self.component_hashes),
            "class_names": list(self.classes),
            "fold_limits": asdict(self.limits),
            "folds": deepcopy(self.folds),
            "canonical_partition_sha256":
                canonical_partition_sha256(self.folds),
            "recording_group_count": 8,
            "eligible_window_count": 320,
        }

    def validate(self, manifest=None, **overrides):
        args = {
            "expected_dataset_sha256": self.dataset_hash,
            "expected_component_sha256": self.component_hashes,
            "limits": self.limits,
        }
        args.update(overrides)

        return validate_fold_manifest(
            self.manifest if manifest is None else manifest,
            self.counts,
            self.classes,
            **args,
        )

    def altered(self, key, value):
        result = deepcopy(self.manifest)
        result[key] = value
        return result

    def test_valid_draft(self):
        result = self.validate()
        self.assertTrue(result["valid"])
        self.assertEqual(result["approval_status"], "DRAFT")

    def test_repeatability(self):
        self.assertEqual(self.validate(), self.validate())

    def test_partition_fingerprint_ignores_fold_names(self):
        renamed = {
            "z": ("B", "A"),
            "w": ("H", "G"),
            "y": ("F", "E"),
            "x": ("D", "C"),
        }
        self.assertEqual(
            canonical_partition_sha256(self.folds),
            canonical_partition_sha256(renamed),
        )

    def test_rejects_unsupported_version(self):
        with self.assertRaises(ValueError):
            self.validate(
                self.altered("schema_version", "unsupported")
            )

    def test_rejects_false_approval(self):
        with self.assertRaises(ValueError):
            self.validate(
                self.altered("approval_status", "APPROVED")
            )

    def test_rejects_missing_field(self):
        modified = deepcopy(self.manifest)
        del modified["class_names"]
        with self.assertRaises(ValueError):
            self.validate(modified)

    def test_rejects_unknown_field(self):
        modified = deepcopy(self.manifest)
        modified["unexpected"] = True
        with self.assertRaises(ValueError):
            self.validate(modified)

    def test_rejects_changed_dataset_hash(self):
        with self.assertRaises(ValueError):
            self.validate(
                self.altered("dataset_manifest_sha256", "f" * 64)
            )

    def test_rejects_changed_component_hash(self):
        modified = deepcopy(self.manifest)
        modified["component_sha256"]["fold_solver"] = "f" * 64
        with self.assertRaises(ValueError):
            self.validate(modified)

    def test_rejects_missing_component(self):
        modified = deepcopy(self.manifest)
        del modified["component_sha256"]["loader"]
        with self.assertRaises(ValueError):
            self.validate(modified)

    def test_rejects_class_order_change(self):
        with self.assertRaises(ValueError):
            self.validate(
                self.altered(
                    "class_names", ["rash", "normal"]
                )
            )

    def test_rejects_limit_change(self):
        modified = deepcopy(self.manifest)
        modified["fold_limits"]["maximum_class_concentration"] = 0.9
        with self.assertRaises(ValueError):
            self.validate(modified)

    def test_rejects_wrong_partition_digest(self):
        with self.assertRaises(ValueError):
            self.validate(
                self.altered(
                    "canonical_partition_sha256", "0" * 64
                )
            )

    def test_rejects_duplicate_recording_even_with_updated_hash(self):
        modified = deepcopy(self.manifest)
        modified["folds"]["fold_4"] = ("A", "H")
        modified["canonical_partition_sha256"] = (
            canonical_partition_sha256(modified["folds"])
        )
        with self.assertRaises(ValueError):
            self.validate(modified)

    def test_rejects_missing_recording_even_with_updated_hash(self):
        modified = deepcopy(self.manifest)
        modified["folds"]["fold_4"] = ("G",)
        modified["canonical_partition_sha256"] = (
            canonical_partition_sha256(modified["folds"])
        )
        with self.assertRaises(ValueError):
            self.validate(modified)

    def test_rejects_wrong_recording_count(self):
        with self.assertRaises(ValueError):
            self.validate(
                self.altered("recording_group_count", 7)
            )

    def test_rejects_wrong_window_count(self):
        with self.assertRaises(ValueError):
            self.validate(
                self.altered("eligible_window_count", 319)
            )

    def test_rejects_boolean_window_count(self):
        with self.assertRaises(ValueError):
            self.validate(
                self.altered("eligible_window_count", True)
            )


if __name__ == "__main__":
    unittest.main()
