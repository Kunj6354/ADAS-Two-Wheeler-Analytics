"""Adversarial tests for draft fold provenance."""

from copy import deepcopy
import unittest

from adas.evaluation.fold_provenance import (
    PROVENANCE_VERSION,
    WINDOW_IDENTITY_VERSION,
    validate_fold_provenance,
)


class FoldProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.runtime = {
            "python": "3.12.10",
            "numpy": "2.5.4",
            "scipy": "1.18.1",
        }
        self.solver = {
            "method": "scipy.optimize.milp",
            "objective": "zero-feasibility",
            "time_limit_seconds": 60.0,
        }
        self.digest = "a" * 64
        self.provenance = {
            "provenance_schema_version": PROVENANCE_VERSION,
            "window_identity_version": WINDOW_IDENTITY_VERSION,
            "window_inventory_sha256": self.digest,
            "eligible_window_count": 3478,
            "runtime": deepcopy(self.runtime),
            "solver_configuration": deepcopy(self.solver),
        }

    def validate(self, value=None):
        return validate_fold_provenance(
            self.provenance if value is None else value,
            expected_window_sha256=self.digest,
            expected_window_count=3478,
            expected_runtime=self.runtime,
            expected_solver_configuration=self.solver,
        )

    def changed(self, field, value):
        obj = deepcopy(self.provenance)
        obj[field] = value
        return obj

    def test_valid(self):
        result = self.validate()
        self.assertTrue(result["valid"])
        self.assertEqual(result["approval_status"], "NOT_APPROVED")

    def test_repeatable(self):
        self.assertEqual(self.validate(), self.validate())

    def test_wrong_window_digest(self):
        with self.assertRaises(ValueError):
            self.validate(self.changed(
                "window_inventory_sha256", "b" * 64
            ))

    def test_wrong_window_count(self):
        with self.assertRaises(ValueError):
            self.validate(self.changed(
                "eligible_window_count", 3477
            ))

    def test_boolean_window_count(self):
        with self.assertRaises(ValueError):
            self.validate(self.changed(
                "eligible_window_count", True
            ))

    def test_wrong_identity_version(self):
        with self.assertRaises(ValueError):
            self.validate(self.changed(
                "window_identity_version", "unsupported"
            ))

    def test_wrong_provenance_version(self):
        with self.assertRaises(ValueError):
            self.validate(self.changed(
                "provenance_schema_version", "unsupported"
            ))

    def test_wrong_runtime(self):
        obj = deepcopy(self.provenance)
        obj["runtime"]["scipy"] = "changed"
        with self.assertRaises(ValueError):
            self.validate(obj)

    def test_missing_runtime_field(self):
        obj = deepcopy(self.provenance)
        del obj["runtime"]["numpy"]
        with self.assertRaises(ValueError):
            self.validate(obj)

    def test_wrong_solver_method(self):
        obj = deepcopy(self.provenance)
        obj["solver_configuration"]["method"] = "unknown"
        with self.assertRaises(ValueError):
            self.validate(obj)

    def test_wrong_solver_time_limit(self):
        obj = deepcopy(self.provenance)
        obj["solver_configuration"]["time_limit_seconds"] = 30.0
        with self.assertRaises(ValueError):
            self.validate(obj)

    def test_missing_field(self):
        obj = deepcopy(self.provenance)
        del obj["window_inventory_sha256"]
        with self.assertRaises(ValueError):
            self.validate(obj)

    def test_unexpected_field(self):
        obj = deepcopy(self.provenance)
        obj["approved"] = True
        with self.assertRaises(ValueError):
            self.validate(obj)


if __name__ == "__main__":
    unittest.main()
