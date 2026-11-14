"""Unit tests for recording-group isolation contracts."""

import unittest

from adas.evaluation.group_isolation import validate_group_partitions


class GroupIsolationTests(unittest.TestCase):

    def test_accepts_disjoint_complete_partitions(self):
        result = validate_group_partitions(
            {
                "train": ["MAIN9.CSV", "MAIN11.CSV"],
                "validation": ["MAIN17.CSV"],
                "test": ["MAIN22.CSV"],
            },
            expected_groups=[
                "MAIN9.CSV",
                "MAIN11.CSV",
                "MAIN17.CSV",
                "MAIN22.CSV",
            ],
        )
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result["train"], frozenset)

    def test_rejects_cross_partition_group_reuse(self):
        with self.assertRaises(ValueError):
            validate_group_partitions(
                {
                    "train": ["MAIN9.CSV", "MAIN11.CSV"],
                    "test": ["MAIN11.CSV"],
                }
            )

    def test_rejects_duplicate_group_inside_partition(self):
        with self.assertRaises(ValueError):
            validate_group_partitions(
                {"train": ["MAIN9.CSV", "MAIN9.CSV"]}
            )

    def test_rejects_missing_expected_group(self):
        with self.assertRaises(ValueError):
            validate_group_partitions(
                {"train": ["MAIN9.CSV"]},
                expected_groups=["MAIN9.CSV", "MAIN11.CSV"],
            )

    def test_rejects_unexpected_group(self):
        with self.assertRaises(ValueError):
            validate_group_partitions(
                {"train": ["MAIN9.CSV", "MAIN11.CSV"]},
                expected_groups=["MAIN9.CSV"],
            )

    def test_rejects_empty_partition(self):
        with self.assertRaises(ValueError):
            validate_group_partitions({"train": []})

    def test_rejects_string_instead_of_group_iterable(self):
        with self.assertRaises(TypeError):
            validate_group_partitions({"train": "MAIN9.CSV"})

    def test_rejects_invalid_group_identifier(self):
        with self.assertRaises(ValueError):
            validate_group_partitions({"train": ["MAIN9.CSV", ""]})

    def test_rejects_duplicate_expected_group(self):
        with self.assertRaises(ValueError):
            validate_group_partitions(
                {"train": ["MAIN9.CSV"]},
                expected_groups=["MAIN9.CSV", "MAIN9.CSV"],
            )

    def test_rejects_missing_partition_mapping(self):
        with self.assertRaises(ValueError):
            validate_group_partitions({})


if __name__ == "__main__":
    unittest.main()
