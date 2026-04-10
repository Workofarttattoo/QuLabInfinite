"""
Tests ensuring dataset registry metadata is available and loaders fail gracefully.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from chemistry_lab.datasets import DATASET_REGISTRY, get_dataset, list_datasets  # noqa: E402


class TestDatasetRegistry(unittest.TestCase):
    """Validate dataset descriptors."""

    def test_registry_contains_expected_entries(self):
        keys = set(list_datasets())
        expected = {
            "qm9s",
            "qcml",
            "gdb9_ex9",
            "ornl_aisd_ex10",
            "ames_quantum",
            "openqdc",
            "nmsu_hydrocarbon_ir",
            "metaboanalyst",
            "quick_qm_spectra",
            "spc2csv",
        }
        self.assertTrue(expected.issubset(keys))

    def test_descriptor_has_minimum_fields(self):
        for name, descriptor in DATASET_REGISTRY.items():
            info = descriptor.as_dict()
            self.assertTrue(info["name"], msg=name)
            self.assertTrue(info["url"], msg=name)
            self.assertIn(descriptor.category, info["category"])
            self.assertGreater(len(info["file_extensions"]), 0, msg=name)

    def test_loader_handles_missing_file(self):
        descriptor = get_dataset("qm9s")
        self.assertIsNotNone(descriptor)
        fake_path = Path("does_not_exist.csv")
        with self.assertRaises(FileNotFoundError):
            descriptor.load_metadata(fake_path)
        sample = descriptor.load_sample_rows(limit=3)
        self.assertIsInstance(sample, list)


if __name__ == "__main__":
    unittest.main()
