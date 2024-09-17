import unittest

from go.package_data import *


class TestPackageData(unittest.TestCase):
    """Tests for package data matching to confirm load function modification does not happen."""

    def test_get_data_directory(self):
        """Ensure package data functions do not get modified."""

        comp = pkg_resources.resource_filename('go', 'data')

        val = get_data_directory()

        self.assertEqual(comp, val)
