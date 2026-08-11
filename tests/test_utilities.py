from __future__ import annotations
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import cloudpickle
from gridops.utilities import (
    clear_restart_files,
    get_prior_restart_file_day,
    get_restart_file,
    load_solver_parameters,
    write_restart_file,
    write_solver_parameters,
)


class TestWriteRestartFile(unittest.TestCase):
    """Tests for write_restart_file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=False)

    def test_day_zero_returns_none(self):
        """write_restart_file with day=0 returns None without writing any file."""
        result = write_restart_file(self.tmpdir, 0, {"data": "test"})
        self.assertIsNone(result)
        # No files should have been created
        self.assertEqual(len(list(Path(self.tmpdir).glob("*.pkl"))), 0)

    def test_creates_file_correct_name(self):
        """The created file follows the model_restart_file_dayDDD_vVVV.pkl naming."""
        fp = write_restart_file(self.tmpdir, 5, {"key": "val"})
        self.assertIsNotNone(fp)
        self.assertTrue(Path(fp).is_file())
        self.assertIn("day005", str(fp))
        self.assertIn("_v000", str(fp))
        self.assertTrue(str(fp).endswith(".pkl"))

    def test_version_auto_increments(self):
        """Repeated writes for the same day create distinct versioned files."""
        data = {"x": 1}
        fp0 = write_restart_file(self.tmpdir, 3, data)
        fp1 = write_restart_file(self.tmpdir, 3, data)
        self.assertNotEqual(str(fp0), str(fp1))
        self.assertIn("_v000", str(fp0))
        self.assertIn("_v001", str(fp1))

    def test_data_is_recoverable(self):
        """Data written by write_restart_file can be read back with cloudpickle."""
        restart_data = {"epoch": 7, "weights": [0.1, 0.9]}
        fp = write_restart_file(self.tmpdir, 10, restart_data)
        with open(fp, "rb") as fh:
            loaded = cloudpickle.load(fh)
        self.assertEqual(loaded["epoch"], 7)
        self.assertAlmostEqual(loaded["weights"][1], 0.9)

    def test_day_padding(self):
        """Day number is zero-padded to 3 digits in the file name."""
        fp = write_restart_file(self.tmpdir, 1, {})
        self.assertIn("day001", str(fp))

        fp2 = write_restart_file(self.tmpdir, 100, {})
        self.assertIn("day100", str(fp2))


class TestGetRestartFile(unittest.TestCase):
    """Tests for get_restart_file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=False)

    def test_empty_dir_returns_none(self):
        """get_restart_file returns None when the directory is empty."""
        result = get_restart_file(self.tmpdir)
        self.assertIsNone(result)

    def test_returns_lexicographically_last_file(self):
        """With day=None, the latest (lexicographically last) file is returned."""
        write_restart_file(self.tmpdir, 1, {})
        write_restart_file(self.tmpdir, 5, {})

        result = get_restart_file(self.tmpdir)
        self.assertIsNotNone(result)
        self.assertIn("day005", result)

    def test_specific_day_lookup(self):
        """get_restart_file returns the correct file when a specific day is given."""
        write_restart_file(self.tmpdir, 3, {})
        write_restart_file(self.tmpdir, 7, {})

        result = get_restart_file(self.tmpdir, day=3)
        self.assertIn("day003", result)

    def test_specific_day_latest_version(self):
        """When multiple versions exist for a day, the latest version is returned."""
        write_restart_file(self.tmpdir, 4, {"v": 0})
        write_restart_file(self.tmpdir, 4, {"v": 1})

        result = get_restart_file(self.tmpdir, day=4)
        self.assertIn("_v001", result)

    def test_missing_day_raises(self):
        """get_restart_file raises Exception when a requested day has no file."""
        with self.assertRaises(Exception):
            get_restart_file(self.tmpdir, day=99)


class TestGetPriorRestartFileDay(unittest.TestCase):
    """Tests for get_prior_restart_file_day."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=False)

    def test_empty_dir_returns_none(self):
        """Returns None when there are no restart files."""
        result = get_prior_restart_file_day(self.tmpdir)
        self.assertIsNone(result)

    def test_single_day_returns_none(self):
        """Returns None when only one day-group exists."""
        write_restart_file(self.tmpdir, 5, {})
        result = get_prior_restart_file_day(self.tmpdir)
        self.assertIsNone(result)

    def test_two_days_returns_first(self):
        """Returns the earlier day number when two day-groups exist."""
        write_restart_file(self.tmpdir, 5, {})
        write_restart_file(self.tmpdir, 10, {})
        result = get_prior_restart_file_day(self.tmpdir)
        self.assertEqual(result, 5)

    def test_three_days_returns_second(self):
        """Returns the second-to-last day when three day-groups exist."""
        write_restart_file(self.tmpdir, 2, {})
        write_restart_file(self.tmpdir, 8, {})
        write_restart_file(self.tmpdir, 15, {})
        result = get_prior_restart_file_day(self.tmpdir)
        self.assertEqual(result, 8)


class TestClearRestartFiles(unittest.TestCase):
    """Tests for clear_restart_files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=False)

    def test_empty_dir_returns_zero(self):
        """Returns 0 when the directory has no restart files."""
        result = clear_restart_files(self.tmpdir)
        self.assertEqual(result, 0)

    def test_deletes_all_restart_files(self):
        """All restart pickle files are removed."""
        write_restart_file(self.tmpdir, 1, {})
        write_restart_file(self.tmpdir, 5, {})
        write_restart_file(self.tmpdir, 10, {})
        count = clear_restart_files(self.tmpdir)
        self.assertEqual(count, 3)
        remaining = list(Path(self.tmpdir).glob("model_restart_file_day*.pkl"))
        self.assertEqual(len(remaining), 0)

    def test_does_not_delete_other_files(self):
        """Non-restart files in the directory are preserved."""
        write_restart_file(self.tmpdir, 1, {})
        other_file = Path(self.tmpdir) / "solver_parameters.json"
        other_file.write_text("{}")
        clear_restart_files(self.tmpdir)
        self.assertTrue(other_file.is_file())

    def test_get_restart_returns_none_after_clear(self):
        """After clearing, get_restart_file returns None."""
        write_restart_file(self.tmpdir, 3, {})
        clear_restart_files(self.tmpdir)
        result = get_restart_file(self.tmpdir)
        self.assertIsNone(result)


class TestSolverParameterIO(unittest.TestCase):
    """Tests for write_solver_parameters and load_solver_parameters."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fpath = os.path.join(self.tmpdir, "solver_params.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=False)

    def test_roundtrip(self):
        """Params written and then loaded back are identical."""
        params = {0: {"time_limit": 600}, 3: {"Seed": 42, "threads": 4}}
        write_solver_parameters(params, self.fpath)
        loaded = load_solver_parameters(self.fpath)

        self.assertEqual(loaded[0]["time_limit"], 600)
        self.assertEqual(loaded[3]["Seed"], 42)
        self.assertEqual(loaded[3]["threads"], 4)

    def test_keys_are_integers_after_load(self):
        """JSON serialisation turns int keys to strings; load restores them as int."""
        params = {0: {}, 5: {}, 23: {}}
        write_solver_parameters(params, self.fpath)
        loaded = load_solver_parameters(self.fpath)

        for k in loaded:
            self.assertIsInstance(k, int, msg=f"Key {k} should be integer")

    def test_empty_dict_roundtrip(self):
        """An empty dict roundtrips cleanly."""
        write_solver_parameters({}, self.fpath)
        loaded = load_solver_parameters(self.fpath)
        self.assertEqual(loaded, {})

    def test_none_values_are_preserved(self):
        """None values (e.g., user didn't set options) survive JSON roundtrip."""
        params = {1: None, 2: {"time_limit": 300}}
        write_solver_parameters(params, self.fpath)
        loaded = load_solver_parameters(self.fpath)

        self.assertIsNone(loaded[1])
        self.assertEqual(loaded[2]["time_limit"], 300)


if __name__ == "__main__":
    unittest.main()
