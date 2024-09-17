import unittest
from unittest.mock import mock_open, patch

from go.utilities import *


class TestLoadSolverParameters(unittest.TestCase):
    def test_load_solver_parameters(self):
        mock_json_content = '{"1": {"solver": "simplex"}, "2": {"solver": "ipm"}}'
        expected_output = {
            1: {"solver": "simplex"},
            2: {"solver": "ipm"}
        }

        with patch("builtins.open", mock_open(read_data=mock_json_content)):
            result = load_solver_parameters("dummy_path")
            self.assertEqual(result, expected_output)

    def test_load_solver_parameters_empty(self):
        mock_json_content = '{}'
        expected_output = {}

        with patch("builtins.open", mock_open(read_data=mock_json_content)):
            result = load_solver_parameters("dummy_path")
            self.assertEqual(result, expected_output)

    def test_load_solver_parameters_invalid_json(self):
        mock_json_content = '{"1": {"solver": "simplex", "2": {"solver": "ipm"}}'  # Invalid JSON

        with patch("builtins.open", mock_open(read_data=mock_json_content)):
            with self.assertRaises(json.JSONDecodeError):
                load_solver_parameters("dummy_path")


if __name__ == "__main__":
    unittest.main()
