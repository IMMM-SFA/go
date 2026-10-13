import os
import tempfile
import unittest
import yaml
from gridops.configuration import Config, generate_config, read_config_file


# ---------------------------------------------------------------------------
# Minimal valid kwargs covering every Config field
# ---------------------------------------------------------------------------
MINIMAL_KWARGS = {
    "generator_parameters_file":         "Input/data_genparams.csv",
    "generator_matrix_file":             "Input/gen_mat.csv",
    "line_to_bus_file":                  "Input/line_to_bus.csv",
    "line_parameters_file":              "Input/line_param.csv",
    "daily_hydro_maximum_file":          "Input/hydro_max.csv",
    "daily_hydro_minimum_file":          "Input/hydro_min.csv",
    "daily_hydro_total_file":            "Input/hydro_total.csv",
    "nodal_solar_file":                  "Input/nodal_solar.csv",
    "nodal_wind_file":                   "Input/nodal_wind.csv",
    "nodal_offshore_wind_file":          "Input/nodal_offshore_wind.csv",
    "nodal_load_file":                   "Input/nodal_load.csv",
    "must_run_file":                     "Input/must_run.csv",
    "fuel_prices_file":                  "Input/fuel_prices.csv",
    "ba_to_ba_hurdle_scaled_file":       "Input/BA_to_BA_hurdle_scaled.csv",
    "ba_to_ba_transmission_matrix_file": "Input/BA_to_BA_transmission_matrix.csv",
    "storage_params_file":               "Input/storage_params.csv",
    "bus_to_storage_matrix_file":        "Input/storage_mat.csv",
    "generator_outage_file":             "Input/gen_outage_cat.npy",
    "thermal_generators_file":           "Input/thermal_gens.csv",
    "lost_capacity_file":                "Input/lost_capacity.csv",
    "vlt_angle_file":                    "Output/vlt_angle.parquet",
    "mwh_file":                          "Output/mwh.parquet",
    "slack_file":                        "Output/slack.parquet",
    "flow_file":                         "Output/flow.parquet",
    "duals_file":                        "Output/duals.parquet",
    "storage_soc_file":                  "Output/storage_soc.parquet",
    "storage_discharge_file":            "Output/storage_discharge.parquet",
    "storage_charge_file":               "Output/storage_charge.parquet",
    "restart_file_directory":            "Restart/",
}


class TestConfigDataclass(unittest.TestCase):
    """Tests for the Config dataclass."""

    def test_create_from_kwargs(self):
        """Config can be instantiated from keyword arguments."""
        config = Config(**MINIMAL_KWARGS)
        self.assertEqual(config.generator_parameters_file, "Input/data_genparams.csv")
        self.assertEqual(config.restart_file_directory, "Restart/")

    def test_all_fields_present(self):
        """Every expected field is present in a valid Config instance."""
        config = Config(**MINIMAL_KWARGS)
        required = list(MINIMAL_KWARGS.keys())
        for field in required:
            self.assertTrue(hasattr(config, field), msg=f"Missing field: {field}")

    def test_field_types_are_str(self):
        """All Config field values are strings."""
        config = Config(**MINIMAL_KWARGS)
        for field, val in config.__dict__.items():
            self.assertIsInstance(val, str, msg=f"Field {field} should be a string")

    def test_missing_required_field_raises(self):
        """Config raises TypeError when a required field is missing."""
        bad_kwargs = {k: v for k, v in MINIMAL_KWARGS.items()
                      if k != "generator_parameters_file"}
        with self.assertRaises(TypeError):
            Config(**bad_kwargs)


class TestReadConfigFile(unittest.TestCase):
    """Tests for read_config_file."""

    def test_roundtrip_yaml(self):
        """read_config_file correctly parses a YAML written by yaml.dump."""
        data = {"key_a": "value_a", "key_b": "value_b"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, encoding="utf-8"
        ) as fh:
            yaml.dump(data, fh)
            tmp_path = fh.name

        try:
            result = read_config_file(tmp_path)
            self.assertEqual(result, data)
        finally:
            os.unlink(tmp_path)


class TestGenerateConfig(unittest.TestCase):
    """Tests for generate_config."""

    def test_from_kwargs(self):
        """generate_config works with keyword arguments when config_file=None."""
        config = generate_config(config_file=None, **MINIMAL_KWARGS)
        self.assertIsInstance(config, Config)
        self.assertEqual(config.mwh_file, "Output/mwh.parquet")

    def test_from_yaml_file(self):
        """generate_config correctly loads a YAML and returns a Config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, encoding="utf-8"
        ) as fh:
            yaml.dump(MINIMAL_KWARGS, fh)
            tmp_path = fh.name
        try:
            config = generate_config(config_file=tmp_path)
            self.assertIsInstance(config, Config)
            self.assertEqual(config.duals_file, "Output/duals.parquet")
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
