from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Union
import yaml


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All file-path settings consumed by a gridops simulation run.

    Every attribute is a string holding a file path (relative or absolute).
    The fields map directly to the YAML keys in the configuration file.

    Input files
    -----------
    generator_parameters_file:
        CSV with generator parameters (name, typ, node, maxcap, etc.).
    generator_matrix_file:
        CSV mapping each generator to its node
        (rows = generators, columns = nodes, values = 0/1).
    line_to_bus_file:
        CSV mapping each transmission line to nodes via DC-OPF incidence
        (rows = lines, columns = nodes, values = -1/0/1).
    line_parameters_file:
        CSV with line parameters (line, reactance, limit).
    daily_hydro_maximum_file:
        CSV of maximum hourly hydro generation per node (rows = days, columns = nodes).
    daily_hydro_minimum_file:
        CSV of minimum hourly hydro generation per node (rows = days, columns = nodes).
    daily_hydro_total_file:
        CSV of daily total hydro energy budget per node (rows = days, columns = nodes).
    nodal_solar_file:
        CSV of hourly available solar generation per node (rows = hours, columns = nodes).
    nodal_wind_file:
        CSV of hourly available wind generation per node (rows = hours, columns = nodes).
    nodal_offshore_wind_file:
        CSV of hourly available offshore-wind generation per node (rows = hours, columns = nodes).
    nodal_load_file:
        CSV of hourly load per node (rows = hours, columns = nodes).
    must_run_file:
        CSV of constant must-run (e.g., nuclear) capacity per node
        (single row showing the capacities, columns = only nodes with must-run generators).
    fuel_prices_file:
        CSV of daily fuel prices per thermal generator (rows = days, columns = thermal generators).
    ba_to_ba_hurdle_scaled_file:
        CSV mapping each BA-to-BA exchange to its hurdle rate ($/MWh).
    ba_to_ba_transmission_matrix_file:
        CSV mapping each BA-to-BA exchange to the lines it covers
        (rows = exchanges, columns = lines).
    storage_params_file:
        CSV with storage-unit parameters (name, charge_rate, etc.).
    bus_to_storage_matrix_file:
        CSV mapping each storage unit to its node
        (rows = storage units, columns = nodes, values = 0/1).
    generator_outage_file:
        NumPy `.npy` file (dict) mapping each outage-group name to a list of
        generator names in that group.
    thermal_generators_file:
        CSV listing of thermal generators including nuclear units
        (used to compute must-run capacity adjustments).
    lost_capacity_file:
        CSV of hourly lost capacity per outage group (rows = hours, columns = groups).
        The first column must be ``Time`` (1-indexed integer).

    Output files
    ------------
    vlt_angle_file:    Parquet - hourly nodal voltage angles.
    mwh_file:          Parquet - hourly generator dispatch (MWh per unit).
    slack_file:        Parquet - hourly unserved demand per node.
    flow_file:         Parquet - hourly transmission line flows per line.
    duals_file:        Parquet - hourly nodal locational marginal prices (LMPs).
    storage_soc_file:  Parquet - hourly storage state-of-charge per storage-unit.
    storage_discharge_file: Parquet - hourly storage discharge per storage-unit.
    storage_charge_file:    Parquet - hourly storage charge per storage-unit.

    Other settings
    --------------
    restart_file_directory:
        Directory where cloudpickle restart files are written/read.
    """

    # ---------------------------------------------------------------------------
    # Input files
    # ---------------------------------------------------------------------------
    generator_parameters_file: str
    generator_matrix_file: str
    line_to_bus_file: str
    line_parameters_file: str
    daily_hydro_maximum_file: str
    daily_hydro_minimum_file: str
    daily_hydro_total_file: str
    nodal_solar_file: str
    nodal_wind_file: str
    nodal_offshore_wind_file: str
    nodal_load_file: str
    must_run_file: str
    fuel_prices_file: str
    ba_to_ba_hurdle_scaled_file: str
    ba_to_ba_transmission_matrix_file: str
    storage_params_file: str
    bus_to_storage_matrix_file: str
    generator_outage_file: str
    thermal_generators_file: str
    lost_capacity_file: str

    # ---------------------------------------------------------------------------
    # Output files
    # ---------------------------------------------------------------------------
    vlt_angle_file: str
    mwh_file: str
    slack_file: str
    flow_file: str
    duals_file: str
    storage_soc_file: str
    storage_discharge_file: str
    storage_charge_file: str

    # ---------------------------------------------------------------------------
    # Restart files directory
    # ---------------------------------------------------------------------------
    restart_file_directory: str



# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def read_config_file(config_file: str) -> dict:
    """Read a YAML configuration file and return its contents as a dict.

    Parameters
    ----------
    config_file:
        Absolute or relative path to the YAML file.

    Returns
    -------
    dict
        Raw key-value pairs from the YAML file.
    """
    with open(config_file, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def generate_config(
    config_file: Union[str, None] = None,
    **kwargs,
) -> Config:
    """Create a :class:`Config` from a YAML file or keyword arguments.

    If *config_file* is provided the YAML is parsed first; any *kwargs* are
    ignored.  If *config_file* is ``None`` the *kwargs* are passed directly to
    the :class:`Config` constructor.

    Parameters
    ----------
    config_file:
        Path to the YAML configuration file, or ``None``.
    **kwargs:
        Keyword arguments forwarded to :class:`Config` when *config_file* is
        ``None``.

    Returns
    -------
    Config
        Fully populated configuration object.
    """
    logger = logging.getLogger(__name__)

    if config_file is None:
        config = Config(**kwargs)
    else:
        logger.info(f"Loading configuration from: {config_file}")
        config_dict = read_config_file(config_file)
        config = Config(**config_dict)

    # Log every attribute for traceability
    for key, val in config.__dict__.items():
        logger.info(f"{key} = {val}")

    return config
