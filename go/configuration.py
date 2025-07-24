from dataclasses import dataclass
import logging
import os
from typing import Union

import yaml


def read_config_file(config_file: str) -> dict:
    """
    Reads the configuration file and returns the configuration as a dictionary.

    Parameters:
    -----------
    config_file : str
        The path to the configuration file.

    Returns:
    --------
    dict:
        The configuration as a dictionary.
    """
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    """
    A data class that represents the configuration for the application.

    Attributes
    ----------
    generator_parameters_file : str
        The path to the generator parameters file.
    generator_matrix_file : str
        The path to the generator matrix file.
    line_to_bus_file : str
        The path to the line to bus file.
    line_parameters_file : str
        The path to the line parameters file.
    daily_hydro_maximum_file : str
        The path to the daily hydro maximum file.
    daily_hydro_minimum_file : str
        The path to the daily hydro minimum file.
    daily_hydro_total_file : str
        The path to the daily hydro total file.
    nodal_solar_file : str
        The path to the nodal solar file.
    nodal_wind_file : str
        The path to the nodal wind file.
    nodal_offshore_wind_file : str
        The path to the nodal offshore wind file.
    nodal_load_file : str
        The path to the nodal load file.
    must_run_file : str
        The path to the must run file.
    fuel_prices_file : str
        The path to the fuel prices file.
    ba_to_ba_hurdle_scaled_file : str
        The path to the BA to BA hurdle scaled file.
    ba_to_ba_transmission_matrix_file : str
        The path to the BA to BA transmission matrix file.
    storage_params_file : str
        The path to the storage parameters file.
    bus_to_storage_matrix_file : str
        The path to the bus to storage matrix file.
    generator_outage_file : str
        The path to the generator outage cat file.
    dat_file : str
        The path to the dat file.
    thermal_generators_file : str
        The path to the thermal generators file.
    lost_capacity_file : str
        The path to the lost capacity file.
    vlt_angle_file : str
        The path to the vlt angle file.
    mwh_file : str
        The path to the mwh file.
    slack_file : str
        The path to the slack file.
    flow_file : str
        The path to the flow file.
    duals_file : str
        The path to the duals file.
    SoC_file : str
        The path to the storage SoC file.
    discharge_file : str
        The path to the storage discharge file.
    charge_file : str
        The path to the storage charge file.
    restart_file_directory: str
        The directory path to save the model restart file to

    """
    
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
    dat_file: str
    thermal_generators_file: str
    lost_capacity_file: str
    vlt_angle_file: str
    mwh_file: str
    slack_file: str
    flow_file: str
    duals_file: str
    SoC_file: str
    discharge_file: str
    charge_file: str
    restart_file_directory: str


def generate_config(config_file: Union[str, None] = None, **kwargs) -> Config:
    """
    Generates a Config object from a configuration file.

    Parameters:
    -----------
    config_file : str
        The path to the configuration file.

    Returns:
    --------
    Config:
        A Config object with attributes set according to the configuration file.
    """

    logger = logging.getLogger(__name__)

    if config_file is None:
        config = Config(**kwargs)

    else:
        logger.info(f"Project configuration file:  {config_file}")

        config_dict = read_config_file(config_file)
        config = Config(**config_dict)

    # ------------------------------------------------------------------
    # Path sanitization: transparently map remote /rcfs paths to the local
    #                    project data mirror.  Any config field that is a
    #                    string beginning with "/rcfs" is rewritten to
    #                    start with the local prefix so downstream code
    #                    does not need to handle this logic explicitly.
    # ------------------------------------------------------------------
    LOCAL_RCSF_PREFIX = "/Users/d3y010/projects/go/data/rcfs"

    for field_name, value in list(config.__dict__.items()):
        if isinstance(value, str) and value.startswith("/rcfs"):
            # Preserve the remainder of the original path while replacing
            # the root.  Avoid duplicate slashes by stripping any leading
            # slash from the remainder.
            remainder = value[len("/rcfs"):].lstrip("/")
            new_value = os.path.join(LOCAL_RCSF_PREFIX, remainder)
            setattr(config, field_name, new_value)

    config_parts = config.__dict__
    for i in config_parts.keys():
        logger.info(f"{i} = {config_parts[i]}")

    return config
