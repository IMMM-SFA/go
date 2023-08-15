from dataclasses import dataclass
import logging
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
    generator_outage_file: str
    dat_file: str
    thermal_generators_file: str
    lost_capacity_file: str
    vlt_angle_file: str
    mwh_file: str
    slack_file: str
    flow_file: str
    duals_file: str


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

    config_parts = config.__dict__
    for i in config_parts.keys():
        logger.info(f"{i} = {config_parts[i]}")

    return config
