import cloudpickle
from glob import glob
import json
from pathlib import Path


def load_solver_parameters(
    solver_parameter_file: str,
) -> dict:
    """
    Load solver parameters from a JSON file.

    This function reads a JSON file containing solver parameters, converts the keys 
    back to integers, and returns the parameters as a dictionary.

    :param solver_parameter_file: Path to the JSON file containing solver parameters.
    :type solver_parameter_file: str

    :return: Dictionary of solver parameters with integer keys.
    :rtype: dict

    Example:
    --------
    >>> params = load_solver_parameters('path/to/solver_parameters.json')
    >>> print(params)
    {1: 'value1', 2: 'value2', 3: 'value3'}
    """
    with open(solver_parameter_file) as json_file:
        data = json.load(json_file)

        # convert keys back to integers so they can be referenced
        return {int(k): v for k, v in data.items()}


def write_solver_parameters(
    solver_parameter_dictionary: dict,
    solver_parameter_file: str,
    indent: int = 4,
):
    """
    Write solver parameters to a JSON file.

    This function writes a dictionary of solver parameters to a specified JSON file 
    with a given indentation level.

    :param solver_parameter_dictionary: Dictionary containing solver parameters.
    :type solver_parameter_dictionary: dict

    :param solver_parameter_file: Path to the JSON file where solver parameters will be written.
    :type solver_parameter_file: str

    :param indent: Indentation level for the JSON file. Default is 4.
    :type indent: int
    """
    with open(solver_parameter_file, 'w') as json_file:
        json.dump(solver_parameter_dictionary, json_file, indent=indent)


def write_restart_file(
    dir: str,
    day: int,
    restart_data,
):
    """
    """

    if day==0:
        return None

    # path for this day's restart file
    version = 0
    fp = Path(f"{dir}/model_restart_file_day{str(day).zfill(3)}_v{str(version).zfill(3)}.pkl")

    # if a file already exists for this day, increment version
    while Path(fp).is_file() and (version < 101):
        version = version + 1
        fp = Path(f"{dir}/model_restart_file_day{str(day).zfill(3)}_v{str(version).zfill(3)}.pkl")
    
    if Path(fp).is_file():
        raise Exception("Too many restart files. Please clean up! Aborting.")
    
    with open(fp, "wb") as f:
        cloudpickle.dump(restart_data, f)
    
    return fp


def get_restart_file(
    dir: str,
    day: int|None = None,
):
    """
    """
    
    # get a list of all available restart files
    available_restart_files = sorted(glob(f"{dir}/model_restart_file_day*.pkl"))

    if day is None:
        # if none available, no restart needed
        if len(available_restart_files) == 0:
            return None
        
        # return latest
        return available_restart_files[-1]
    
    else:

        # if day requested, find that day's latest restart file
        day_files = [fp for fp in available_restart_files if f"{dir}/model_restart_file_day{str(day).zfill(3)}" in fp]

        if len(day_files) > 0:
            return day_files[-1]
    
    raise Exception(f"No restart file found for day {day}. Aborting.")


def get_prior_restart_file_day(
    dir: str,       
):
    """
    """

    # get a list of all available restart files
    available_restart_files = sorted(glob(f"{dir}/model_restart_file_day*.pkl"))

    if len(available_restart_files) > 0:

        # find the latest day available
        latest_day = "_".join(available_restart_files[-1].split("_")[:-1])
        prior_files = [fp for fp in available_restart_files if not latest_day in fp]
        
        if len(prior_files) > 0:
            prior_file = prior_files[-1]
            prior_day = prior_file.split("_")[-2].split("day")[-1]
            return int(prior_day)
    
    return None
