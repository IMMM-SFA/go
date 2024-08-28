import json 


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
    """
    with open(solver_parameter_file) as json_file:
        data = json.load(json_file)

        # convert keys back to integers so they can be referenced
        return {int(k): v for k, v in data.items()}
