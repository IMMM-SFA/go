from typing import Union

from go.west.launch import west_linear_multi


class Model:
    """
    This is the main Model class that is used to run the GO model with a specified configuration.

    :param region: The region for the model.
    :type region: str

    :param problem: The problem for the model.
    :type problem: str

    :param complexity: The complexity for the model.
    :type complexity: str
    """

    def __init__(self, region: str, problem: str, complexity: str):
        """
        Initialize the Model with the specified region, problem, and complexity.

        :param region: The region for the model. Either 'west', 'ercot', or 'east'
        :type region: str

        :param problem: The problem for the model. Either 'linear' or 'mip'
        :type problem: str

        :param complexity: The complexity for the model. Either 'simple' or 'multi'
        :type complexity: str
        """
        
        error_msg = f"Configuration: {region}_{problem}_{complexity} is not currently supported."

        if region == "west" and problem == "linear" and complexity == "multi":
            self.model = west_linear_multi

        else:
            raise AssertionError(error_msg)

    def run(self, config_file: Union[str, None] = None, **kwargs):
        """
        Run the GO model with a specified configuration.

        :param config_file: The configuration file to use. If None, the default configuration is used.
        :type config_file: Union[str, None]
        
        :param kwargs: Additional keyword arguments to pass to the model.
        :type kwargs: dict
        """

        self.model(config_file=config_file, **kwargs)
