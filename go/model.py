import logging
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

    def __init__(
            self,
            region: str,
            problem: str,
            complexity: str,
            solver_name: str = "appsi_highs",
            solver_params: Union[None, dict] = None
    ):
        """
        Initialize the Model with the specified region, problem, and complexity.

        :param region:              The region for the model. Either 'west', 'ercot', or 'east'
        :type region:               str

        :param problem:             The problem for the model. Either 'linear' or 'mip'
        :type problem:              str

        :param complexity:          The complexity for the model. Either 'simple' or 'multi'
        :type complexity:           str

        :param solver_name:         The solver to use.  Options are 'appsi_highs', 'gurobi', and 'cplex'
                                    Default: 'appsi_highs'
        :type solver_name:          str

        :param solver_params:       Parameter dictionary for the chosen solver to set options for the solver natively.
        :type solver_params:        Union[None, dict]; Default None

        """

        logger = logging.getLogger(__name__)

        if region == "west" and problem == "linear" and complexity == "multi":
            self.model = west_linear_multi
            logger.info(f"Using configuration:  {region}_{problem}_{complexity}")

        else:
            config_error_msg = f"Configuration: {region}_{problem}_{complexity} is not currently supported."
            logger.error(config_error_msg)
            raise AssertionError(config_error_msg)

        if solver_name not in ('appsi_highs', 'gurobi', 'cplex'):
            solver_error_msg = f"Solver '{solver_name}' not in available solvers:  'appsi_highs', 'gurobi', and 'cplex'"
            logger.error(solver_error_msg)
            raise AssertionError(solver_error_msg)

        else:
            self.solver_name = solver_name

        self.solver_params = solver_params

    def run(self, config_file: Union[str, None] = None, **kwargs):
        """
        Run the GO model with a specified configuration.

        :param config_file: The configuration file to use. If None, the default configuration is used.
        :type config_file: Union[str, None]
        
        :param kwargs: Additional keyword arguments to pass to the model.
        :type kwargs: dict
        """

        self.model(
            config_file=config_file,
            solver_name=self.solver_name,
            solver_params=self.solver_params,
            **kwargs
        )
