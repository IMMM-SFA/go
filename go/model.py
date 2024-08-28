import os
import logging
from typing import Union, List

import numpy as np

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
    # for HiGHS:  https://ergo-code.github.io/HiGHS/stable/options/definitions/#random_seed
    MAX_RANDOM_SEED_VALUE = 2147483647

    def __init__(
            self,
            region: str,
            problem: str,
            complexity: str,
            solver_name: str = "appsi_highs",
            solver_params: Union[None, dict] = None,
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

        self.logger = logging.getLogger(__name__)

        if region == "west" and problem == "linear" and complexity == "multi":
            self.model = west_linear_multi
            self.logger.info(f"Using configuration:  {region}_{problem}_{complexity}")

        else:
            config_error_msg = f"Configuration: {region}_{problem}_{complexity} is not currently supported."
            self.logger.error(config_error_msg)
            raise AssertionError(config_error_msg)

        if solver_name not in ('appsi_highs', 'gurobi', 'cplex'):
            solver_error_msg = f"Solver '{solver_name}' not in available solvers:  'appsi_highs', 'gurobi', and 'cplex'"
            self.logger.error(solver_error_msg)
            raise AssertionError(solver_error_msg)

        else:
            self.solver_name = solver_name

        self.solver_params = solver_params

    def triage(        
        self, 
        config_file: Union[str, None] = None, 
        restart_file: Union[None, str] = None,
        n_days: int = 365,
        n_seeds: int = 4,
        triage_solver_list: List[str] = ["ipm"],
        **kwargs
    ):

        self.logger.warning("Solver triage mode initiated. Trials will now begin.")

        # restart protocol for trial testing
        restart_protocol = {
            "order": ["random_seed", "solver"],
            "trials": {
                "random_seed": {"n_seeds": n_seeds},
                "solver": triage_solver_list,
            }
        }

        # compile a list of trials
        trial_list = []
        for parameter in restart_protocol["order"]:

            # fetch the trial
            trial = restart_protocol["trials"][parameter]

            if parameter == "random_seed" and trial["n_seeds"] > 0: 
                seeds = [np.random.choice(Model.MAX_RANDOM_SEED_VALUE) for i in range(trial["n_seeds"])]

                for random_seed in seeds:
                    trial_list.append({"random_seed": random_seed})

            elif parameter == "solver" and len(trial) > 0:

                for solver in trial:
                    trial_list.append({"solver": solver})

            else:
                self.logger.warning("Trail parameter '{}' not yet supported.")

        # run each trial until it either succeeds or fails
        success = False
        for modification in trial_list:

            # copy of the original parameters to modify so we do not inherit any previous modifications
            local_solver_parameters = self.solver_params.copy()

            # update solver paramters with trial parameters
            local_solver_parameters.update(modification)

            try:
                success_day = self.model(
                    config_file=config_file,
                    solver_name=self.solver_name,
                    solver_params=local_solver_parameters,
                    restart_file=restart_file,
                    **kwargs
                )

                success = True 
                break

            # if it fails, try the next one
            except Exception as e:
                solver_exception = e
                pass 

        if success:
            self.logger.info("Solver triage mode solution achieved.  Reverting back to original solver settings.")

            if n_days > success_day:
                self.run(
                    config_file=config_file,
                    restart_file=restart_file,
                    n_days=n_days,
                    **kwargs
                )

            else:
                self.logger.info("All days completed successfully.")

        else:
            self.logger.error("Solver triage mode unable to find solution.  Exiting.")
            raise solver_exception

    def run(
        self, 
        config_file: Union[str, None] = None, 
        restart_file: Union[None, str] = None,
        n_days: int = 365,
        **kwargs
    ):
        """
        Run the GO model with a specified configuration.

        :param config_file: The configuration file to use. If None, the default configuration is used.
        :type config_file: Union[str, None]

        :param restart_file:        Full path to cloudpickled restart file.
        :type restart_file:         Union[None, str]; Default None

        :param n_days:              The number of the day in the calendar year to process through.
        :type n_days:               int; Default 365
        
        :param kwargs:              Additional keyword arguments to pass to the model.
        :type kwargs:               dict
        """

        try:

            self.model(
                config_file=config_file,
                solver_name=self.solver_name,
                solver_params=self.solver_params,
                restart_file=restart_file,
                n_days=n_days,
                **kwargs
            )

            self.logger.info("All days completed successfully.")

        except:

            self.triage(
                config_file=config_file,
                restart_file=restart_file,
                n_days=n_days,
                **kwargs  
            )




