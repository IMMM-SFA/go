import logging
from typing import Union

from pyomo.opt import SolverFactory


class GoSolver:
    SUPPORTED = ("appsi_highs", "cplex", "gurobi")

    def __init__(
            self,
            solver_name: str = "appsi_highs",
            solver_params: Union[None, dict] = None
    ):

        self.logger = logging.getLogger(__name__)

        self.solver_name = self.validate_name(solver_name)
        self.go_solver = SolverFactory(self.solver_name)
        self.params = solver_params

        # apply options
        self.option_generator()

        # log custom options
        self.logger.info(f"Solver name:  {self.solver_name}")

        for i in self.go_solver.options.keys():
            self.logger.info(f"Customized solver parameters:  {i} = {self.go_solver.options[i]}")

    @staticmethod
    def validate_name(solver_name):
        """Validate solver name."""

        if solver_name not in GoSolver.SUPPORTED:
            raise KeyError(
                f"Solver '{solver_name}' is not currently supported.  Use 'appsi_highs', 'cplex', or 'gurobi'"
            )

        return solver_name

    def option_generator(self):
        """Apply options based on solver and user desired parameters."""

        if self.solver_name == "appsi_highs":
            self.set_highs_options()

        elif self.solver_name == "cplex":
            self.set_cplex_options()

        elif self.solver_name == "gurobi":
            self.set_gurobi_options()

    def set_highs_options(self):
        """Setup HiGHS options.  Use defaults if no parameter values are specified.
        Some options are preset due to optimal run configuration for GO on a cluster.  However,
        any of these settings can be modified.

        See full list here:  https://ergo-code.github.io/HiGHS/stable/options/definitions/#option-definitions
        """

        # initialize presets
        self.go_solver.options["presolve"] = "choose"
        self.go_solver.options["solver"] = "simplex"
        self.go_solver.options["parallel"] = "on"
        self.go_solver.options["run_crossover"] = "on"
        self.go_solver.options["time_limit"] = 3600
        self.go_solver.options["threads"] = 8
        self.go_solver.options["simplex_strategy"] = 2  # dual simplex

        # update
        if self.params is not None:

            for i in self.params.keys():
                self.go_solver.options[i] = self.params[i]

    def set_cplex_options(self):
        """Setup CPLEX options. Use defaults of no parameter values are specified."""

        # update
        if self.params is not None:

            for i in self.params.keys():
                self.go_solver.options[i] = self.params[i]

    def set_gurobi_options(self):
        """Setup GUROBI options. Use defaults of no parameter values are specified."""

        # update
        if self.params is not None:

            for i in self.params.keys():
                self.go_solver.options[i] = self.params[i]

