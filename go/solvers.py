import pyomo.environ as pyo
from pyomo.opt import SolverFactory


class GoSolver:
    SUPPORTED = ("appsi_highs", "cplex", "gurobi")

    def __init__(self, solver_name, **kwargs):

        self.solver_name = self.validate_name(solver_name)
        self.go_solver = SolverFactory(self.solver_name)
        self.params = kwargs

        # apply options
        self.option_generator()

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
        """Setup HiGHS options.  Use defaults of no parameter values are specified.

        See full list here:  https://ergo-code.github.io/HiGHS/stable/options/definitions/#option-definitions
        """

        self.go_solver.options["presolve"] = self.params.get("presolve", "choose")
        self.go_solver.options["solver"] = self.params.get("solver", "simplex")
        self.go_solver.options["parallel"] = self.params.get("parallel", "on")
        self.go_solver.options["run_crossover"] = self.params.get("run_crossover", "on")
        self.go_solver.options["time_limit"] = self.params.get("time_limit", 3600)
        self.go_solver.options["threads"] = self.params.get("threads", 8)
        self.go_solver.options["simplex_strategy"] = self.params.get("simplex_strategy", 2)  # dual simplex

    def set_cplex_options(self):
        """Setup CPLEX options. Use defaults of no parameter values are specified."""

        pass

    def set_gurobi_options(self):
        """Setup GUROBI options. Use defaults of no parameter values are specified."""

        pass
