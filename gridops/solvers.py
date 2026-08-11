from __future__ import annotations
import logging
from typing import Union
from pyomo.opt import SolverFactory


class GoSolver:
    """Wrapper around Pyomo's `SolverFactory`.

    Validates the solver name, instantiates the solver, and applies any
    user-supplied options.

    Parameters
    ----------
    solver_name:
        One of ``"appsi_highs"``, ``"appsi_gurobi"``, or ``"gurobi"``.
    solver_params:
        Optional dict of solver-specific option name → value pairs.
        These are applied directly to the solver's ``options`` dict.
        Pass ``None`` (default) to use the solver's own built-in defaults.

    Attributes
    ----------
    go_solver:
        The configured Pyomo solver object, ready to call ``.solve()``.

    Examples
    --------
    HiGHS solver with a time limit::

        solver = GoSolver("appsi_highs", {"time_limit": 600})

    Gurobi with a fixed random seed::

        solver = GoSolver("appsi_gurobi", {"Seed": 123})
    """

    #: All solver names accepted by gridops
    SUPPORTED = ("appsi_highs", "appsi_gurobi", "gurobi")

    def __init__(
        self,
        solver_name: str = "appsi_highs",
        solver_params: Union[None, dict] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        # Validate and store the name
        self.solver_name = self._validate_name(solver_name)

        # Create the Pyomo solver object
        self.go_solver = SolverFactory(self.solver_name)

        # Apply user-supplied options (no hidden defaults are set here)
        self._apply_options(solver_params)

        # Log the final option set for transparency
        self.logger.info(f"Solver: {self.solver_name}")
        for key, val in self.go_solver.options.items():
            self.logger.info(f" Solver option: {key} = {val}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_name(solver_name: str) -> str:
        """Return *solver_name* unchanged, or raise :class:`KeyError`.

        Parameters
        ----------
        solver_name:
            Candidate solver name to check.

        Returns
        -------
        str
            The validated solver name.

        Raises
        ------
        KeyError
            If *solver_name* is not in :attr:`SUPPORTED`.
        """
        if solver_name not in GoSolver.SUPPORTED:
            raise KeyError(
                f"Solver '{solver_name}' is not supported. "
                f"Choose from: {GoSolver.SUPPORTED}"
            )
        return solver_name


    def _apply_options(self, solver_params: Union[None, dict]) -> None:
        """Write every key-value pair from *solver_params* into the solver's
        ``options`` dict.

        Parameters
        ----------
        solver_params:
            Mapping of solver option names to their desired values, or
            ``None`` to apply no options at all.
        """
        if solver_params is not None:
            for key, val in solver_params.items():
                self.go_solver.options[key] = val
