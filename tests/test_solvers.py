import unittest
from gridops.solvers import GoSolver


class TestGoSolverHiGHS(unittest.TestCase):
    """Tests for GoSolver with the HiGHS backend."""

    def test_creates_without_error(self):
        """GoSolver instantiation with appsi_highs without error."""
        solver = GoSolver(solver_name="appsi_highs", solver_params=None)
        self.assertIsNotNone(solver.go_solver)

    def test_no_default_options_injected(self):
        """When solver_params=None, no options are written to the solver."""
        solver = GoSolver(solver_name="appsi_highs", solver_params=None)
        self.assertEqual(dict(solver.go_solver.options), {})

    def test_user_options_applied(self):
        """Provided solver_params are written to go_solver.options."""
        params = {"time_limit": 600, "solver": "simplex"}
        solver = GoSolver(solver_name="appsi_highs", solver_params=params)
        self.assertEqual(solver.go_solver.options["time_limit"], 600)
        self.assertEqual(solver.go_solver.options["solver"], "simplex")


class TestGoSolverGurobi(unittest.TestCase):
    """Tests for GoSolver with the Gurobi backend."""

    def test_creates_without_error(self):
        """GoSolver instantiation with appsi_gurobi without error (no license needed)."""
        # SolverFactory creation does not check for a license; it only
        # checks the license when .solve() is called.
        solver = GoSolver(solver_name="appsi_gurobi", solver_params=None)
        self.assertIsNotNone(solver.go_solver)

    def test_seed_option_applied(self):
        """Seed option is forwarded to Gurobi solver."""
        solver = GoSolver(solver_name="appsi_gurobi", solver_params={"Seed": 123})
        self.assertEqual(solver.go_solver.options["Seed"], 123)


class TestGoSolverInvalidName(unittest.TestCase):
    """Tests that raise an error when an unsupported solver is requested."""

    def test_unknown_name_raises(self):
        """Any unrecognised solver name raises KeyError."""
        with self.assertRaises(KeyError):
            GoSolver(solver_name="glpk")

    def test_empty_name_raises(self):
        """An empty string raises KeyError."""
        with self.assertRaises(KeyError):
            GoSolver(solver_name="")


if __name__ == "__main__":
    unittest.main()
