from __future__ import annotations
import unittest
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from gridops.west.linear import build_west_linear


def _make_minimal_data(horizon_hours: int = 2) -> dict:
    """Return a minimal synthetic ``sim`` data dict for unit testing.

    The network has one bus (``bus_100011``, the hardcoded reference bus),
    one coal generator, one gas generator, no lines, no exchanges, and no
    storage.  Mutable Horizon parameters are left at their initial value of
    zero; a test that needs meaningful values should update them after
    calling :func:`~gridops.west.linear.build_west_linear`.

    Parameters
    ----------
    horizon_hours:
        Number of hours in the horizon (default 2).

    Returns
    -------
    dict
        Minimal ``sim_data`` dict accepted by
        :func:`~gridops.west.linear.build_west_linear`.
    """
    nodes = ["bus_100011"]
    lines = []
    exchanges = []
    storage_units = []

    gen_sets = {
        "Coal":        ["coal_1"],
        "Oil":         [],
        "Gas":         ["gas_1"],
        "Hydro":       [],
        "Solar":       [],
        "Wind":        [],
        "Biomass":     [],
        "Geothermal":  [],
        "OffshoreWind": [],
    }

    gen_params = {
        "coal_1": {
            "typ":       "coal",
            "bus":      "bus_100011",
            "maxcap":    400.0,
            "mincap":    50.0,
            "heat_rate": 10.5,
            "var_om":    2.0,
            "no_load":   0.0,
            "st_cost":   100.0,
            "ramp":      150.0,
            "minup":     8.0,
            "mindn":     8.0,
        },
        "gas_1": {
            "typ":       "ngcc",
            "bus":      "bus_100011",
            "maxcap":    200.0,
            "mincap":    30.0,
            "heat_rate": 7.5,
            "var_om":    3.0,
            "no_load":   0.0,
            "st_cost":   50.0,
            "ramp":      100.0,
            "minup":     4.0,
            "mindn":     4.0,
        },
    }

    return {
        # Sets
        "gen_sets":          gen_sets,
        "outage_group_sets": {},
        "buses":             nodes,
        "lines":             lines,
        "exchanges":         exchanges,
        "storage_units":     storage_units,
        # Static params
        "gen_params":        gen_params,
        "line_params":       {},
        "exchange_hurdle":   {},
        "storage_params":    {},
        # Sparse mapping dicts
        "bus_to_unit_map": {
            ("coal_1", "bus_100011"): 1.0,
            ("gas_1",  "bus_100011"): 1.0,
        },
        "line_to_bus_map":    {},
        "exchange_map":       {},
        "bus_to_storage_map": {},
        # Initial states (both zero for construction test)
        "initial_gen": {},
        "initial_soc": {},
    }


class TestBuildWestLinearMulti(unittest.TestCase):
    """Verify that build_west_linear returns a correct ConcreteModel."""

    def setUp(self):
        self.data = _make_minimal_data(horizon_hours=2)
        self.model = build_west_linear(self.data, horizon_hours=2)

    def test_returns_concrete_model(self):
        """The return type is a Pyomo ConcreteModel."""
        self.assertIsInstance(self.model, pyo.ConcreteModel)

    def test_fuel_type_sets_populated(self):
        """Coal and Gas sets contain the expected generator names."""
        self.assertIn("coal_1", list(self.model.Coal))
        self.assertIn("gas_1", list(self.model.Gas))

    def test_empty_sets_created(self):
        """Empty generator sets (Hydro, Solar, …) are defined but contain no members."""
        for set_name in ["Hydro", "Solar", "Wind", "Biomass", "Geothermal", "OffshoreWind"]:
            self.assertTrue(hasattr(self.model, set_name))
            self.assertEqual(len(list(getattr(self.model, set_name))), 0)

    def test_derived_sets_thermal(self):
        """Generators are under Thermal sets.
        Thermal = Coal | Oil | Gas | Biomass | Geothermal."""
        thermal = set(self.model.Thermal)
        self.assertIn("coal_1", thermal)
        self.assertIn("gas_1", thermal)

    def test_derived_sets_generators(self):
        """Generators includes every generator in the data."""
        gens = set(self.model.Generators)
        self.assertIn("coal_1", gens)
        self.assertIn("gas_1", gens)

    def test_time_sets_range(self):
        """time_periods = 1..H."""
        self.assertEqual(list(self.model.time_periods), [1, 2])
        
    def test_buses_populated(self):
        """model.buses contains the reference bus."""
        self.assertIn("bus_100011", list(self.model.buses))

    def test_maxcap_param(self):
        """model.maxcap stores the correct generator capacities."""
        self.assertAlmostEqual(pyo.value(self.model.maxcap["coal_1"]), 400.0)
        self.assertAlmostEqual(pyo.value(self.model.maxcap["gas_1"]),  200.0)

    def test_ramp_param(self):
        """model.ramp stores the correct ramp rates."""
        self.assertAlmostEqual(pyo.value(self.model.ramp["coal_1"]), 150.0)
        self.assertAlmostEqual(pyo.value(self.model.ramp["gas_1"]),  100.0)

    def test_horizon_demand_initialised_to_zero(self):
        """HorizonDemand is a mutable param initialised to 0."""
        val = pyo.value(self.model.HorizonDemand["bus_100011", 1])
        self.assertAlmostEqual(val, 0.0)

    def test_horizon_gen_limit_initialised_to_zero(self):
        """HorizonGenLimit is a mutable param initialised to 0."""
        val = pyo.value(self.model.HorizonGenLimit["coal_1", 1])
        self.assertAlmostEqual(val, 0.0)

    def test_mutable_demand_can_be_updated(self):
        """HorizonDemand can be updated in-place (confirms mutable=True)."""
        self.model.HorizonDemand["bus_100011", 1] = 250.0
        val = pyo.value(self.model.HorizonDemand["bus_100011", 1])
        self.assertAlmostEqual(val, 250.0)

    def test_decision_variables_exist(self):
        """All expected decision variable components are present."""
        for var_name in ["mwh", "S", "Flow", "Theta", "DummyFlow", "SoC", "Charge", "Discharge"]:
            self.assertTrue(hasattr(self.model, var_name), msg=f"Missing variable: {var_name}")

    def test_mwh_hour0_is_fixed(self):
        """InitialMwh mutable param exists and defaults to 0 for Dispatchable generators."""
        self.assertTrue(hasattr(self.model, "InitialMwh"))
        for gen in ["coal_1", "gas_1"]:
            self.assertAlmostEqual(pyo.value(self.model.InitialMwh[gen]), 0.0)
        
    def test_mwh_hour1_is_not_fixed(self):
        """mwh[gen, 1] is a free variable (not fixed)."""
        for gen in ["coal_1", "gas_1"]:
            self.assertFalse(
                self.model.mwh[gen, 1].is_fixed(),
                msg=f"mwh[{gen}, 1] should not be fixed",
            )

    def test_objective_exists(self):
        """SystemCost objective is registered on the model."""
        self.assertTrue(hasattr(self.model, "SystemCost"))

    def test_ramp_constraints_exist(self):
        """RampCon1 and RampCon2 are defined on the model."""
        self.assertTrue(hasattr(self.model, "RampCon1"))
        self.assertTrue(hasattr(self.model, "RampCon2"))

    def test_nodal_balance_constraint_exists(self):
        """Bus_Constraint is registered on the model."""
        self.assertTrue(hasattr(self.model, "Bus_Constraint"))
    def test_dual_suffix_exists(self):
        """model.dual Suffix is created (for extracting LMPs)."""
        self.assertTrue(hasattr(self.model, "dual"))


class TestSolveWestLinearMulti(unittest.TestCase):
    """Attempt a full LP solve using HiGHS on a minimal 1-bus model.

    All tests in this class are skipped if HiGHS is not available.
    """

    @classmethod
    def setUpClass(cls):
        opt = SolverFactory("appsi_highs")
        if not opt.available():
            raise unittest.SkipTest("appsi_highs not available - skipping solve tests.")
        cls.opt = opt

    def _build_and_load(self, horizon_hours: int = 2, demand: float = 100.0):
        """Build the model and update mutable params for a feasible solve."""
        data = _make_minimal_data(horizon_hours=horizon_hours)
        model = build_west_linear(data, horizon_hours=horizon_hours)

        # Update mutable params
        for b in model.buses:
            for t in model.time_periods:
                model.HorizonDemand[b, t] = demand

        # Lift capacity limits to maxcap so the generators can actually dispatch
        for g in model.Thermal:
            cap = pyo.value(model.maxcap[g])
            for t in model.time_periods:
                model.HorizonGenLimit[g, t] = cap

        # Set fuel prices so the cost function is non-trivial
        for g in model.Thermal:
            model.FuelPrice[g] = 3.0

        return model

    def test_solves_to_optimality(self):
        """Model solves to an optimal solution for a 100 MW constant demand."""
        model = self._build_and_load(horizon_hours=2, demand=100.0)

        result = self.opt.solve(model, tee=False)
        term_cond = result.solver.termination_condition

        self.assertEqual(term_cond, pyo.TerminationCondition.optimal)

    def test_demand_is_met(self):
        """Generation + slack equals demand at each hour (system balance check)."""
        demand = 120.0
        model = self._build_and_load(horizon_hours=2, demand=demand)
        self.opt.solve(model, tee=False)

        for t in model.time_periods:
            total_gen = sum(pyo.value(model.mwh[g, t]) for g in model.Generators)
            total_slack = sum(pyo.value(model.S[b, t]) for b in model.buses)
            # Must-run is 0 in this minimal test; no lines, so net_import = 0
            self.assertAlmostEqual(total_gen + total_slack, demand, places=1)

    def test_gas_preferred_over_coal(self):
        """Gas (cheaper) is dispatched before coal when demand is small."""
        # Gas cost: 7.5 × 3 + 3 = 25.5 $/MWh  vs  Coal: 10.5 × 3 + 2 = 33.5 $/MWh
        # With demand = 50 MW (below gas maxcap = 200 MW), gas should cover all.
        demand = 50.0
        model = self._build_and_load(horizon_hours=2, demand=demand)
        self.opt.solve(model, tee=False)

        coal_h1 = pyo.value(model.mwh["coal_1", 1])
        gas_h1  = pyo.value(model.mwh["gas_1",  1])

        self.assertAlmostEqual(gas_h1, demand, places=0)
        self.assertAlmostEqual(coal_h1, 0.0, places=0)

    def test_no_negative_generation(self):
        """All generator dispatch values are non-negative after solving."""
        model = self._build_and_load(horizon_hours=2, demand=100.0)
        self.opt.solve(model, tee=False)

        for g in model.Generators:
            for t in model.time_periods:
                val = pyo.value(model.mwh[g, t])
                self.assertGreaterEqual(val, -1e-6, msg=f"mwh[{g},{t}] = {val:.6f} < 0")


if __name__ == "__main__":
    unittest.main()

