from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from gridops import configuration
from gridops.solvers import GoSolver
from gridops.utilities import (
    clear_restart_files,
    write_restart_file,
    get_restart_file,
    write_solver_parameters,
)
from gridops.west.linear import build_west_linear


# ---------------------------------------------------------------------------
# Numerical snap tolerance
# Any value whose absolute magnitude is below SNAP_TOL is snapped to the
# nearest physical boundary (0.0).  
# ---------------------------------------------------------------------------
SNAP_TOL: float = 1e-6

# ---------------------------------------------------------------------------
# Fuel-type → Pyomo set name mapping
# ---------------------------------------------------------------------------

# Maps the ``typ`` field values from ``generator_parameters_file`` to the
# canonical Pyomo set names used by the model.
FUEL_TYPE_MAP: Dict[str, str] = {
    "Coal":          "Coal",
    "coal":          "Coal",
    "Oil":           "Oil",
    "oil":           "Oil",
    "Petroleum":     "Oil",
    "petroleum":     "Oil",
    "NatGas":        "Gas",
    "Nat_Gas":       "Gas",
    "Natgas":        "Gas",
    "Nat_gas":       "Gas",
    "natgas":        "Gas",
    "nat_gas":       "Gas",
    "NaturalGas":    "Gas",
    "Natural_Gas":   "Gas",
    "Naturalgas":    "Gas",
    "Natural_gas":   "Gas",
    "naturalgas":    "Gas",
    "natural_gas":   "Gas",
    "Gas":           "Gas",
    "gas":           "Gas",
    "NGCC":          "Gas",
    "ngcc":          "Gas",
    "NGCT":          "Gas",
    "ngct":          "Gas",
    "NGST":          "Gas",
    "ngst":          "Gas",
    "NGGT":          "Gas",
    "nggt":          "Gas",
    "NG":            "Gas",
    "ng":            "Gas",
    "GT":            "Gas",
    "gt":            "Gas",
    "Hydro":         "Hydro",
    "hydro":         "Hydro",
    "Hydropower":    "Hydro",
    "hydropower":    "Hydro",
    "Hydro_power":   "Hydro",
    "hydro_power":   "Hydro",
    "Solar":         "Solar",
    "solar":         "Solar",
    "Wind":          "Wind",
    "wind":          "Wind",
    "Bio":           "Biomass",
    "bio":           "Biomass",
    "Biomass":       "Biomass",
    "biomass":       "Biomass",
    "Geo":           "Geothermal",
    "geo":           "Geothermal",
    "Geothermal":    "Geothermal",
    "geothermal":    "Geothermal",
    "OffshoreWind":  "OffshoreWind",
    "Offshorewind":  "OffshoreWind",
    "Offshore_Wind": "OffshoreWind",
    "Offshore_wind": "OffshoreWind",
    "offshorewind":  "OffshoreWind",
    "offshore_wind": "OffshoreWind",
}


# ===========================================================================
# Data-loading helpers
# ===========================================================================

def _build_gen_sets(df_gen: pd.DataFrame) -> Dict[str, List[str]]:
    """Group generator names by their Pyomo fuel-type set.

    Parameters
    ----------
    df_gen:
        DataFrame loaded from ``generator_parameters_file`` with at least
        columns ``name`` and ``typ``.

    Returns
    -------
    dict
        ``{pyomo_set_name: [gen_name, …], …}`` covering all recognized types.
        Empty lists are included so the model sees every set even when a
        region has no generators of a given type.
    """
    gen_sets: Dict[str, List[str]] = {
        name: [] for name in [
            "Coal", "Oil", "Gas", "Hydro", "Solar", "Wind",
            "Biomass", "Geothermal", "OffshoreWind",
        ]
    }
    for _, row in df_gen.iterrows():
        pyomo_type = FUEL_TYPE_MAP.get(row["typ"])
        if pyomo_type is not None:
            gen_sets[pyomo_type].append(row["name"])
        else:
            raise ValueError(f"Unknown generator type {row['typ']} for generator {row['name']}")
    return gen_sets


def _build_sparse_maps(
    df_gen_matrix: pd.DataFrame,
    df_line_to_bus: pd.DataFrame,
    df_ba_matrix: pd.DataFrame,
    df_storage_matrix: pd.DataFrame,
) -> Tuple[dict, dict, dict, dict]:
    """Convert incidence-matrix DataFrames to sparse dicts for Pyomo Params.

    Parameters
    ----------
    df_gen_matrix:
        Bus-to-unit mapping matrix (rows = generator names, cols = bus names,
        values = 0 / 1).
    df_line_to_bus:
        Line-to-bus incidence matrix (rows = line names, cols = bus names,
        values = -1 / 0 / 1).
    df_ba_matrix:
        Exchange-to-line mapping matrix (rows = exchange names, cols = line names,
        values = -1 / 0 / 1).
    df_storage_matrix:
        Storage-to-bus mapping matrix (rows = storage unit names, cols = bus names,
        values = 0 / 1).

    Returns
    -------
    Tuple of four sparse dicts:
        (bus_to_unit_map, line_to_bus_map, exchange_map, bus_to_storage_map)
    """
    def _to_sparse(df: pd.DataFrame) -> dict:
        """Return {(row_idx, col_name): value} for all non-zero entries."""
        stacked = df.stack()
        nonzero = stacked[stacked != 0]
        return {idx: float(val) for idx, val in nonzero.items()}

    bus_to_unit_map = _to_sparse(df_gen_matrix)       # (gen, bus) → 1
    line_to_bus_map = _to_sparse(df_line_to_bus)      # (line, bus) → ±1
    exchange_map = _to_sparse(df_ba_matrix)        # (exchange, line) → ±1
    bus_to_storage_map = _to_sparse(df_storage_matrix)  # (storage, bus) → 1

    return bus_to_unit_map, line_to_bus_map, exchange_map, bus_to_storage_map


# ===========================================================================
# Main data loader
# ===========================================================================

def load_simulation_data(config: configuration.Config) -> Dict[str, Any]:
    """Load all simulation inputs from CSV/NPY files into Python structures.

    This function is called **once** per simulation run.  The returned ``sim``
    dictionary is passed to :func:`~gridops.west.linear.build_west_linear` to
    build the ConcreteModel and to :func:`_update_horizon_params` before each
    horizon solve.

    Parameters
    ----------
    config:
        :class:`~gridops.configuration.Config` instance with all file paths.

    Returns
    -------
    dict
        ``sim`` data dictionary for the optimization.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading simulation data from input files…")

    # -------------------------------------------------------------------
    # Read all input CSVs
    # -------------------------------------------------------------------

    # Generator parameters 
    df_gen = pd.read_csv(config.generator_parameters_file, header=0)

    # Generator–bus mapping matrix
    df_gen_matrix = pd.read_csv(config.generator_matrix_file, index_col=0)

    # Transmission line–bus incidence
    df_line_to_bus = pd.read_csv(config.line_to_bus_file, index_col=0)

    # Transmission line parameters
    df_line_params = pd.read_csv(config.line_parameters_file, index_col=0)

    # BA-to-BA exchange–line mapping
    df_ba_matrix = pd.read_csv(config.ba_to_ba_transmission_matrix_file, index_col=0)

    # BA-to-BA hurdle rates
    df_ba_hurdle = pd.read_csv(config.ba_to_ba_hurdle_scaled_file, header=0)

    # Hydropower daily limits and totals
    df_hydro_max = pd.read_csv(config.daily_hydro_maximum_file, header=0)
    df_hydro_min = pd.read_csv(config.daily_hydro_minimum_file, header=0)
    df_hydro_total = pd.read_csv(config.daily_hydro_total_file, header=0)

    # Renewable hourly profiles
    df_solar = pd.read_csv(config.nodal_solar_file, header=0)
    df_wind = pd.read_csv(config.nodal_wind_file, header=0)
    df_offshorewind = pd.read_csv(config.nodal_offshore_wind_file, header=0)

    # Hourly load
    df_load = pd.read_csv(config.nodal_load_file, header=0)

    # Must-run capacity (single row, columns = node names; constant)
    df_must_run = pd.read_csv(config.must_run_file, header=0)

    # Daily fuel prices
    df_fuel_prices = pd.read_csv(config.fuel_prices_file, header=0)

    # Storage parameters
    df_storage_params = pd.read_csv(config.storage_params_file, header=0)

    # Storage–bus mapping matrix
    df_storage_matrix = pd.read_csv(config.bus_to_storage_matrix_file, index_col=0)

    # Thermal generator list
    df_thermal = pd.read_csv(config.thermal_generators_file, header=0)

    # Hourly lost-capacity per outage group
    df_losses = pd.read_csv(config.lost_capacity_file, header=0, index_col=0)
    # Ensure Time index is integer for clean .loc[] access
    df_losses.index = df_losses.index.astype(int)

    # Generator outage categories (.npy dict: group_name → [gen_names])
    loss_dict: Dict[str, List[str]] = np.load(
        config.generator_outage_file, allow_pickle=True
    ).item()

    logger.info("All input files loaded.")

    # -------------------------------------------------------------------
    # Build generator sets
    # -------------------------------------------------------------------
    gen_sets = _build_gen_sets(df_gen)

    # -------------------------------------------------------------------
    # Build network entity lists
    # -------------------------------------------------------------------
    # Buses: column names from the load file
    buses: List[str] = list(df_load.columns)

    # Lines: row index from the line-to-bus incidence file
    lines: List[str] = list(df_line_to_bus.index)

    # Exchanges: row index from the BA-to-BA mapping file
    exchanges: List[str] = list(df_ba_matrix.index)

    # Storage units: 'name' column of storage-params file
    storage_units: List[str] = list(df_storage_params["name"])

    # -------------------------------------------------------------------
    # Build sparse mapping dicts
    # -------------------------------------------------------------------
    bus_to_unit_map, line_to_bus_map, exchange_map, bus_to_storage_map = (
        _build_sparse_maps(
            df_gen_matrix,
            df_line_to_bus,
            df_ba_matrix,
            df_storage_matrix,
        )
    )

    # -------------------------------------------------------------------
    # Generator parameter dict
    # -------------------------------------------------------------------
    gen_params: Dict[str, Dict] = {}
    for _, row in df_gen.iterrows():
        gen_params[row["name"]] = {
            "typ":       row["typ"],
            "bus":       row["node"],
            "maxcap":    float(row["maxcap"]),
            "mincap":    float(row["mincap"]),
            "heat_rate": float(row["heat_rate"]),
            "var_om":    float(row["var_om"]),
            "no_load":   float(row["no_load"]),
            "st_cost":   float(row["st_cost"]),
            "ramp":      float(row["ramp"]),
            "minup":     float(row["minup"]),
            "mindn":     float(row["mindn"]),
        }

    # -------------------------------------------------------------------
    # Line parameter dict
    # -------------------------------------------------------------------
    line_params: Dict[str, Dict] = {}
    for line_name, row in df_line_params.iterrows():
        line_params[line_name] = {
            "reactance": float(row["reactance"]),
            "flow_lim":  float(row["limit"]),
        }

    # -------------------------------------------------------------------
    # Exchange hurdle dict
    # -------------------------------------------------------------------
    exchange_hurdle: Dict[str, float] = {}
    for _, row in df_ba_hurdle.iterrows():
        exchange_hurdle[row["BA_to_BA"]] = float(row["Hurdle_$/MWh"])

    # -------------------------------------------------------------------
    # Storage parameter dict
    # -------------------------------------------------------------------
    storage_params: Dict[str, Dict] = {}
    for _, row in df_storage_params.iterrows():
        storage_params[row["name"]] = {
            "s_typ":          row["s_typ"],
            "s_bus":          row["s_node"],
            "charge_rate":    float(row["charge_rate"]),
            "discharge_rate": float(row["discharge_rate"]),
            "duration":       float(row["duration"]),
            "max_SoC":        float(row["max_SoC"]),
            "min_SoC":        float(row["min_SoC"]),
            "charge_eff":     float(row["charge_eff"]),
            "discharge_eff":  float(row["discharge_eff"]),
        }

    # -------------------------------------------------------------------
    # Generator → node column lookup dicts for time-series slicing
    # Each renewable/hydro generator name encodes its node: 'bus_XXXXX_TYPE'
    # -------------------------------------------------------------------
    def _bus_from_gen(gen_name: str) -> str:
        """Extract bus name from a generator name of the form bus_XXX_TYPE."""
        # bus is stored in gen_params
        return gen_params[gen_name]["bus"]

    hydro_bus_map = {g: _bus_from_gen(g) for g in gen_sets["Hydro"]}
    solar_bus_map = {g: _bus_from_gen(g) for g in gen_sets["Solar"]}
    wind_bus_map = {g: _bus_from_gen(g) for g in gen_sets["Wind"]}
    offshorewind_bus_map = {g: _bus_from_gen(g) for g in gen_sets["OffshoreWind"]}

    # -------------------------------------------------------------------
    # Must-run (nuclear/other must-run units): constant per node
    # -------------------------------------------------------------------
    sim_must_run: Dict[str, float] = {}
    if not df_must_run.empty:
        for bus_col in df_must_run.columns:
            sim_must_run[bus_col] = float(df_must_run[bus_col].iloc[0])

    # -------------------------------------------------------------------
    # Nuclear generator list (for must-run capacity loss adjustment)
    # -------------------------------------------------------------------
    nucs: List[str] = list(
        df_thermal[df_thermal["Fuel"] == "NUC (Nuclear)"]["Name"]
    )
    
    # -------------------------------------------------------------------
    # Assemble and return the sim dict
    # -------------------------------------------------------------------
    sim: Dict[str, Any] = {
        # Sets
        "gen_sets":             gen_sets,
        "outage_group_sets":    loss_dict, # {group: [gen_names]} 
        "buses":                buses,
        "lines":                lines,
        "exchanges":            exchanges,
        "storage_units":        storage_units,
        # Static params
        "gen_params":           gen_params,
        "line_params":          line_params,
        "exchange_hurdle":      exchange_hurdle,
        "storage_params":       storage_params,
        # Sparse mapping dicts
        "bus_to_unit_map":      bus_to_unit_map,
        "line_to_bus_map":      line_to_bus_map,
        "exchange_map":         exchange_map,
        "bus_to_storage_map":   bus_to_storage_map,
        # Time-series DataFrames (0-indexed rows)
        "df_demand":            df_load,
        "df_hydro_max":         df_hydro_max,
        "df_hydro_min":         df_hydro_min,
        "df_hydro_total":       df_hydro_total,
        "df_solar":             df_solar,
        "df_wind":              df_wind,
        "df_offshorewind":      df_offshorewind,
        "df_fuel_prices":       df_fuel_prices,
        "df_losses":            df_losses, # Time column is the int index
        # Generator → bus lookup for renewable/hydro time series
        "hydro_bus_map":        hydro_bus_map,
        "solar_bus_map":        solar_bus_map,
        "wind_bus_map":         wind_bus_map,
        "offshorewind_bus_map": offshorewind_bus_map,
        # Must-run and nuclear
        "sim_must_run":         sim_must_run,
        "nucs":                 nucs,
        "loss_dict":            loss_dict,
        # Initial states (populated/updated per horizon in the launch loop)
        "initial_gen":          {}, # {gen: mwh_val} – filled before first horizon
        "initial_soc":          {}, # {storage: soc_val} – filled before first horizon
    }
    return sim


# ===========================================================================
# Horizon-parameter updater
# ===========================================================================

def _update_horizon_params(
    model: pyo.ConcreteModel,
    sim_data: Dict[str, Any],
    horizon_idx: int,
    horizon_hours: int,
) -> None:
    """Slices simulation arrays for horizon *horizon_idx* and assign values
    to all mutable Pyomo parameters in *model*.

    Also, applies historical capacity-loss adjustments to ``HorizonGenLimit``
    and nuclear outage adjustments to ``HorizonMustrunLimit``.

    Parameters
    ----------
    model:
        The active ConcreteModel whose mutable Params are updated in-place.
    sim_data:
        Dict returned by :func:`load_simulation_data`.
    horizon_idx:
        Zero-based horizon counter (0 = first horizon, 1 = second horizon, …).
    horizon_hours:
        Number of hours in the horizon (by default 24).
    """
    # Absolute hour indices for this horizon window (0-indexed for DataFrames)
    h_start_0idx = horizon_idx * horizon_hours
    # 1-indexed hours used to access the losses DataFrame (Time index starts at 1)
    h_start_1idx = h_start_0idx + 1
    # 0-indexed day of the first hour in this horizon (for daily-resolution data)
    day_0idx = h_start_0idx // 24

    # Number of full calendar days covered by this horizon (for hydro totals)
    n_horizon_days = max(1, horizon_hours // 24)

    # Pre-slice the large time-series DataFrames for this horizon window.
    h_slice = slice(h_start_0idx, h_start_0idx + horizon_hours)
    demand_slice = sim_data["df_demand"].iloc[h_slice]
    solar_slice = sim_data["df_solar"].iloc[h_slice]
    wind_slice = sim_data["df_wind"].iloc[h_slice]
    offshorewind_slice = sim_data["df_offshorewind"].iloc[h_slice]
    df_losses = sim_data["df_losses"]

    # -------------------------------------------------------------------
    # Demand (hourly resolution, per bus)
    # -------------------------------------------------------------------
    for t in model.time_periods:
        row = demand_slice.iloc[t - 1]
        for b in model.buses:
            raw = float(row.get(b, 0.0))
            model.HorizonDemand[b, t] = raw if raw >= SNAP_TOL else 0.0

    # -------------------------------------------------------------------
    # Fuel prices (daily resolution → per thermal generator)
    # -------------------------------------------------------------------
    fp_row = sim_data["df_fuel_prices"].iloc[day_0idx]
    for g in model.Thermal:
        _fp = fp_row.get(g, None)
        if _fp is None:
            raise ValueError(f"Fuel price for generator {g} not found in fuel_prices_file.")
        else:
            _fp = float(_fp)
        model.FuelPrice[g] = _fp if abs(_fp) >= SNAP_TOL else 0.0

    # -------------------------------------------------------------------
    # Hydro limits
    # For each hydro generator the relevant time-series column is the bus
    # that the generator is located at (hydro_bus_map).
    # -------------------------------------------------------------------
    hydro_bus_map = sim_data["hydro_bus_map"]
    df_hydro_max = sim_data["df_hydro_max"]
    df_hydro_min = sim_data["df_hydro_min"]
    df_hydro_total = sim_data["df_hydro_total"]

    for g in model.Hydro:
        bus = hydro_bus_map[g]
        _hmax = float(df_hydro_max.iloc[day_0idx][bus])
        _hmin = float(df_hydro_min.iloc[day_0idx][bus])
        model.HorizonHydro_MAX[g] = _hmax if _hmax >= SNAP_TOL else 0.0
        model.HorizonHydro_MIN[g] = _hmin if _hmin >= SNAP_TOL else 0.0
        # Sum daily energy budgets over all days in this horizon
        _htotal = sum(
            float(df_hydro_total.iloc[day_0idx + d][bus])
            for d in range(n_horizon_days)
        )
        model.HorizonHydro_TOTAL[g] = _htotal if _htotal >= SNAP_TOL else 0.0

    # -------------------------------------------------------------------
    # Solar
    # -------------------------------------------------------------------
    solar_bus_map = sim_data["solar_bus_map"]
    for g in model.Solar:
        bus = solar_bus_map.get(g)
        if bus:
            for t in model.time_periods:
                raw = float(solar_slice.iloc[t - 1].get(bus, 0.0))
                model.HorizonSolar[g, t] = raw if raw >= SNAP_TOL else 0.0

    # -------------------------------------------------------------------
    # Wind
    # -------------------------------------------------------------------
    wind_bus_map = sim_data["wind_bus_map"]
    for g in model.Wind:
        bus = wind_bus_map.get(g)
        if bus:
            for t in model.time_periods:
                raw = float(wind_slice.iloc[t - 1].get(bus, 0.0))
                model.HorizonWind[g, t] = raw if raw >= SNAP_TOL else 0.0

    # -------------------------------------------------------------------
    # Offshore wind
    # -------------------------------------------------------------------
    offshorewind_bus_map = sim_data["offshorewind_bus_map"]
    for g in model.OffshoreWind:
        bus = offshorewind_bus_map.get(g)
        if bus:
            for t in model.time_periods:
                raw = float(offshorewind_slice.iloc[t - 1].get(bus, 0.0))
                model.HorizonOffshoreWind[g, t] = raw if raw >= SNAP_TOL else 0.0

    # -------------------------------------------------------------------
    # HorizonGenLimit: base = maxcap; losses subtracted afterwards
    # -------------------------------------------------------------------
    gp = sim_data["gen_params"]
    for g in model.Outage:
        base_cap = gp[g]["maxcap"]
        for t in model.time_periods:
            model.HorizonGenLimit[g, t] = base_cap

    # -------------------------------------------------------------------
    # Must-run limit (constant per bus, then nuclear loss subtracted afterwards)
    # -------------------------------------------------------------------
    sim_must_run = sim_data["sim_must_run"]
    for b in model.buses:
        base_mr = sim_must_run.get(b, 0.0)
        for t in model.time_periods:
            model.HorizonMustrunLimit[b, t] = base_mr

    # -------------------------------------------------------------------
    # Capacity-loss adjustments for each outage group
    # loss_dict maps: group_name → list of generator names in that group
    # The entire group's loss for a given hour is spread equally across
    # all generators in the group.
    # -------------------------------------------------------------------
    loss_dict = sim_data["loss_dict"]
    for group_name, gen_list in loss_dict.items():
        n_in_group = len(gen_list)
        if n_in_group == 0:
            continue
        # Retrieve the Pyomo set for this outage group
        group_set = getattr(model, group_name, None)
        if group_set is None:
            continue
        for g in group_set:
            for t in model.time_periods:
                abs_h = h_start_1idx + t - 1 # 1-indexed absolute hour
                loss = float(df_losses.loc[abs_h, group_name]) / n_in_group
                current = pyo.value(model.HorizonGenLimit[g, t])
                raw = current - loss
                model.HorizonGenLimit[g, t] = raw if raw >= SNAP_TOL else 0.0

    # -------------------------------------------------------------------
    # Nuclear capacity-loss adjustment applied to all nodes' must-run limit.
    # Only Nuclear_ovr_1000 column is used (matching original model behaviour).
    # -------------------------------------------------------------------
    nucs = sim_data["nucs"]
    n_nucs = len(nucs)
    if n_nucs > 0:
        for b in model.buses:
            for t in model.time_periods:
                abs_h = h_start_1idx + t - 1 # 1-indexed absolute hour
                nuc_loss = float(df_losses.loc[abs_h, "Nuclear_ovr_1000"]) / n_nucs
                current = pyo.value(model.HorizonMustrunLimit[b, t])
                raw = current - nuc_loss
                model.HorizonMustrunLimit[b, t] = raw if raw >= SNAP_TOL else 0.0


# ===========================================================================
# Main launch function
# ===========================================================================

def west_linear(
    config_file: Union[str, None],
    solver_name: str = "appsi_highs",
    solver_params: Union[None, dict] = None,
    warmstart: bool = False,
    n_days: int = 365,
    horizon_hours: int = 24,
    restart_day: Union[None, int] = None,
    save_restart_file: bool = True,
    break_run: bool = False,
    restart_write_frequency: int = 30,
    fresh_start: bool = False,
    clear_restart: bool = True,
    **kwargs,
) -> int:
    """Run the U.S. Western Interconnection economic dispatch model.

    Parameters
    ----------
    config_file:
        Path to the YAML model configuration file.
    solver_name:
        Solver interface to be used. Default ``"appsi_highs"``.
    solver_params:
        Dict of solver options (option_name → value). ``None`` = solver
        built-in defaults.
    n_days:
        Total number of simulation days to process. The function stops after
        ``n_days * 24 / horizon_hours`` horizons. Default 365.
    horizon_hours:
        Length of each optimization horizon in hours. Default 24.
    restart_day:
        Specific day whose restart file should be used as the starting point.
        ``None`` = use the latest available restart file (or start fresh if
        none exist).
    save_restart_file:
        If ``True``, write a restart file periodically (controlled by
        *restart_write_frequency*). Default ``True``.
    break_run:
        If ``True``, run exactly one horizon and return. Used internally by
        the solver-retry mechanism. Default ``False``.
    restart_write_frequency:
        Write a restart file every this many solved horizons. Default 30.
    fresh_start:
        If ``True``, delete all existing restart files before starting the
        simulation so it always begins from scratch. Default ``False``.
    clear_restart:
        If ``True``, delete all restart files after all *n_days* have been
        solved successfully. Files are kept if the simulation fails before
        completion. Default ``True``.
    **kwargs:
        Extra keyword arguments forwarded to
        :func:`~gridops.configuration.generate_config`.

    Returns
    -------
    int
        The 1-based index of the last successfully solved **day** (for
        compatibility with the ``Model`` class retry logic).

    Raises
    ------
    RuntimeError
        If the solver fails to find an optimal solution for any horizon.
    AssertionError
        If *n_days* is less than the implied start day from the restart file.
    """
    logger = logging.getLogger(__name__)
    logger.info("Preparing the U.S. Western Interconnection economic dispatch simulation.")

    # -------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------
    config = configuration.generate_config(config_file=config_file, **kwargs)

    # Ensure the restart directory exists
    os.makedirs(config.restart_file_directory, exist_ok=True)

    # -------------------------------------------------------------------
    # Fresh start: wipe any previous restart files
    # -------------------------------------------------------------------
    if fresh_start:
        clear_restart_files(config.restart_file_directory)
        logger.info("fresh_start=True - All prior restart files deleted.")

    # -------------------------------------------------------------------
    # Solver
    # -------------------------------------------------------------------
    opt = GoSolver(
        solver_name=solver_name,
        solver_params=solver_params,
    ).go_solver

    # -------------------------------------------------------------------
    # Total number of horizons to solve
    # -------------------------------------------------------------------
    n_total_hours = n_days * 24
    n_horizons = n_total_hours // horizon_hours

    # -------------------------------------------------------------------
    # Restart or fresh start
    # -------------------------------------------------------------------
    restart_file = get_restart_file(
        dir=config.restart_file_directory,
        day=restart_day,
    )

    # Accumulator lists for the optimization results
    mwh: List = []
    flow: List = []
    slack: List = []
    vlt_angle: List = []
    duals: List = []
    charge: List = []
    discharge: List = []
    soc: List = []
    # Per-horizon solver parameters
    solver_parameters: Dict[int, Any] = {}

    if restart_file is None:
        # -------------------------------------------------------------------
        # Fresh start: load data and build the ConcreteModel
        # -------------------------------------------------------------------
        logger.info("No restart file found - Starting from the first horizon.")
        start_horizon = 0

        sim_data = load_simulation_data(config)

        # Initialise boundary conditions for hour 0
        # Generation: 0 MW for all Dispatchable generators
        # SoC: min_SoC for all Storage units (first-horizon initial state)
        # These are stored in sim_data so they persist across horizons.
        sim_data["initial_gen"] = {}    # all zeros → will be set by model fix
        sim_data["initial_soc"] = {
            s: sim_data["storage_params"][s]["min_SoC"]
            for s in sim_data["storage_units"]
        }

        model = build_west_linear(sim_data, horizon_hours=horizon_hours)
        restart_data = None # No prior restart to carry forward

    else:
        # -------------------------------------------------------------------
        # Warm restart: Restore model and result accumulators from pickle
        # -------------------------------------------------------------------
        logger.info(f"Loading restart file: {restart_file}")
        with open(restart_file, "rb") as fh:
            restart_data = cloudpickle.load(fh)

        model = restart_data["model"]
        mwh = restart_data["mwh"]
        flow = restart_data["flow"]
        slack = restart_data["slack"]
        vlt_angle = restart_data["vlt_angle"]
        duals = restart_data["duals"]
        charge = restart_data["charge"]
        discharge = restart_data["discharge"]
        soc = restart_data["soc"]
        sim_data = restart_data["sim_data"]
        solver_parameters = restart_data["solver_parameters"]

        # The restart records the last completed day; start one horizon after it.
        last_day = restart_data["day"]
        start_horizon = (last_day * 24) // horizon_hours
        logger.info(f"Resuming from horizon {start_horizon} (after day {last_day}).")

    # Validate that we still have horizons left to solve.
    start_day = (start_horizon * horizon_hours) // 24 + 1
    if n_days < start_day:
        raise AssertionError(
            f"n_days ({n_days}) must be ≥ the start day ({start_day}). "
            "n_days represents the last calendar day to process."
        )

    # Pre-build a generator-type lookup to avoid repeated set membership tests.
    gen_type_lookup: Dict[str, str] = {}
    for fuel_type, gen_list in sim_data["gen_sets"].items():
        for g in gen_list:
            gen_type_lookup[g] = fuel_type

    # -------------------------------------------------------------------
    # Horizon loop
    # -------------------------------------------------------------------
    last_solved_horizon = start_horizon - 1  # track last successfully completed horizon

    for horizon_idx in range(start_horizon, n_horizons):
        # Map horizon index back to a 1-based day number for logging/restart
        current_day = (horizon_idx * horizon_hours) // 24 + 1

        logger.info(
            f"Horizon {horizon_idx + 1} / {n_horizons}  (day {current_day}, hours {horizon_idx * horizon_hours + 1}-{(horizon_idx + 1) * horizon_hours})"
        )

        # Record solver parameters for this horizon
        solver_parameters[horizon_idx] = solver_params

        # ---------------------------------------------------------------
        # Clear HiGHS solver state before horizon 2+ when warm start is
        # disabled.  This avoids expensive internal basis reconciliation
        # after parameter changes and forces a fresh presolve+solve.
        # ---------------------------------------------------------------
        if (
            horizon_idx > start_horizon
            and solver_name == "appsi_highs"
            and not warmstart
        ):
            try:
                opt._solver_model.clearSolver()
                logger.info(
                    f"Horizon {horizon_idx}: cleared HiGHS solver state (warmstart=False)."
                )
            except Exception as e:
                logger.warning(
                    f"Horizon {horizon_idx}: clearSolver failed: {e}"
                )

        # ---------------------------------------------------------------
        # Update mutable Pyomo parameters for this horizon window
        # ---------------------------------------------------------------
        logger.info(f"Horizon {horizon_idx}: updating parameters.")
        _update_horizon_params(model, sim_data, horizon_idx, horizon_hours)

        # ---------------------------------------------------------------
        # Ramp constraint at t=1: skip for the very first horizon (no prior
        # dispatch history); enforce from horizon 1 onward using the
        # InitialMwh carry-forward values.
        # ---------------------------------------------------------------
        if horizon_idx == 0:
            for g in model.Thermal:
                model.RampCon1[g, 1].deactivate()
                model.RampCon2[g, 1].deactivate()
        elif horizon_idx == 1:
            for g in model.Thermal:
                model.RampCon1[g, 1].activate()
                model.RampCon2[g, 1].activate()

        # ---------------------------------------------------------------
        # Solve
        # ---------------------------------------------------------------
        logger.info(f"Horizon {horizon_idx}: starting optimization.")

        try:
            result = opt.solve(
                model,
                tee=True,
                symbolic_solver_labels=False,
            )
        except Exception as solve_err:
            # APPSI raises an error when load_solutions=True and the
            # solver cannot find a feasible solution. Save a restart for the
            # previous horizon before re-raising so retry can pick up.
            logger.error(
                f"Horizon {horizon_idx}: Solver raised an error - {solve_err}"
            )
            if save_restart_file and restart_data is not None:
                write_restart_file(
                    dir=config.restart_file_directory,
                    day=current_day - 1,
                    restart_data=restart_data,
                )
            raise RuntimeError(
                f"Optimization failed at horizon {horizon_idx} "
                f"(day {current_day}): {solve_err}"
            ) from solve_err

        # Check optimality
        term_cond = result.solver.termination_condition
        status    = result.solver.status

        if (term_cond != pyo.TerminationCondition.optimal) or (
            status != pyo.SolverStatus.ok
        ):
            logger.error(
                f"Horizon {horizon_idx} failed - Termination: {term_cond}, status: {status}."
            )
            # Write restart for the *previous* horizon before raising
            if save_restart_file and restart_data is not None:
                write_restart_file(
                    dir=config.restart_file_directory,
                    day=current_day - 1,
                    restart_data=restart_data,
                )
            raise RuntimeError(
                f"Optimization failed at horizon {horizon_idx} "
                f"(day {current_day}) with termination '{term_cond}'."
            )

        logger.info(f"Horizon {horizon_idx}: Optimization complete.")

        instance = model

        # ---------------------------------------------------------------
        # Extract results (with integrated negative-generation check)
        # ---------------------------------------------------------------
        abs_hour_offset = horizon_idx * horizon_hours

        # Generator dispatch (mwh) — includes negative-generation sanity check
        has_negative_gen = False
        for g in instance.Generators:
            fuel_type = gen_type_lookup.get(g, "Unknown")
            for t in instance.time_periods:
                abs_h = abs_hour_offset + t
                val = pyo.value(instance.mwh[g, t])
                mwh.append((g, fuel_type, abs_h, val))
                if val is not None and val < -1e-3:
                    has_negative_gen = True
                    logger.error(
                        f"Negative generation: mwh[{g}, {t}] = {val:.6f} at horizon {horizon_idx}."
                    )

        if has_negative_gen:
            if save_restart_file and restart_data is not None:
                write_restart_file(
                    dir=config.restart_file_directory,
                    day=current_day - 1,
                    restart_data=restart_data,
                )
            raise RuntimeError(
                f"Negative generation detected at horizon {horizon_idx} "
                f"(day {current_day})."
            )

        # Node-level results: duals (LMPs), voltage angles, and slack
        for b in instance.buses:
            for t in instance.time_periods:
                abs_h = abs_hour_offset + t
                duals.append((b, abs_h, instance.dual[instance.Bus_Constraint[b, t]]))
                vlt_angle.append((b, abs_h, pyo.value(instance.Theta[b, t])))
                slack.append((b, abs_h, pyo.value(instance.S[b, t])))

        # Transmission line flows
        for l in instance.lines:
            for t in instance.time_periods:
                abs_h = abs_hour_offset + t
                flow.append((l, abs_h, pyo.value(instance.Flow[l, t])))

        # Storage: SoC, charge, and discharge in a single loop
        for s in instance.Storage:
            for t in instance.time_periods:
                abs_h = abs_hour_offset + t
                soc.append((s, abs_h, pyo.value(instance.SoC[s, t])))
                charge.append((s, abs_h, pyo.value(instance.Charge[s, t])))
                discharge.append((s, abs_h, pyo.value(instance.Discharge[s, t])))

        # ---------------------------------------------------------------
        # Pass state variables to the next horizon (mutable param update)
        # ---------------------------------------------------------------

        # Thermal generation at the last hour becomes the initial
        # generation (InitialMwh) for the following horizon's ramp constraints.
        for g in instance.Thermal:
            raw_val = pyo.value(instance.mwh[g, horizon_hours])
            if raw_val is None or abs(raw_val) < SNAP_TOL:
                val = 0.0
            else:
                val = max(0.0, raw_val)
            instance.InitialMwh[g] = val

        # Storage SoC at the last hour becomes InitialSoC for the next horizon.
        for s in instance.Storage:
            raw_val = pyo.value(instance.SoC[s, horizon_hours])
            min_s = pyo.value(instance.min_SoC[s])
            max_s = pyo.value(instance.max_SoC[s])
            if raw_val is None or raw_val < min_s - SNAP_TOL:
                val = min_s
            elif raw_val > max_s + SNAP_TOL:
                val = max_s
            else:
                val = raw_val
            instance.InitialSoC[s] = val

        last_solved_horizon = horizon_idx

        # ---------------------------------------------------------------
        # Periodic restart file
        # ---------------------------------------------------------------
        if save_restart_file:
            restart_data = {
                "model":             model,
                "mwh":               mwh,
                "flow":              flow,
                "slack":             slack,
                "vlt_angle":         vlt_angle,
                "duals":             duals,
                "charge":            charge,
                "discharge":         discharge,
                "soc":               soc,
                "sim_data":          sim_data,
                "solver_parameters": solver_parameters,
                "day":               current_day,
            }
            if (horizon_idx + 1) % restart_write_frequency == 0:
                write_restart_file(
                    dir=config.restart_file_directory,
                    day=current_day,
                    restart_data=restart_data,
                )

        logger.info(f"Horizon {horizon_idx} (day {current_day}) completed.")

        # Break after one horizon if requested (used by retry mode)
        if break_run:
            break

    # -------------------------------------------------------------------
    # Save outputs as Parquet files
    # -------------------------------------------------------------------
    logger.info("Writing output Parquet files …")

    pd.DataFrame(vlt_angle, columns=("Node", "Time", "Value")).to_parquet(config.vlt_angle_file, index=False)
    pd.DataFrame(mwh, columns=("Generator", "Type", "Time", "Value")).to_parquet(config.mwh_file, index=False)
    pd.DataFrame(slack, columns=("Node", "Time", "Value")).to_parquet(config.slack_file, index=False)
    pd.DataFrame(flow, columns=("Line", "Time", "Value")).to_parquet(config.flow_file, index=False)
    pd.DataFrame(duals, columns=("Node", "Time", "Value")).to_parquet(config.duals_file, index=False)
    pd.DataFrame(soc, columns=("Storage", "Time", "Value")).to_parquet(config.storage_soc_file, index=False)
    pd.DataFrame(discharge, columns=("Storage", "Time", "Value")).to_parquet(config.storage_discharge_file, index=False)
    pd.DataFrame(charge, columns=("Storage", "Time", "Value")).to_parquet(config.storage_charge_file, index=False)

    # Write per-horizon solver parameters as JSON for auditing
    write_solver_parameters(
        solver_parameter_dictionary=solver_parameters,
        solver_parameter_file=os.path.join(
            config.restart_file_directory, "solver_parameters.json"
        ),
    )

    logger.info("All outputs written.")

    # -------------------------------------------------------------------
    # Clean up restart files after successful completion of all days
    # -------------------------------------------------------------------
    if clear_restart and not break_run:
        clear_restart_files(config.restart_file_directory)

    # Return the last solved day (1-based) for compatibility with Model.run()
    last_day = (last_solved_horizon * horizon_hours) // 24 + 1
    return last_day
