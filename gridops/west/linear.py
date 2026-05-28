from __future__ import annotations
from typing import Any, Dict, List
import pyomo.environ as pyo


# ===========================================================================
# Defining constants
# ===========================================================================

# Penalty for unserved energy/load shedding ($/MWh).
SLACK_PENALTY: float = 2000.0

# Negligible marginal cost for hydro, wind, solar, and offshore-wind dispatch ($/MWh).
RENEWABLE_GEN_COST: float = 0.01

# Penalty on absolute flow proxy (DummyFlow) to discourage slack routing across the transmission network ($/MWh).
DUMMY_FLOW_PENALTY: float = 0.01

# Negligible cost for storage charge/discharge ($/MWh) to prevent simultaneous charge-discharge at optimality.
STORAGE_COST: float = 0.001

# Reference bus whose voltage angle is fixed to zero in the DC-OPF.
REFERENCE_BUS: str = "bus_100011"

# Per-unit to MW conversion factor for DC power-flow equations (reactance is in p.u. on a 100 MVA base).
PU_BASE_MVA: float = 100.0


# ===========================================================================
# Objective rule
# ===========================================================================

def _sys_cost_rule(model: pyo.ConcreteModel) -> pyo.Expression:
    """Returns the total system operating cost expression.

    Cost components:
    1. **Thermal generation** = (heat-rate × fuel-price + variable O&M) × output.
    2. **Slack/unserved demand** = penalised at :data:`SLACK_PENALTY` $/MWh to
       force the solver to prefer physical generation over load shedding.
    3. **Hydro, wind, solar, offshore-wind** = small :data:`RENEWABLE_GEN_COST`
       $/MWh cost to break ties and produce a unique dispatch.
    4. **BA-to-BA exchange hurdles** = transmission charges on cross-BA flows.
    5. **Dummy flow** = :data:`DUMMY_FLOW_PENALTY` $/MWh on the absolute-value
       proxy variable to discourage routing slack generation.
    6. **Storage charge/discharge** = :data:`STORAGE_COST` $/MWh to prevent 
    simultaneous charge-discharge.

    Parameters
    ----------
    model:
        The active ConcreteModel instance.

    Returns
    -------
    pyomo.Expression
        System cost expression to be minimised.
    """
    # Thermal generation cost (fuel + variable O&M)
    gen_cost = pyo.quicksum(
        model.mwh[g, t] * (model.heat_rate[g] * model.FuelPrice[g] + model.var_om[g])
        for t in model.time_periods
        for g in model.Thermal
    )
    # Unserved energy penalty
    slack_cost = pyo.quicksum(
        model.S[b, t] * SLACK_PENALTY
        for t in model.time_periods
        for b in model.buses
    )
    # Small costs for renewable/hydro dispatch
    hydro_cost = pyo.quicksum(
        model.mwh[g, t] * RENEWABLE_GEN_COST
        for t in model.time_periods
        for g in model.Hydro
    )
    wind_cost = pyo.quicksum(
        model.mwh[g, t] * RENEWABLE_GEN_COST
        for t in model.time_periods
        for g in model.Wind
    )
    offshore_wind_cost = pyo.quicksum(
        model.mwh[g, t] * RENEWABLE_GEN_COST
        for t in model.time_periods
        for g in model.OffshoreWind
    )
    solar_cost = pyo.quicksum(
        model.mwh[g, t] * RENEWABLE_GEN_COST
        for t in model.time_periods
        for g in model.Solar
    )
    # BA-to-BA exchange hurdle costs
    exchange_cost = pyo.quicksum(
        model.Flow[l, t] * coeff * model.ExchangeHurdle[k]
        for (k, l), coeff in model._exchange_line_pairs
        for t in model.time_periods
    )
    # Absolute-flow proxy penalty (discourages slack routing)
    dummy_flow_cost = pyo.quicksum(
        model.DummyFlow[l, t] * DUMMY_FLOW_PENALTY
        for l in model.lines
        for t in model.time_periods
    )
    # Small costs for storage cycle 
    charge_cost = pyo.quicksum(
        model.Charge[s, t] * STORAGE_COST
        for t in model.time_periods
        for s in model.Storage
    )
    discharge_cost = pyo.quicksum(
        model.Discharge[s, t] * STORAGE_COST
        for t in model.time_periods
        for s in model.Storage
    )
    return (
        gen_cost + slack_cost + hydro_cost + wind_cost + solar_cost
        + exchange_cost + offshore_wind_cost + dummy_flow_cost
        + charge_cost + discharge_cost
    )


# ===========================================================================
# Constraint rules
# ===========================================================================

# Ramp constraints

def _ramp_up_rule(
    model: pyo.ConcreteModel,
    g: Any,
    t: int,
) -> pyo.Expression:
    """Ramp-up constraint: generation increase from t-1 to t <= ramp rate.

    Applied to thermal generators for all hours t = 1 ... horizon_hours.
    At t=1, the previous dispatch is taken from the mutable parameter
    InitialMwh[g] (updated between horizons by the launch loop).

    Parameters
    ----------
    model: Active ConcreteModel.
    g:     Thermal generator index.
    t:     Hour index.
    """
    prev_mwh = model.InitialMwh[g] if t == 1 else model.mwh[g, t - 1]
    return model.mwh[g, t] - prev_mwh <= model.ramp[g]


def _ramp_down_rule(
    model: pyo.ConcreteModel,
    g: Any,
    t: int,
) -> pyo.Expression:
    """Ramp-down constraint: generation decrease from t-1 to t <= ramp rate.

    At t=1, the previous dispatch is taken from the mutable parameter
    InitialMwh[g] (updated between horizons by the launch loop).

    Parameters
    ----------
    model: Active ConcreteModel.
    g:     Thermal generator index.
    t:     Hour index.
    """
    prev_mwh = model.InitialMwh[g] if t == 1 else model.mwh[g, t - 1]
    return prev_mwh - model.mwh[g, t] <= model.ramp[g]


# Capacity constraints

def _max_cap_outage_rule(
    model: pyo.ConcreteModel,
    g: Any,
    t: int,
) -> pyo.Expression:
    """Capacity ceiling for dispatchable generators subject to outages.

    Uses the mutable ``HorizonGenLimit`` which accounts for the base capacity
    reduced by any historical/scheduled capacity losses.

    Parameters
    ----------
    model: Active ConcreteModel.
    g:     Generator in the Outage set.
    t:     Hour index.
    """
    return model.mwh[g, t] <= model.HorizonGenLimit[g, t]


def _max_cap_dispatchable_rule(
    model: pyo.ConcreteModel,
    g: Any,
    t: int,
) -> pyo.Expression:
    """Capacity ceiling for other dispatchable generators.

    Uses the static ``maxcap`` parameter (no outage adjustment for these types).

    Parameters
    ----------
    model: Active ConcreteModel.
    g:     Generator in the Dispatchable set but *not* in the Outage set.
    t:     Hour index.
    """
    return model.mwh[g, t] <= model.maxcap[g]


# Hydro dispatch constraints

def _hydro_total_rule(
    model: pyo.ConcreteModel,
    g: Any,
) -> pyo.Expression:
    """Hydro energy budget: total horizon generation <= energy budget.

    One constraint per hydro unit (not per hour) since the summation
    over all hours is identical regardless of hour index.

    Parameters
    ----------
    model: Active ConcreteModel.
    g:     Hydro generator.
    """
    total_gen = pyo.quicksum(model.mwh[g, t] for t in model.time_periods)
    return total_gen <= model.HorizonHydro_TOTAL[g]


def _hydro_max_rule(
    model: pyo.ConcreteModel,
    g: Any,
    t: int,
) -> pyo.Expression:
    """Hourly generation ceiling for hydro generators.

    Parameters
    ----------
    model: Active ConcreteModel.
    g:     Hydro generator.
    t:     Hour index.
    """
    return model.mwh[g, t] <= model.HorizonHydro_MAX[g]


def _hydro_min_rule(
    model: pyo.ConcreteModel,
    g: Any,
    t: int,
) -> pyo.Expression:
    """Hourly generation floor for hydro generators (minimum must-flow requirement).

    Parameters
    ----------
    model: Active ConcreteModel.
    g:     Hydro generator.
    t:     Hour index.
    """
    return model.mwh[g, t] >= model.HorizonHydro_MIN[g]


# Renewable availability constraints

def _solar_cap_rule(
    model: pyo.ConcreteModel,
    g: Any,
    t: int,
) -> pyo.Expression:
    """Solar capacity ceiling (hourly resource profile).

    Parameters
    ----------
    model: Active ConcreteModel.
    g:     Solar generator.
    t:     Hour index.
    """
    return model.mwh[g, t] <= model.HorizonSolar[g, t]


def _wind_cap_rule(
    model: pyo.ConcreteModel,
    g: Any,
    t: int,
) -> pyo.Expression:
    """Wind capacity ceiling (hourly resource profile).

    Parameters
    ----------
    model: Active ConcreteModel.
    g:     Wind generator.
    t:     Hour index.
    """
    return model.mwh[g, t] <= model.HorizonWind[g, t]


def _offshore_wind_cap_rule(
    model: pyo.ConcreteModel,
    g: Any,
    t: int,
) -> pyo.Expression:
    """Offshore-wind capacity ceiling (hourly resource profile).

    Parameters
    ----------
    model: Active ConcreteModel.
    g:     Offshore-wind generator.
    t:     Hour index.
    """
    return model.mwh[g, t] <= model.HorizonOffshoreWind[g, t]


# Bus energy balance constraint

def _bus_balance_rule(
    model: pyo.ConcreteModel,
    b: Any,
    t: int,
) -> pyo.Expression:
    """Bus energy balance constraint (Kirchhoff's current law).

    At each bus ``b`` and hour ``t``::

        generation + slack + must_run - power_flow
        = load + storage_net_charge

    where power_flow = pyo.quicksum(Flow[l,t] * LineToBusMap[l,b]) over all lines.

    Parameters
    ----------
    model: Active ConcreteModel.
    b:     Bus index.
    t:     Hour index.
    """
    # Power flow to this bus via DC power flow
    # Uses pre-computed sparse lookup: only lines connected to this bus.
    power_flow = pyo.quicksum(
        model.Flow[l, t] * coeff
        for l, coeff in model._lines_at_bus[b]
    )
    # Generator contribution at this bus (sparse: only generators at this bus)
    gen_at_bus = pyo.quicksum(
        model.mwh[g, t]
        for g in model._gens_at_bus[b]
    )
    # Slack contribution at this bus (unserved load)
    slack_at_bus = model.S[b, t]
    # Must-run contribution at this bus (e.g., nuclear)
    mustrun_at_bus = model.HorizonMustrunLimit[b, t]
    # Demand at this bus
    demand_at_bus = model.HorizonDemand[b, t]
    # Storage net flow: charge draws from the bus, discharge feeds the bus
    storage_charge_at_bus = pyo.quicksum(
        model.Charge[s, t] for s in model._storage_at_bus[b]
    )
    storage_discharge_at_bus = pyo.quicksum(
        model.Discharge[s, t] for s in model._storage_at_bus[b]
    )

    return (
        gen_at_bus
        + slack_at_bus
        + mustrun_at_bus
        - power_flow
        == demand_at_bus
        + storage_charge_at_bus
        - storage_discharge_at_bus
    )


# DC power-flow and line capacity constraints

def _dc_flow_rule(
    model: pyo.ConcreteModel,
    l: Any,
    t: int,
) -> pyo.Expression:
    """DC power-flow equation: Flow[l,t] * Reactance[l] = PU_BASE_MVA * DeltaTheta.

    The factor :data:`PU_BASE_MVA` normalises the per-unit reactance (which is
    expressed in p.u. on a 100 MVA base) so that flows are in MW.

    Parameters
    ----------
    model: Active ConcreteModel.
    l:     Transmission line index.
    t:     Hour index.
    """
    delta_theta = pyo.quicksum(
        model.Theta[b, t] * coeff
        for b, coeff in model._buses_on_line[l]
    )
    return model.Flow[l, t] * model.Reactance[l] == PU_BASE_MVA * delta_theta


def _reference_bus_angle_rule(
    model: pyo.ConcreteModel,
    t: int,
) -> pyo.Expression:
    """Set the reference bus angle to zero.

    The reference bus is :data:`REFERENCE_BUS`

    Parameters
    ----------
    model: Active ConcreteModel.
    t:     Hour index.
    """
    return model.Theta[REFERENCE_BUS, t] == 0


# Absolute-value proxy (dummy flow) constraints

def _dummy_flow_pos_rule(
    model: pyo.ConcreteModel,
    l: Any,
    t: int,
) -> pyo.Expression:
    """DummyFlow >= Flow (positive side of absolute value linearization).

    Parameters
    ----------
    model: Active ConcreteModel.
    l:     Transmission line index.
    t:     Hour index.
    """
    return model.DummyFlow[l, t] >= model.Flow[l, t]


def _dummy_flow_neg_rule(
    model: pyo.ConcreteModel,
    l: Any,
    t: int,
) -> pyo.Expression:
    """DummyFlow >= -Flow (negative side of absolute value linearization).

    Parameters
    ----------
    model: Active ConcreteModel.
    l:     Transmission line index.
    t:     Hour index.
    """
    return model.DummyFlow[l, t] >= -model.Flow[l, t]


# Storage constraints

def _max_charge_soc_rule(
    model: pyo.ConcreteModel,
    s: Any,
    t: int,
) -> pyo.Expression:
    """Charge cannot exceed the remaining headroom in the state of charge.

    At t=1, SoC[s,0] is replaced by the mutable parameter InitialSoC[s].

    Parameters
    ----------
    model: Active ConcreteModel.
    s:     Storage unit index.
    t:     Hour index.
    """
    prev_soc = model.InitialSoC[s] if t == 1 else model.SoC[s, t - 1]
    return (
        model.Charge[s, t]
        <= (model.max_SoC[s] - prev_soc) / model.charge_eff[s]
    )


def _max_discharge_soc_rule(
    model: pyo.ConcreteModel,
    s: Any,
    t: int,
) -> pyo.Expression:
    """Discharge cannot exceed available energy above the minimum SoC.

    At t=1, SoC[s,0] is replaced by the mutable parameter InitialSoC[s].

    Parameters
    ----------
    model: Active ConcreteModel.
    s:     Storage unit index.
    t:     Hour index.
    """
    prev_soc = model.InitialSoC[s] if t == 1 else model.SoC[s, t - 1]
    return (
        model.Discharge[s, t]
        <= (prev_soc - model.min_SoC[s]) * model.discharge_eff[s]
    )


def _soc_balance_rule(
    model: pyo.ConcreteModel,
    s: Any,
    t: int,
) -> pyo.Expression:
    """SoC energy balance across each time step.

    At t=1, SoC[s,0] is replaced by the mutable parameter InitialSoC[s].

    Parameters
    ----------
    model: Active ConcreteModel.
    s:     Storage unit index.
    t:     Hour index.
    """
    prev_soc = model.InitialSoC[s] if t == 1 else model.SoC[s, t - 1]
    return (
        model.SoC[s, t]
        == prev_soc
        + model.Charge[s, t] * model.charge_eff[s]
        - model.Discharge[s, t] / model.discharge_eff[s]
    )


def _sim_charge_discharge_rule(
    model: pyo.ConcreteModel,
    s: Any,
    t: int,
) -> pyo.Expression:
    """Prevent (or minimize) simultaneous charging and discharging.

    Discharge is bounded so that the combined charge and discharge do not both
    occur at their rated maxima simultaneously.

    Parameters
    ----------
    model: Active ConcreteModel.
    s:     Storage unit index.
    t:     Hour index.
    """
    return model.Discharge[s, t] <= model.discharge_rate[s] - (
        (model.discharge_rate[s] / model.charge_rate[s]) * model.Charge[s, t]
    )


# ===========================================================================
# ConcreteModel builder
# ===========================================================================

def build_west_linear(
    data: Dict[str, Any],
    horizon_hours: int = 24,
) -> pyo.ConcreteModel:
    """Build and return a Pyomo :class:`~pyomo.environ.ConcreteModel` for the
    U.S. Western Interconnection linear economic dispatch problem.

    The model is built once from the *data* dictionary returned by
    :func:`~gridops.west.launch.load_simulation_data`.  Between solving
    horizons the mutable ``Horizon...`` parameters are updated in-place by
    :func:`~gridops.west.launch._update_horizon_params` — no rebuild is
    needed.

    Parameters
    ----------
    data:
        Data dictionary returned by :func:`~gridops.west.launch.load_simulation_data`.
    horizon_hours:
        Number of hours in each optimization horizon (default 24 hours).

    Returns
    -------
    pyomo.ConcreteModel
        Fully instantiated model with all sets, parameters, variables,
        constraints, and objective defined. Mutable Horizon parameters are
        initialized to 0 and must be updated before the first solve.
    """
    model = pyo.ConcreteModel(name="West_Economic_Dispatch")

    # -----------------------------------------------------------------------
    # TIME SETS
    # -----------------------------------------------------------------------

    # time_periods: 1 ... horizon_hours
    model.time_periods = pyo.RangeSet(1, horizon_hours)

    # -----------------------------------------------------------------------
    # GENERATOR SETS (by fuel type)
    # -----------------------------------------------------------------------

    gen_sets = data["gen_sets"]

    # Primary fuel-type sets
    model.Coal = pyo.Set(initialize=gen_sets.get("Coal", []))
    model.Oil = pyo.Set(initialize=gen_sets.get("Oil", []))
    model.Gas = pyo.Set(initialize=gen_sets.get("Gas", []))
    model.Hydro = pyo.Set(initialize=gen_sets.get("Hydro", []))
    model.Solar = pyo.Set(initialize=gen_sets.get("Solar", []))
    model.Wind = pyo.Set(initialize=gen_sets.get("Wind", []))
    model.Biomass = pyo.Set(initialize=gen_sets.get("Biomass", []))
    model.Geothermal = pyo.Set(initialize=gen_sets.get("Geothermal", []))
    model.OffshoreWind = pyo.Set(initialize=gen_sets.get("OffshoreWind", []))

    # Derived generator sets
    model.Thermal = model.Coal | model.Oil | model.Gas | model.Biomass | model.Geothermal
    model.Generators = model.Thermal | model.Hydro | model.Solar | model.Wind | model.OffshoreWind
    model.Dispatchable = model.Hydro | model.Oil | model.Gas | model.Coal | model.Biomass | model.Geothermal
    model.Outage = model.Coal | model.Gas # Generators subject to capacity-outage adjustments
    model.DispatchableNoOutage = model.Dispatchable - model.Outage # Generators not subject to capacity-outage adjustments

    # -----------------------------------------------------------------------
    # OUTAGE-GROUP SETS (capacity-range buckets used for generation capacity 
    # loss distribution)
    # -----------------------------------------------------------------------
    for group_name, gen_list in data["outage_group_sets"].items():
        setattr(model, group_name, pyo.Set(initialize=gen_list))

    # -----------------------------------------------------------------------
    # STORAGE SET
    # -----------------------------------------------------------------------
    model.Storage = pyo.Set(initialize=data["storage_units"])

    # -----------------------------------------------------------------------
    # NETWORK SETS
    # -----------------------------------------------------------------------
    model.buses = pyo.Set(initialize=data["buses"])
    model.lines = pyo.Set(initialize=data["lines"])
    model.exchanges = pyo.Set(initialize=data["exchanges"])

    # -----------------------------------------------------------------------
    # STATIC GENERATOR PARAMETERS
    # -----------------------------------------------------------------------
    gp = data["gen_params"] # {gen_name: {field: value}}

    model.typ = pyo.Param(model.Generators, initialize={g: gp[g]["typ"] for g in gp}, within=pyo.Any)
    model.bus = pyo.Param(model.Generators, initialize={g: gp[g]["bus"] for g in gp}, within=pyo.Any)
    model.maxcap = pyo.Param(model.Generators, initialize={g: gp[g]["maxcap"] for g in gp})
    model.mincap = pyo.Param(model.Generators, initialize={g: gp[g]["mincap"] for g in gp})
    model.heat_rate = pyo.Param(model.Generators, initialize={g: gp[g]["heat_rate"] for g in gp})
    model.var_om = pyo.Param(model.Generators, initialize={g: gp[g]["var_om"] for g in gp})
    model.no_load = pyo.Param(model.Generators, initialize={g: gp[g]["no_load"] for g in gp})
    model.st_cost = pyo.Param(model.Generators, initialize={g: gp[g]["st_cost"] for g in gp})
    model.ramp = pyo.Param(model.Generators, initialize={g: gp[g]["ramp"] for g in gp})
    model.minup = pyo.Param(model.Generators, initialize={g: gp[g]["minup"] for g in gp})
    model.mindn = pyo.Param(model.Generators, initialize={g: gp[g]["mindn"] for g in gp})

    # -----------------------------------------------------------------------
    # STATIC TRANSMISSION LINE PARAMETERS
    # -----------------------------------------------------------------------
    lp = data["line_params"] # {line_name: {reactance, flow_lim}}

    model.Reactance = pyo.Param(
        model.lines,
        initialize={l: lp[l]["reactance"] for l in lp},
    )
    model.FlowLim = pyo.Param(
        model.lines,
        initialize={l: lp[l]["flow_lim"] for l in lp},
    )

    # -----------------------------------------------------------------------
    # MAPPING PARAMETERS (sparse; default = 0 for unlisted index pairs)
    # -----------------------------------------------------------------------

    # LineToBusMap[line, bus]: DC incidence matrix (−1, 0, +1)
    model.LineToBusMap = pyo.Param(
        model.lines, model.buses,
        initialize=data["line_to_bus_map"],
        default=0,
    )

    # BusToUnitMap[gen, bus]: 1 if generator gen is located at bus
    model.BusToUnitMap = pyo.Param(
        model.Generators, model.buses,
        initialize=data["bus_to_unit_map"],
        default=0,
    )

    # ExchangeHurdle[exchange]: $/MWh hurdle rate for each BA-to-BA exchange
    model.ExchangeHurdle = pyo.Param(
        model.exchanges,
        initialize=data["exchange_hurdle"],
    )

    # ExchangeMap[exchange, line]: Maps exchanges to the lines they encompass
    model.ExchangeMap = pyo.Param(
        model.exchanges, model.lines,
        initialize=data["exchange_map"],
        default=0,
    )

    # BusToStorageMap[storage, bus]: 1 if storage unit is at that bus
    model.BusToStorageMap = pyo.Param(
        model.Storage, model.buses,
        initialize=data["bus_to_storage_map"],
        default=0,
    )

    # -----------------------------------------------------------------------
    # STATIC STORAGE PARAMETERS
    # -----------------------------------------------------------------------
    sp = data["storage_params"]  # {storage_name: {field: value}}

    model.s_typ = pyo.Param(model.Storage, initialize={s: sp[s]["s_typ"] for s in sp}, within=pyo.Any)
    model.s_bus = pyo.Param(model.Storage, initialize={s: sp[s]["s_bus"] for s in sp}, within=pyo.Any)
    model.charge_rate = pyo.Param(model.Storage, initialize={s: sp[s]["charge_rate"] for s in sp})
    model.discharge_rate = pyo.Param(model.Storage, initialize={s: sp[s]["discharge_rate"] for s in sp})
    model.duration = pyo.Param(model.Storage, initialize={s: sp[s]["duration"] for s in sp})
    model.max_SoC = pyo.Param(model.Storage, initialize={s: sp[s]["max_SoC"] for s in sp})
    model.min_SoC = pyo.Param(model.Storage, initialize={s: sp[s]["min_SoC"] for s in sp})
    model.charge_eff = pyo.Param(model.Storage, initialize={s: sp[s]["charge_eff"] for s in sp})
    model.discharge_eff = pyo.Param(model.Storage, initialize={s: sp[s]["discharge_eff"] for s in sp})

    # -----------------------------------------------------------------------
    # MUTABLE HORIZON PARAMETERS
    # Initialised to 0; updated each horizon by _update_horizon_params().
    # -----------------------------------------------------------------------

    # Hourly load per bus for the current horizon
    model.HorizonDemand = pyo.Param(
        model.buses, model.time_periods,
        initialize=0,
        within=pyo.NonNegativeReals,
        mutable=True,
    )
    # Hourly available solar generation per solar generator for the current horizon
    model.HorizonSolar = pyo.Param(
        model.Solar, model.time_periods,
        initialize=0,
        within=pyo.NonNegativeReals,
        mutable=True,
    )
    # Hourly available wind generation per wind generator for the current horizon
    model.HorizonWind = pyo.Param(
        model.Wind, model.time_periods,
        initialize=0,
        within=pyo.NonNegativeReals,
        mutable=True,
    )
    # Hourly available offshore-wind generation per offshore-wind generator for the current horizon
    model.HorizonOffshoreWind = pyo.Param(
        model.OffshoreWind, model.time_periods,
        initialize=0,
        within=pyo.NonNegativeReals,
        mutable=True,
    )
    # Hourly capacity limits for outage-eligible generators (after loss adjustment)
    model.HorizonGenLimit = pyo.Param(
        model.Outage, model.time_periods,
        initialize=0,
        within=pyo.NonNegativeReals,
        mutable=True,
    )
    # Hourly must-run (e.g., nuclear) capacity per bus
    model.HorizonMustrunLimit = pyo.Param(
        model.buses, model.time_periods,
        initialize=0,
        within=pyo.NonNegativeReals,
        mutable=True,
    )
    # Daily scalar hydro limits (same value applied to all hours in the horizon)
    model.HorizonHydro_MAX = pyo.Param(
        model.Hydro,
        initialize=0,
        within=pyo.NonNegativeReals,
        mutable=True,
    )
    model.HorizonHydro_MIN = pyo.Param(
        model.Hydro,
        initialize=0,
        within=pyo.NonNegativeReals,
        mutable=True,
    )
    # Total hydro energy budget for the horizon
    model.HorizonHydro_TOTAL = pyo.Param(
        model.Hydro,
        initialize=0,
        within=pyo.NonNegativeReals,
        mutable=True,
    )
    # Daily fuel price per thermal generator ($/MMBtu)
    model.FuelPrice = pyo.Param(
        model.Thermal,
        initialize=0,
        within=pyo.Reals,
        mutable=True,
    )

    # -----------------------------------------------------------------------
    # PRE-COMPUTED SPARSE LOOKUPS
    # These Python-level dicts accelerate constraint / objective building by
    # avoiding iteration over the full Cartesian product of indices. Each
    # lookup maps an entity to the *non-zero* entries of the corresponding
    # sparse incidence matrix.
    # -----------------------------------------------------------------------

    # {bus: [(line, coeff), ...]} – lines connected to each bus
    _lines_at_bus: Dict[str, list] = {b: [] for b in data["buses"]}
    for (l, b), coeff in data["line_to_bus_map"].items():
        _lines_at_bus[b].append((l, coeff))
    model._lines_at_bus = _lines_at_bus

    # {bus: [gen, ...]} – generators located at each bus
    _gens_at_bus: Dict[str, list] = {b: [] for b in data["buses"]}
    for (g, b) in data["bus_to_unit_map"]:
        _gens_at_bus[b].append(g)
    model._gens_at_bus = _gens_at_bus

    # {bus: [storage, ...]} – storage units located at each bus
    _storage_at_bus: Dict[str, list] = {b: [] for b in data["buses"]}
    for (s, b) in data["bus_to_storage_map"]:
        _storage_at_bus[b].append(s)
    model._storage_at_bus = _storage_at_bus

    # {line: [(bus, coeff), ...]} – buses at each end of a line (typically 2)
    _buses_on_line: Dict[str, list] = {l: [] for l in data["lines"]}
    for (l, b), coeff in data["line_to_bus_map"].items():
        _buses_on_line[l].append((b, coeff))
    model._buses_on_line = _buses_on_line

    # [(key, coeff), ...] – non-zero (exchange, line) pairs with coefficients
    model._exchange_line_pairs = [
        ((k, l), coeff)
        for (k, l), coeff in data["exchange_map"].items()
    ]

    # -----------------------------------------------------------------------
    # DECISION VARIABLES
    # -----------------------------------------------------------------------

    # Generation output (MW) for optimization hours 1...H.
    # Initial state for hour 1 ramp constraints is provided by the mutable
    # parameter InitialMwh (updated between horizons by the launch loop).
    model.mwh = pyo.Var(
        model.Generators, model.time_periods,
        within=pyo.NonNegativeReals,
    )

    # Unserved energy/slack (MW) – only for optimization hours 1...H
    model.S = pyo.Var(
        model.buses, model.time_periods,
        within=pyo.NonNegativeReals,
        initialize=0,
    )

    # DC line flow (MW) – can be bi-directional; bounds = +-flow_lim per line.
    def _flow_bounds(model, l, t):
        _lim = lp[l]["flow_lim"]
        return (-_lim, _lim)

    model.Flow = pyo.Var(
        model.lines, model.time_periods,
        within=pyo.Reals,
        bounds=_flow_bounds,
    )

    # Bus voltage angle (rad) – bounded +-pi
    model.Theta = pyo.Var(
        model.buses, model.time_periods,
        within=pyo.Reals,
        bounds=(-3.1415, 3.1415),
    )

    # Absolute-value proxy for flow (MW)
    model.DummyFlow = pyo.Var(
        model.lines, model.time_periods,
        within=pyo.NonNegativeReals,
    )

    # Storage state of charge (MWh) for optimization hours 1...H.
    # Initial SoC for hour 1 storage constraints is provided by the mutable
    # parameter InitialSoC (updated between horizons by the launch loop).
    def _soc_bounds(model, s, t):
        return (sp[s]["min_SoC"], sp[s]["max_SoC"])

    model.SoC = pyo.Var(
        model.Storage, model.time_periods,
        within=pyo.NonNegativeReals,
        bounds=_soc_bounds,
    )

    # Storage charging power (MW) – upper bound = charge_rate per storage unit.
    def _charge_bounds(model, s, t):
        return (0, sp[s]["charge_rate"])

    model.Charge = pyo.Var(
        model.Storage, model.time_periods,
        within=pyo.NonNegativeReals,
        bounds=_charge_bounds,
    )

    # Storage discharging power (MW) – upper bound = discharge_rate per storage unit.
    def _discharge_bounds(model, s, t):
        return (0, sp[s]["discharge_rate"])

    model.Discharge = pyo.Var(
        model.Storage, model.time_periods,
        within=pyo.NonNegativeReals,
        bounds=_discharge_bounds,
    )

    # -----------------------------------------------------------------------
    # INITIAL STATE – mutable parameters for hour 1 boundary conditions.
    # InitialMwh and InitialSoC are updated between horizons by the launch
    # loop so that storage / ramp constraints at i=1 use the correct values.
    # -----------------------------------------------------------------------

    initial_gen = data.get("initial_gen", {})
    model.InitialMwh = pyo.Param(
        model.Thermal,
        initialize={j: max(0.0, initial_gen.get(j, 0.0)) for j in model.Thermal},
        mutable=True,
    )

    initial_soc = data.get("initial_soc", {})
    model.InitialSoC = pyo.Param(
        model.Storage,
        initialize={
            j: initial_soc.get(j, pyo.value(model.min_SoC[j]))
            for j in model.Storage
        },
        mutable=True,
    )

    # -----------------------------------------------------------------------
    # DUAL VARIABLE SUFFIX (for extracting LMPs from nodal balance duals)
    # -----------------------------------------------------------------------
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # -----------------------------------------------------------------------
    # OBJECTIVE
    # -----------------------------------------------------------------------
    model.SystemCost = pyo.Objective(rule=_sys_cost_rule, sense=pyo.minimize)

    # -----------------------------------------------------------------------
    # CONSTRAINTS
    # -----------------------------------------------------------------------

    # Ramp-rate limits (thermal generators only, all time periods)
    model.RampCon1 = pyo.Constraint(
        model.Thermal, model.time_periods, rule=_ramp_up_rule
    )
    model.RampCon2 = pyo.Constraint(
        model.Thermal, model.time_periods, rule=_ramp_down_rule
    )

    # Generator capacity limits
    model.MaxCap = pyo.Constraint(
        model.Outage, model.time_periods, rule=_max_cap_outage_rule
    )
    
    model.MaxCap2 = pyo.Constraint(
        model.DispatchableNoOutage, model.time_periods, rule=_max_cap_dispatchable_rule
    )

    # Hydro constraints
    model.HydroTOTAL = pyo.Constraint(
        model.Hydro, rule=_hydro_total_rule
    )
    model.HydroMAX = pyo.Constraint(
        model.Hydro, model.time_periods, rule=_hydro_max_rule
    )
    model.HydroMIN = pyo.Constraint(
        model.Hydro, model.time_periods, rule=_hydro_min_rule
    )

    # Available renewable capacity limits
    model.SolarConstraint = pyo.Constraint(
        model.Solar, model.time_periods, rule=_solar_cap_rule
    )
    model.WindConstraint = pyo.Constraint(
        model.Wind, model.time_periods, rule=_wind_cap_rule
    )
    model.OffshoreWindConstraint = pyo.Constraint(
        model.OffshoreWind, model.time_periods, rule=_offshore_wind_cap_rule
    )

    # Bus energy balance
    model.Bus_Constraint = pyo.Constraint(
        model.buses, model.time_periods, rule=_bus_balance_rule
    )

    # DC power-flow equations
    model.DC_Flow_Constraint = pyo.Constraint(
        model.lines, model.time_periods, rule=_dc_flow_rule
    )

    # Reference bus voltage angle = 0
    model.ThetaB_Constraint = pyo.Constraint(
        model.time_periods, rule=_reference_bus_angle_rule
    )

    # Absolute value proxy constraints for dummy flow penalty
    model.DummyFlow1_Constraint = pyo.Constraint(
        model.lines, model.time_periods, rule=_dummy_flow_pos_rule
    )
    model.DummyFlow2_Constraint = pyo.Constraint(
        model.lines, model.time_periods, rule=_dummy_flow_neg_rule
    )

    # Storage operational constraints
    model.MaxCharge_Constraint = pyo.Constraint(
        model.Storage, model.time_periods, rule=_max_charge_soc_rule
    )
    model.MaxDischarge_Constraint = pyo.Constraint(
        model.Storage, model.time_periods, rule=_max_discharge_soc_rule
    )
    model.SoCBalance_Constraint = pyo.Constraint(
        model.Storage, model.time_periods, rule=_soc_balance_rule
    )
    model.SimChargeDischarge_Constraint = pyo.Constraint(
        model.Storage, model.time_periods, rule=_sim_charge_discharge_rule
    )

    return model
