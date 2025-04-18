import pyomo.environ as pyo


def SysCost(model):
    """Objective function that calculates the total system cost based on the generation,
    slack, hydro, wind, solar, exchange, offshore wind, charging storage, discharging storage, demand response up, demand response down, and power flow costs.

    Returns:
    float: Total system cost
    """

    gen = sum(
        model.mwh[j, i] * (model.heat_rate[j] * model.FuelPrice[j] + model.var_om[j]) for i in model.hh_periods for j in
        model.Thermal)
    slack = sum(model.S[z, i] * 2000 for i in model.hh_periods for z in model.buses)
    hydro_cost = sum(model.mwh[j, i] * 0.01 for i in model.hh_periods for j in model.Hydro)
    wind_cost = sum(model.mwh[j, i] * 0.01 for i in model.hh_periods for j in model.Wind)
    offshorewind_cost = sum(model.mwh[j, i] * 0.01 for i in model.hh_periods for j in model.OffshoreWind)
    solar_cost = sum(model.mwh[j, i] * 0.01 for i in model.hh_periods for j in model.Solar)
    exchange_cost = sum(
        model.Flow[l, i] * model.ExchangeMap[k, l] * model.ExchangeHurdle[k] for l in model.lines for i in
        model.hh_periods for k in model.exchanges)
    powerflow_cost = sum(model.DummyFlow[l, i] * 0.01 for l in model.lines for i in model.hh_periods)
    charging_cost = sum(model.Charge[j, i] * 0.001 for i in model.hh_periods for j in model.Storage)
    discharging_cost = sum(model.Discharge[j, i] * 0.001 for i in model.hh_periods for j in model.Storage)
    dr_down_cost = sum(model.DR_Down[z, i] * model.DRCost for i in model.hh_periods for z in model.buses)
    dr_up_cost = sum(model.DR_Up[z, i] * 0.005 for i in model.hh_periods for z in model.buses)
    return gen + slack + hydro_cost + wind_cost + solar_cost + exchange_cost + offshorewind_cost + powerflow_cost + charging_cost + discharging_cost + dr_down_cost + dr_up_cost


def Ramp1(model, j, i):
    """This function defines the ramp down constraint for a generator.
    It ensures that the change in power output from one hour to the next is within the ramp down limit.

    Parameters:
    j (int): Index of the generator
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the ramp down constraint
    """
    a = model.mwh[j, i]
    b = model.mwh[j, i - 1]
    return a - b <= model.ramp[j]


def Ramp2(model, j, i):
    """This function defines the ramp down constraint for a generator.
    It ensures that the change in power output from one hour to the next is within the ramp down limit.

    Parameters:
    j (int): Index of the generator
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the ramp down constraint
    """
    current_output = model.mwh[j, i]
    previous_output = model.mwh[j, i - 1]
    return previous_output - current_output <= model.ramp[j]


def MaxC(model, j, i):
    """Max capacity constraint for outage set generators (coal, NG)

    Parameters:
    j (int): Index of the generator
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the max capacity constraint
    """
    return model.mwh[j, i] <= model.HorizonGenLimit[j, i]


def MaxC2(model, j, i):
    """Max capacity constraint for other dispatchable generators

    Parameters:
    j (int): Index of the generator
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the max capacity constraint
    """
    return model.mwh[j, i] <= model.maxcap[j]


def HydroP(model, j, i):
    """Max production constraints on domestic hydropower

    Parameters:
    j (int): Index of the generator
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the max production constraint
    """
    daily = sum(model.mwh[j, i] for i in model.hh_periods)
    return daily <= model.HorizonHydro_TOTAL[j]


def HydroX(model, j, i):
    """Max capacity constraints on domestic hydropower

    Parameters:
    j (int): Index of the generator
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the max capacity constraint
    """
    return model.mwh[j, i] <= model.HorizonHydro_MAX[j]


def HydroM(model, j, i):
    """Max capacity constraints on domestic hydropower

    Parameters:
    j (int): Index of the generator
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the max capacity constraint
    """
    return model.mwh[j, i] >= model.HorizonHydro_MIN[j]


def SolarC(model, j, i):
    """Max capacity constraints on solar

    Parameters:
    j (int): Index of the generator
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the max capacity constraint
    """
    return model.mwh[j, i] <= model.HorizonSolar[j, i]


def WindC(model, j, i):
    """Max capacity constraints on wind

    Parameters:
    j (int): Index of the generator
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the max capacity constraint
    """
    return model.mwh[j, i] <= model.HorizonWind[j, i]


def OffshoreWindC(model, j, i):
    """Max capacity constraints on offshorewind

    Parameters:
    j (int): Index of the generator
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the max capacity constraint
    """
    return model.mwh[j, i] <= model.HorizonOffshoreWind[j, i]

def Nodal_Balance(model, z, i):
    """Nodal balance constraint.

    Parameters:
    z (int): Index of the bus
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the nodal balance constraint
    """
    power_flow = sum(model.Flow[l, i] * model.LinetoBusMap[l, z] for l in model.lines)
    gen = sum(model.mwh[j, i] * model.BustoUnitMap[j, z] for j in model.Generators)
    slack = model.S[z, i]
    must_run = model.HorizonMustrunLimit[z, i]
    storage_charge = sum(model.Charge[j,i]*model.BustoStorageMap[j,z] for j in model.Storage)
    storage_discharge = sum(model.Discharge[j,i]*model.BustoStorageMap[j,z] for j in model.Storage)
    demand_response_up = model.DR_Up[z, i]
    demand_response_down = model.DR_Down[z, i]
    return gen + slack + must_run - power_flow == model.HorizonDemand[z,i] + storage_charge - storage_discharge + demand_response_up - demand_response_down


def Flow_line(model, l, i):
    """Transmission line flow constraint.

    Parameters:
    l (int): Index of the transmission line
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the transmission line flow constraint
    """
    value = sum(model.Theta[z, i] * model.LinetoBusMap[l, z] for z in model.buses)
    return 100 * value == model.Flow[l, i] * model.Reactance[l]


def Theta_bus(model, i):
    """Bus angle constraint.

    Parameters:
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the bus angle constraint
    """
    return model.Theta['bus_100011', i] == 0


def FlowUP_line(model, l, i):
    """Transmission line flow constraint (upper limit).

    Parameters:
    l (int): Index of the transmission line
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the transmission line flow constraint (upper limit)
    """
    return model.Flow[l, i] <= model.FlowLim[l]


def FlowLow_line(model, l, i):
    """Transmission line flow constraint (lower limit).

    Parameters:
    l (int): Index of the transmission line
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the transmission line flow constraint (lower limit)
    """
    return -1 * model.Flow[l, i] <= model.FlowLim[l]


def DummyFlow1(model, l, i):
    """Dummy flow constraint 1.

    Parameters:
    l (int): Index of the transmission line
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the dummy flow constraint 1
    """
    return model.DummyFlow[l, i] >= model.Flow[l, i]


def DummyFlow2(model, l, i):
    """Dummy flow constraint 2.

    Parameters:
    l (int): Index of the transmission line
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the dummy flow constraint 2
    """
    return model.DummyFlow[l, i] >= model.Flow[l, i] * -1


def MaxCharge1(model, j, i):
    """First maximum charge rate constraint of batteries

    Parameters:
    j (int): Index of the storage facility
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the first maximum charge rate constraint
    """
    return model.Charge[j,i] <= model.charge_rate[j]


def MaxCharge2(model, j, i):
    """Second maximum charge rate constraint of batteries

    Parameters:
    j (int): Index of the storage facility
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the second maximum charge rate constraint
    """
    return model.Charge[j,i] <= (model.max_SoC[j]-model.SoC[j,i-1])/model.charge_eff[j]


def MaxDischarge1(model, j, i):
    """First maximum discharge rate constraint of batteries

    Parameters:
    j (int): Index of the storage facility
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the first maximum discharge rate constraint
    """
    return model.Discharge[j,i] <= model.discharge_rate[j]


def MaxDischarge2(model, j, i):
    """Second maximum discharge rate constraint of batteries

    Parameters:
    j (int): Index of the storage facility
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the second maximum discharge rate constraint
    """
    return model.Discharge[j,i] <= (model.SoC[j,i-1]-model.min_SoC[j])*model.discharge_eff[j]


def MaximumSoC(model, j, i):
    """Maximum state of charge constraint of batteries

    Parameters:
    j (int): Index of the storage facility
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the maximum state of charge constraint
    """
    return model.SoC[j,i] <= model.max_SoC[j]


def MinimumSoC(model, j, i):
    """Minimum state of charge constraint of batteries

    Parameters:
    j (int): Index of the storage facility
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the minimum state of charge constraint
    """
    return model.SoC[j,i] >= model.min_SoC[j]


def SoCBalance(model, j, i):
    """State of charge balance constraint of batteries

    Parameters:
    j (int): Index of the storage facility
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the state of charge balance constraint
    """
    return model.SoC[j,i] == model.SoC[j,i-1] + (model.Charge[j,i]*model.charge_eff[j]) - (model.Discharge[j,i]/model.discharge_eff[j])


def SimChargeDischarge(model, j, i):
    """Constraint to minimize simultaneous charge and discharge of batteries

    Parameters:
    j (int): Index of the storage facility
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the simultaneous charge and discharge of batteries constraint
    """
    return model.Discharge[j,i] <= model.discharge_rate[j]-((model.discharge_rate[j]/model.charge_rate[j])*model.Charge[j,i])


def MaxDR_Up(model, z, i):
    """Maximum demand response up amount constraint 

    Parameters:
    z (int): Index of demand response node
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the maximum demand response up amount 
    """
    return model.DR_Up[z, i] <= model.HorizonDR_up[z, i]


def MaxDR_Down(model, z, i):
    """Maximum demand response down amount constraint 

    Parameters:
    z (int): Index of demand response node
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for the maximum demand response down amount 
    """
    return model.DR_Down[z, i] <= model.HorizonDR_down[z, i]


def Sim_DRUp_DRDown(model, z, i):
    """Constraint to minimize simultaneous demand response up and demand response down

    Parameters:
    z (int): Index of demand response node
    i (int): Index of the hour

    Returns:
    pyo.Constraint: Constraint object for minimizing simultaneous demand response up and demand response down
    """
    if model.HorizonDR_down[z, i].value != 0 and model.HorizonDR_up[z, i].value != 0:
        return model.DR_Down[z, i] <= model.HorizonDR_down[z, i]-((model.HorizonDR_down[z, i]/model.HorizonDR_up[z, i])*model.DR_Up[z, i])
    
    else:
        return pyo.Constraint.Skip
    

def DR_shifting(model, z):
    """Constraint to shift the demand within a day with demand response

    Parameters:
    z (int): Index of demand response node
    
    Returns:
    pyo.Constraint: Constraint object for shifting the demand within a day with demand response (i.e., total demand within a day does not change but only shifts between hours)
    """
    
    daily_DR_down_limit = sum(model.HorizonDR_down[z, i].value for i in model.hh_periods)
    daily_DR_up_limit = sum(model.HorizonDR_up[z, i].value for i in model.hh_periods)

    if daily_DR_down_limit != 0 and daily_DR_up_limit != 0:

        daily_DR_down = sum(model.DR_Down[z, i] for i in model.hh_periods)
        daily_DR_up = sum(model.DR_Up[z, i] for i in model.hh_periods)

        return daily_DR_down == daily_DR_up
    
    else:
        return pyo.Constraint.Skip


def model_west_linear_multi(*args, **kwargs):
    """This class defines an abstract model for a linear optimization problem for the Western Interconnection."""

    # instantiate model
    model = pyo.AbstractModel()

    # Sets of generators by fuel-type
    # Coal generators
    model.Coal = pyo.Set()

    # Oil generators
    model.Oil = pyo.Set()

    # Gas generators
    model.Gas = pyo.Set()

    # Hydro generators
    model.Hydro = pyo.Set()

    # Solar generators
    model.Solar = pyo.Set()

    # Wind generators
    model.Wind = pyo.Set()

    # Biomass generators
    model.Biomass = pyo.Set()

    # Geothermal generators
    model.Geothermal = pyo.Set()

    # Offshore wind generators
    model.OffshoreWind = pyo.Set()

    # Storage facilities
    model.Storage = pyo.Set()

    # Define sets of generators by fuel-type
    model.Thermal = model.Coal | model.Oil | model.Gas | model.Biomass | model.Geothermal
    model.Generators = model.Thermal | model.Hydro | model.Solar | model.Wind | model.OffshoreWind
    model.Dispatchable = model.Hydro | model.Oil | model.Gas | model.Coal | model.Biomass | model.Geothermal

    # Define set of generators that can experience outages
    model.Outage = model.Coal | model.Gas

    # Define sets of gas generators by capacity range
    model.Gas_below_50 = pyo.Set()
    model.Gas_50_100 = pyo.Set()
    model.Gas_100_200 = pyo.Set()
    model.Gas_200_300 = pyo.Set()
    model.Gas_300_400 = pyo.Set()
    model.Gas_400_600 = pyo.Set()
    model.Gas_600_800 = pyo.Set()
    model.Gas_800_1000 = pyo.Set()
    model.Gas_ovr_1000 = pyo.Set()

    # Define sets of gas generators by capacity range and no-load cost
    model.Gas_All_n_0_100 = pyo.Set()
    model.Gas_All_n_100_200 = pyo.Set()
    model.Gas_All_n_ovr_200 = pyo.Set()

    # Define sets of coal generators by capacity range
    model.Coal_below_50 = pyo.Set()
    model.Coal_50_100 = pyo.Set()
    model.Coal_100_200 = pyo.Set()
    model.Coal_200_300 = pyo.Set()
    model.Coal_300_400 = pyo.Set()
    model.Coal_400_600 = pyo.Set()
    model.Coal_600_800 = pyo.Set()
    model.Coal_800_1000 = pyo.Set()
    model.Coal_ovr_1000 = pyo.Set()

    # Define sets of coal generators by capacity range and no-load cost
    model.Coal_All_n_0_100 = pyo.Set()
    model.Coal_All_n_100_200 = pyo.Set()
    model.Coal_All_n_ovr_200 = pyo.Set()

    # Transmission sets
    # Set of transmission lines
    model.lines = pyo.Set()

    # Set of buses
    model.buses = pyo.Set()

    # BA to BA transmission sets
    model.exchanges = pyo.Set()  # Set of exchanges between balancing authorities

    # Generator parameters
    # Type of generator (e.g. coal, gas, wind)
    model.typ = pyo.Param(model.Generators, within=pyo.Any)

    # Name of the node where the generator is located
    model.node = pyo.Param(model.Generators, within=pyo.Any)

    # Maximum capacity of the generator
    model.maxcap = pyo.Param(model.Generators)

    # Minimum capacity of the generator
    model.mincap = pyo.Param(model.Generators)

    # Heat rate of the generator
    model.heat_rate = pyo.Param(model.Generators)

    # Variable O&M cost of the generator
    model.var_om = pyo.Param(model.Generators)

    # Fixed O&M cost
    model.no_load = pyo.Param(model.Generators)

    # Start cost
    model.st_cost = pyo.Param(model.Generators)

    # Ramp rate
    model.ramp = pyo.Param(model.Generators)

    # Minimum up time
    model.minup = pyo.Param(model.Generators)

    # Minimum down time
    model.mindn = pyo.Param(model.Generators)

    # Transmission line reactance
    model.Reactance = pyo.Param(model.lines)

    # Transmission line flow limit
    model.FlowLim = pyo.Param(model.lines)

    # Mapping of transmission lines to buses
    model.LinetoBusMap = pyo.Param(model.lines, model.buses)

    # Mapping of generators to buses
    model.BustoUnitMap = pyo.Param(model.Generators, model.buses)

    # Exchange hurdle
    model.ExchangeHurdle = pyo.Param(model.exchanges)

    # Mapping of exchanges to transmission lines
    model.ExchangeMap = pyo.Param(model.exchanges, model.lines, mutable=True)

    # Type of storage (e.g., battery, pumped storage hydro)
    model.s_typ = pyo.Param(model.Storage, within=pyo.Any)

    # Name of the node where the storage facility is located
    model.s_node = pyo.Param(model.Storage, within=pyo.Any)

    # Charge rate of storage facility
    model.charge_rate = pyo.Param(model.Storage)

    # Discharge rate of storage facility
    model.discharge_rate = pyo.Param(model.Storage)

    # Duration of storage facility
    model.duration = pyo.Param(model.Storage)

    # Maximum state of charge (SoC) of storage facility
    model.max_SoC = pyo.Param(model.Storage)

    # Minimum state of charge (SoC) of storage facility
    model.min_SoC = pyo.Param(model.Storage)

    # Charge efficiency of storage facility
    model.charge_eff = pyo.Param(model.Storage)

    # Discharge efficiency of storage facility
    model.discharge_eff = pyo.Param(model.Storage)

    # Mapping of storage facilities to buses
    model.BustoStorageMap = pyo.Param(model.Storage, model.buses)

    # Full range of time series information
    # Total number of simulation hours
    model.SimHours = pyo.Param(within=pyo.PositiveIntegers)

    # Range set for simulation hours
    model.SH_periods = pyo.RangeSet(1, model.SimHours + 1)

    # Total number of simulation days
    model.SimDays = pyo.Param(within=pyo.PositiveIntegers)

    # Range set for simulation days
    model.SD_periods = pyo.RangeSet(1, model.SimDays + 1)

    # Operating horizon information
    # Total number of hours in the operating horizon
    model.HorizonHours = pyo.Param(within=pyo.PositiveIntegers)

    # Cost of demand response down
    model.DRCost = pyo.Param(within=pyo.NonNegativeReals)

    # Range set for the operating horizon
    model.HH_periods = pyo.RangeSet(0, model.HorizonHours)

    # Range set for the operating horizon, excluding the first hour
    model.hh_periods = pyo.RangeSet(1, model.HorizonHours)

    # Range set for ramping constraints
    model.ramp_periods = pyo.RangeSet(2, 24)

    # Demand over simulation period
    model.SimDemand = pyo.Param(model.buses * model.SH_periods, within=pyo.NonNegativeReals)

    # Horizon demand
    model.HorizonDemand = pyo.Param(model.buses * model.hh_periods, within=pyo.NonNegativeReals, mutable=True)

    # Variable resources over simulation period
    # Maximum hydro capacity over simulation period
    model.SimHydro_MAX = pyo.Param(model.Hydro, model.SH_periods, within=pyo.NonNegativeReals)

    # Minimum hydro capacity over simulation period
    model.SimHydro_MIN = pyo.Param(model.Hydro, model.SH_periods, within=pyo.NonNegativeReals)

    # Total hydro capacity over simulation period
    model.SimHydro_TOTAL = pyo.Param(model.Hydro, model.SH_periods, within=pyo.NonNegativeReals)

    # Solar
    # Solar capacity over simulation period
    model.SimSolar = pyo.Param(model.Solar, model.SH_periods, within=pyo.NonNegativeReals)

    # Wind
    # Wind capacity over simulation period
    model.SimWind = pyo.Param(model.Wind, model.SH_periods, within=pyo.NonNegativeReals)

    # Offshore Wind
    # Offshore wind capacity over simulation period
    model.SimOffshoreWind = pyo.Param(model.OffshoreWind, model.SH_periods, within=pyo.NonNegativeReals)

    # Lost capacity due to outage
    model.SimGenLimit = pyo.Param(model.Outage, model.SH_periods, within=pyo.NonNegativeReals)
    model.SimMustrunLimit = pyo.Param(model.buses, model.SH_periods, within=pyo.NonNegativeReals)

    # Maximum hydro capacity over horizon
    model.HorizonHydro_MAX = pyo.Param(model.Hydro, within=pyo.NonNegativeReals, mutable=True)

    # Minimum hydro capacity over horizon
    model.HorizonHydro_MIN = pyo.Param(model.Hydro, within=pyo.NonNegativeReals, mutable=True)

    # Total hydro capacity over horizon
    model.HorizonHydro_TOTAL = pyo.Param(model.Hydro, within=pyo.NonNegativeReals, mutable=True)

    # Solar capacity over horizon
    model.HorizonSolar = pyo.Param(model.Solar, model.hh_periods, within=pyo.NonNegativeReals, mutable=True)

    # Wind capacity over horizon
    model.HorizonWind = pyo.Param(model.Wind, model.hh_periods, within=pyo.NonNegativeReals, mutable=True)

    # Offshore wind capacity over horizon
    model.HorizonOffshoreWind = pyo.Param(model.OffshoreWind, model.hh_periods, within=pyo.NonNegativeReals,
                                            mutable=True)

    # Lost capacity due to outage over horizon
    model.HorizonGenLimit = pyo.Param(model.Outage, model.hh_periods, within=pyo.NonNegativeReals, mutable=True)

    # Maximum amount of time a generator must run over horizon
    model.HorizonMustrunLimit = pyo.Param(model.buses, model.hh_periods, within=pyo.NonNegativeReals, mutable=True)

    # Fuel prices over simulation period
    model.SimFuelPrice = pyo.Param(model.Thermal, model.SD_periods, within=pyo.Reals)

    # Fuel prices over horizon
    model.FuelPrice = pyo.Param(model.Thermal, within=pyo.Reals, mutable=True)

    # Maximum demand response up amount
    model.SimDR_up = pyo.Param(model.buses, model.SH_periods, within=pyo.NonNegativeReals, initialize=0)
    model.HorizonDR_up = pyo.Param(model.buses, model.hh_periods, within=pyo.NonNegativeReals, mutable=True, initialize=0)
    
    # Maximum demand response down amount
    model.SimDR_down = pyo.Param(model.buses, model.SH_periods, within=pyo.NonNegativeReals, initialize=0)
    model.HorizonDR_down = pyo.Param(model.buses, model.hh_periods, within=pyo.NonNegativeReals, mutable=True, initialize=0)

    # Amount of day-ahead energy generated by each generator at each hour
    model.mwh = pyo.Var(model.Generators, model.HH_periods, within=pyo.NonNegativeReals, initialize=0)

    # slack variables
    model.S = pyo.Var(model.buses, model.hh_periods, within=pyo.NonNegativeReals, initialize=0)

    # transmission line variables
    model.Flow = pyo.Var(model.lines, model.hh_periods, initialize=0)

    # transmission line variables
    model.Theta = pyo.Var(model.buses, model.hh_periods, bounds=(-3.1415, 3.1415))

    # This is created to enforce a penalty on power flows, which prevents slack generation to be transmitted elsewhere in the grid.
    model.DummyFlow = pyo.Var(model.lines, model.hh_periods, initialize=0)

    # State of charge variables of batteries
    model.SoC = pyo.Var(model.Storage, model.HH_periods, within=pyo.NonNegativeReals)

    # Charging variables of batteries
    model.Charge = pyo.Var(model.Storage, model.hh_periods, within=pyo.NonNegativeReals, initialize=0)

    # Discharging variables of batteries
    model.Discharge = pyo.Var(model.Storage, model.hh_periods, within=pyo.NonNegativeReals, initialize=0)

    # Demand response up variables
    model.DR_Up = pyo.Var(model.buses, model.hh_periods, within=pyo.NonNegativeReals, initialize=0)

    # Demand response down variables
    model.DR_Down = pyo.Var(model.buses, model.hh_periods, within=pyo.NonNegativeReals, initialize=0)
    
    # Objective function to minimize system cost
    model.SystemCost = pyo.Objective(rule=SysCost, sense=pyo.minimize)

    # Ramp up constraint
    model.RampCon1 = pyo.Constraint(model.Thermal, model.ramp_periods, rule=Ramp1)

    # Ramp down constraint
    model.RampCon2 = pyo.Constraint(model.Thermal, model.ramp_periods, rule=Ramp2)

    # Maximum capacity constraint
    model.MaxCap = pyo.Constraint(model.Outage, model.hh_periods, rule=MaxC)

    # Maximum capacity constraint for dispatchable generators
    model.MaxCap2 = pyo.Constraint(model.Dispatchable, model.hh_periods, rule=MaxC2)

    # Maximum production constraints on domestic hydropower
    model.HydroPROD = pyo.Constraint(model.Hydro, model.hh_periods, rule=HydroP)

    # Maximum hydro capacity constraints
    model.HydroMAX = pyo.Constraint(model.Hydro, model.hh_periods, rule=HydroX)

    # Minimum hydro capacity constraints
    model.HydroMIN = pyo.Constraint(model.Hydro, model.hh_periods, rule=HydroM)

    # Solar capacity constraints
    model.SolarConstraint = pyo.Constraint(model.Solar, model.hh_periods, rule=SolarC)

    # Wind capacity constraints
    model.WindConstraint = pyo.Constraint(model.Wind, model.hh_periods, rule=WindC)

    # Offshore wind capacity constraints
    model.OffshoreWindConstraint = pyo.Constraint(model.OffshoreWind, model.hh_periods, rule=OffshoreWindC)

    # Nodal balance constraints
    model.Node_Constraint = pyo.Constraint(model.buses, model.hh_periods, rule=Nodal_Balance)

    # Transmission line flow constraints
    model.FlowL_Constraint = pyo.Constraint(model.lines, model.hh_periods, rule=Flow_line)

    # Bus angle constraints
    model.ThetaB_Constraint = pyo.Constraint(model.hh_periods, rule=Theta_bus)

    # Transmission line flow constraints
    model.FlowU_Constraint = pyo.Constraint(model.lines, model.hh_periods, rule=FlowUP_line)

    # Transmission line flow constraints
    model.FlowLL_Constraint = pyo.Constraint(model.lines, model.hh_periods, rule=FlowLow_line)

    # Dummy flow constraints
    model.DummyFlow1_Constraint = pyo.Constraint(model.lines, model.hh_periods, rule=DummyFlow1)

    # Dummy flow constraints
    model.DummyFlow2_Constraint = pyo.Constraint(model.lines, model.hh_periods, rule=DummyFlow2)

    # First maximum charge constraint
    model.MaxCharge1_Constraint = pyo.Constraint(model.Storage, model.hh_periods, rule=MaxCharge1)

    # Second maximum charge constraint
    model.MaxCharge2_Constraint = pyo.Constraint(model.Storage, model.hh_periods, rule=MaxCharge2)

    # First maximum discharge constraint
    model.MaxDischarge1_Constraint = pyo.Constraint(model.Storage, model.hh_periods, rule=MaxDischarge1)

    # Second maximum discharge constraint
    model.MaxDischarge2_Constraint = pyo.Constraint(model.Storage, model.hh_periods, rule=MaxDischarge2)

    # Maximum state of charge constraint
    model.MaximumSoC_Constraint = pyo.Constraint(model.Storage, model.hh_periods, rule=MaximumSoC)

    # Minimum state of charge constraint
    model.MinimumSoC_Constraint = pyo.Constraint(model.Storage, model.hh_periods, rule=MinimumSoC)

    # State of charge balance constraint
    model.SoCBalance_Constraint = pyo.Constraint(model.Storage, model.hh_periods, rule=SoCBalance)

    # Simultaneous charge and discharge constraint
    model.SimChargeDischarge_Constraint = pyo.Constraint(model.Storage, model.hh_periods, rule=SimChargeDischarge)

    # Maximum demand response up constraint
    model.MaxDR_Up_Constraint = pyo.Constraint(model.buses, model.hh_periods, rule=MaxDR_Up)

    # Maximum demand response down constraint
    model.MaxDR_Down_Constraint = pyo.Constraint(model.buses, model.hh_periods, rule=MaxDR_Down)

    # Simultaneous demand response up and demand response down constraint
    model.Sim_DRUp_DRDown_Constraint = pyo.Constraint(model.buses, model.hh_periods, rule=Sim_DRUp_DRDown)

    # Shifting the demand within a day with demand response constraint
    model.DR_shifting_Constraint = pyo.Constraint(model.buses, rule=DR_shifting)

    return model