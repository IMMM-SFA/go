import pyomo.environ as pyo


class WestLinearMultiModel(pyo.AbstractModel):
    """This class defines an abstract model for a linear optimization problem for the Western Interconnection."""

    def __init__(self, *args, **kwargs):
        """
        Initializes the WestLinearMultiModel class.

        Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)

        # Sets of generators by fuel-type
        # Coal generators
        self.Coal = pyo.Set()

        # Oil generators
        self.Oil = pyo.Set()

        # Gas generators
        self.Gas = pyo.Set()

        # Hydro generators
        self.Hydro = pyo.Set()

        # Solar generators
        self.Solar = pyo.Set()

        # Wind generators
        self.Wind = pyo.Set()

        # Biomass generators
        self.Biomass = pyo.Set()

        # Geothermal generators
        self.Geothermal = pyo.Set()

        # Offshore wind generators
        self.OffshoreWind = pyo.Set()

        # Define sets of generators by fuel-type
        self.Thermal = self.Coal | self.Oil | self.Gas | self.Biomass | self.Geothermal
        self.Generators = self.Thermal | self.Hydro | self.Solar | self.Wind | self.OffshoreWind
        self.Dispatchable = self.Hydro | self.Oil | self.Gas | self.Coal | self.Biomass | self.Geothermal

        # Define set of generators that can experience outages
        self.Outage = self.Coal | self.Gas

        # Define sets of gas generators by capacity range
        self.Gas_below_50 = pyo.Set()
        self.Gas_50_100 = pyo.Set()
        self.Gas_100_200 = pyo.Set()
        self.Gas_200_300 = pyo.Set()
        self.Gas_300_400 = pyo.Set()
        self.Gas_400_600 = pyo.Set()
        self.Gas_600_800 = pyo.Set()
        self.Gas_800_1000 = pyo.Set()
        self.Gas_ovr_1000 = pyo.Set()

        # Define sets of gas generators by capacity range and no-load cost
        self.Gas_All_n_0_100 = pyo.Set()
        self.Gas_All_n_100_200 = pyo.Set()
        self.Gas_All_n_ovr_200 = pyo.Set()

        # Define sets of coal generators by capacity range
        self.Coal_below_50 = pyo.Set()
        self.Coal_50_100 = pyo.Set()
        self.Coal_100_200 = pyo.Set()
        self.Coal_200_300 = pyo.Set()
        self.Coal_300_400 = pyo.Set()
        self.Coal_400_600 = pyo.Set()
        self.Coal_600_800 = pyo.Set()
        self.Coal_800_1000 = pyo.Set()
        self.Coal_ovr_1000 = pyo.Set()

        # Define sets of coal generators by capacity range and no-load cost
        self.Coal_All_n_0_100 = pyo.Set()
        self.Coal_All_n_100_200 = pyo.Set()
        self.Coal_All_n_ovr_200 = pyo.Set()

        # Transmission sets
        # Set of transmission lines
        self.lines = pyo.Set()

        # Set of buses
        self.buses = pyo.Set()

        # BA to BA transmission sets
        self.exchanges = pyo.Set()  # Set of exchanges between balancing authorities

        # Generator parameters
        # Type of generator (e.g. coal, gas, wind)
        self.typ = pyo.Param(self.Generators, within=pyo.Any)

        # Name of the node where the generator is located
        self.node = pyo.Param(self.Generators, within=pyo.Any)

        # Maximum capacity of the generator
        self.maxcap = pyo.Param(self.Generators)

        # Minimum capacity of the generator
        self.mincap = pyo.Param(self.Generators)

        # Heat rate of the generator
        self.heat_rate = pyo.Param(self.Generators)

        # Variable O&M cost of the generator
        self.var_om = pyo.Param(self.Generators)

        # Fixed O&M cost
        self.no_load = pyo.Param(self.Generators)

        # Start cost
        self.st_cost = pyo.Param(self.Generators)

        # Ramp rate
        self.ramp = pyo.Param(self.Generators)

        # Minimum up time
        self.minup = pyo.Param(self.Generators)

        # Minimum down time
        self.mindn = pyo.Param(self.Generators)

        # Transmission line reactance
        self.Reactance = pyo.Param(self.lines)

        # Transmission line flow limit
        self.FlowLim = pyo.Param(self.lines)

        # Mapping of transmission lines to buses
        self.LinetoBusMap = pyo.Param(self.lines, self.buses)

        # Mapping of generators to buses
        self.BustoUnitMap = pyo.Param(self.Generators, self.buses)

        # Exchange hurdle
        self.ExchangeHurdle = pyo.Param(self.exchanges)

        # Mapping of exchanges to transmission lines
        self.ExchangeMap = pyo.Param(self.exchanges, self.lines, mutable=True)

        # Full range of time series information
        # Total number of simulation hours
        self.SimHours = pyo.Param(within=pyo.PositiveIntegers)

        # Range set for simulation hours
        self.SH_periods = pyo.RangeSet(1, self.SimHours + 1)

        # Total number of simulation days
        self.SimDays = pyo.Param(within=pyo.PositiveIntegers)

        # Range set for simulation days
        self.SD_periods = pyo.RangeSet(1, self.SimDays + 1)

        # Operating horizon information
        # Total number of hours in the operating horizon
        self.HorizonHours = pyo.Param(within=pyo.PositiveIntegers)

        # Range set for the operating horizon
        self.HH_periods = pyo.RangeSet(0, self.HorizonHours)

        # Range set for the operating horizon, excluding the first hour
        self.hh_periods = pyo.RangeSet(1, self.HorizonHours)

        # Range set for ramping constraints
        self.ramp_periods = pyo.RangeSet(2, 24)

        # Demand over simulation period
        self.SimDemand = pyo.Param(self.buses * self.SH_periods, within=pyo.NonNegativeReals)

        # Horizon demand
        self.HorizonDemand = pyo.Param(self.buses * self.hh_periods, within=pyo.NonNegativeReals, mutable=True)

        # Variable resources over simulation period
        # Maximum hydro capacity over simulation period
        self.SimHydro_MAX = pyo.Param(self.Hydro, self.SH_periods, within=pyo.NonNegativeReals)

        # Minimum hydro capacity over simulation period
        self.SimHydro_MIN = pyo.Param(self.Hydro, self.SH_periods, within=pyo.NonNegativeReals)

        # Total hydro capacity over simulation period
        self.SimHydro_TOTAL = pyo.Param(self.Hydro, self.SH_periods, within=pyo.NonNegativeReals)

        # Solar
        # Solar capacity over simulation period
        self.SimSolar = pyo.Param(self.Solar, self.SH_periods, within=pyo.NonNegativeReals)

        # Wind
        # Wind capacity over simulation period
        self.SimWind = pyo.Param(self.Wind, self.SH_periods, within=pyo.NonNegativeReals)

        # Offshore Wind
        # Offshore wind capacity over simulation period
        self.SimOffshoreWind = pyo.Param(self.OffshoreWind, self.SH_periods, within=pyo.NonNegativeReals)

        # Lost capacity due to outage
        self.SimGenLimit = pyo.Param(self.Outage, self.SH_periods, within=pyo.NonNegativeReals)
        self.SimMustrunLimit = pyo.Param(self.buses, self.SH_periods, within=pyo.NonNegativeReals)

        # Maximum hydro capacity over horizon
        self.HorizonHydro_MAX = pyo.Param(self.Hydro, within=pyo.NonNegativeReals, mutable=True)

        # Minimum hydro capacity over horizon
        self.HorizonHydro_MIN = pyo.Param(self.Hydro, within=pyo.NonNegativeReals, mutable=True)

        # Total hydro capacity over horizon
        self.HorizonHydro_TOTAL = pyo.Param(self.Hydro, within=pyo.NonNegativeReals, mutable=True)

        # Solar capacity over horizon
        self.HorizonSolar = pyo.Param(self.Solar, self.hh_periods, within=pyo.NonNegativeReals, mutable=True)

        # Wind capacity over horizon
        self.HorizonWind = pyo.Param(self.Wind, self.hh_periods, within=pyo.NonNegativeReals, mutable=True)

        # Offshore wind capacity over horizon
        self.HorizonOffshoreWind = pyo.Param(self.OffshoreWind, self.hh_periods, within=pyo.NonNegativeReals,
                                             mutable=True)

        # Lost capacity due to outage over horizon
        self.HorizonGenLimit = pyo.Param(self.Outage, self.hh_periods, within=pyo.NonNegativeReals, mutable=True)

        # Maximum amount of time a generator must run over horizon
        self.HorizonMustrunLimit = pyo.Param(self.buses, self.hh_periods, within=pyo.NonNegativeReals, mutable=True)

        # Fuel prices over simulation period
        self.SimFuelPrice = pyo.Param(self.Thermal, self.SD_periods, within=pyo.NonNegativeReals)

        # Fuel prices over horizon
        self.FuelPrice = pyo.Param(self.Thermal, within=pyo.NonNegativeReals, mutable=True)

        # Amount of day-ahead energy generated by each generator at each hour
        self.mwh = pyo.Var(self.Generators, self.HH_periods, within=pyo.NonNegativeReals, initialize=0)

        # slack variables
        self.S = pyo.Var(self.buses, self.hh_periods, within=pyo.NonNegativeReals, initialize=0)

        # transmission line variables
        self.Flow = pyo.Var(self.lines, self.hh_periods, initialize=0)

        # transmission line variables
        self.Theta = pyo.Var(self.buses, self.hh_periods)

        # This is created to enforce a penalty on power flows, which prevents slack generation to be transmitted elsewhere in the grid.
        self.DummyFlow = pyo.Var(self.lines, self.hh_periods, initialize=0)

        # Objective function to minimize system cost
        self.SystemCost = pyo.Objective(rule=self.SysCost, sense=pyo.minimize)

        # Ramp up constraint
        self.RampCon1 = pyo.Constraint(self.Thermal, self.ramp_periods, rule=self.Ramp1)

        # Ramp down constraint
        self.RampCon2 = pyo.Constraint(self.Thermal, self.ramp_periods, rule=self.Ramp2)

        # Maximum capacity constraint
        self.MaxCap = pyo.Constraint(self.Outage, self.hh_periods, rule=self.MaxC)

        # Maximum capacity constraint for dispatchable generators
        self.MaxCap2 = pyo.Constraint(self.Dispatchable, self.hh_periods, rule=self.MaxC2)

        # Maximum production constraints on domestic hydropower
        self.HydroPROD = pyo.Constraint(self.Hydro, self.hh_periods, rule=self.HydroP)

        # Maximum hydro capacity constraints
        self.HydroMAX = pyo.Constraint(self.Hydro, self.hh_periods, rule=self.HydroX)

        # Minimum hydro capacity constraints
        self.HydroMIN = pyo.Constraint(self.Hydro, self.hh_periods, rule=self.HydroM)

        # Solar capacity constraints
        self.SolarConstraint = pyo.Constraint(self.Solar, self.hh_periods, rule=self.SolarC)

        # Wind capacity constraints
        self.WindConstraint = pyo.Constraint(self.Wind, self.hh_periods, rule=self.WindC)

        # Offshore wind capacity constraints
        self.OffshoreWindConstraint = pyo.Constraint(self.OffshoreWind, self.hh_periods, rule=self.OffshoreWindC)

        # Nodal balance constraints
        self.Node_Constraint = pyo.Constraint(self.buses, self.hh_periods, rule=self.Nodal_Balance)

        # Transmission line flow constraints
        self.FlowL_Constraint = pyo.Constraint(self.lines, self.hh_periods, rule=self.Flow_line)

        # Bus angle constraints
        self.ThetaB_Constraint = pyo.Constraint(self.hh_periods, rule=self.Theta_bus)

        # Transmission line flow constraints
        self.FlowU_Constraint = pyo.Constraint(self.lines, self.hh_periods, rule=self.FlowUP_line)

        # Transmission line flow constraints
        self.FlowLL_Constraint = pyo.Constraint(self.lines, self.hh_periods, rule=self.FlowLow_line)

        # Dummy flow constraints
        self.DummyFlow1_Constraint = pyo.Constraint(self.lines, self.hh_periods, rule=self.DummyFlow1)

        # Dummy flow constraints
        self.DummyFlow2_Constraint = pyo.Constraint(self.lines, self.hh_periods, rule=self.DummyFlow2)

    def SysCost(self):
        """Objective function that calculates the total system cost based on the generation,
        slack, hydro, wind, solar, exchange, offshore wind, and power flow costs.

        Returns:
        float: Total system cost
        """

        gen = sum(
            self.mwh[j, i] * (self.heat_rate[j] * self.FuelPrice[j] + self.var_om[j]) for i in self.hh_periods for j in
            self.Thermal)
        slack = sum(self.S[z, i] * 2000 for i in self.hh_periods for z in self.buses)
        hydro_cost = sum(self.mwh[j, i] * 0.01 for i in self.hh_periods for j in self.Hydro)
        wind_cost = sum(self.mwh[j, i] * 0.01 for i in self.hh_periods for j in self.Wind)
        offshorewind_cost = sum(self.mwh[j, i] * 0.01 for i in self.hh_periods for j in self.OffshoreWind)
        solar_cost = sum(self.mwh[j, i] * 0.01 for i in self.hh_periods for j in self.Solar)
        exchange_cost = sum(
            self.Flow[l, i] * self.ExchangeMap[k, l] * self.ExchangeHurdle[k] for l in self.lines for i in
            self.hh_periods for k in self.exchanges)
        powerflow_cost = sum(self.DummyFlow[l, i] * 0.01 for l in self.lines for i in self.hh_periods)
        return gen + slack + hydro_cost + wind_cost + solar_cost + exchange_cost + offshorewind_cost + powerflow_cost

    def Ramp1(self, j, i):
        """This function defines the ramp down constraint for a generator.
        It ensures that the change in power output from one hour to the next is within the ramp down limit.

        Parameters:
        j (int): Index of the generator
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the ramp down constraint
        """
        a = self.mwh[j, i]
        b = self.mwh[j, i - 1]
        return a - b <= self.ramp[j]

    def Ramp2(self, j, i):
        """This function defines the ramp down constraint for a generator.
        It ensures that the change in power output from one hour to the next is within the ramp down limit.

        Parameters:
        j (int): Index of the generator
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the ramp down constraint
        """
        current_output = self.mwh[j, i]
        previous_output = self.mwh[j, i - 1]
        return previous_output - current_output <= self.ramp[j]

    def MaxC(self, j, i):
        """Max capacity constraint for outage set generators (coal, NG)

        Parameters:
        j (int): Index of the generator
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the max capacity constraint
        """
        return self.mwh[j, i] <= self.HorizonGenLimit[j, i]

    def MaxC2(self, j, i):
        """Max capacity constraint for other dispatchable generators

        Parameters:
        j (int): Index of the generator
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the max capacity constraint
        """
        return self.mwh[j, i] <= self.maxcap[j]


    def HydroP(self, j, i):
        """Max production constraints on domestic hydropower

        Parameters:
        j (int): Index of the generator
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the max production constraint
        """
        daily = sum(self.mwh[j, i] for i in self.hh_periods)
        return daily <= self.HorizonHydro_TOTAL[j]


    def HydroX(self, j, i):
        """Max capacity constraints on domestic hydropower

        Parameters:
        j (int): Index of the generator
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the max capacity constraint
        """
        return self.mwh[j, i] <= self.HorizonHydro_MAX[j]

    def HydroM(self, j, i):
        """Max capacity constraints on domestic hydropower

        Parameters:
        j (int): Index of the generator
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the max capacity constraint
        """
        return self.mwh[j, i] >= self.HorizonHydro_MIN[j]

    def SolarC(self, j, i):
        """Max capacity constraints on solar

        Parameters:
        j (int): Index of the generator
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the max capacity constraint
        """
        return self.mwh[j, i] <= self.HorizonSolar[j, i]

    def WindC(self, j, i):
        """Max capacity constraints on wind

        Parameters:
        j (int): Index of the generator
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the max capacity constraint
        """
        return self.mwh[j, i] <= self.HorizonWind[j, i]

    def OffshoreWindC(self, j, i):
        """Max capacity constraints on offshorewind

        Parameters:
        j (int): Index of the generator
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the max capacity constraint
        """
        return self.mwh[j, i] <= self.HorizonOffshoreWind[j, i]

    def Nodal_Balance(self, z, i):
        """Nodal balance constraint.

        Parameters:
        z (int): Index of the bus
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the nodal balance constraint
        """
        power_flow = sum(self.Flow[l, i] * self.LinetoBusMap[l, z] for l in self.lines)
        gen = sum(self.mwh[j, i] * self.BustoUnitMap[j, z] for j in self.Generators)
        slack = self.S[z, i]
        must_run = self.HorizonMustrunLimit[z, i]
        return gen + slack + must_run - power_flow == self.HorizonDemand[z, i]

    def Flow_line(self, l, i):
        """Transmission line flow constraint.

        Parameters:
        l (int): Index of the transmission line
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the transmission line flow constraint
        """
        value = sum(self.Theta[z, i] * self.LinetoBusMap[l, z] for z in self.buses)
        return 100 * value == self.Flow[l, i] * self.Reactance[l]

    def Theta_bus(self, i):
        """Bus angle constraint.

        Parameters:
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the bus angle constraint
        """
        return self.Theta['bus_100011', i] == 0

    def FlowUP_line(self, l, i):
        """Transmission line flow constraint (upper limit).

        Parameters:
        l (int): Index of the transmission line
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the transmission line flow constraint (upper limit)
        """
        return self.Flow[l, i] <= self.FlowLim[l]

    def FlowLow_line(self, l, i):
        """Transmission line flow constraint (lower limit).

        Parameters:
        l (int): Index of the transmission line
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the transmission line flow constraint (lower limit)
        """
        return -1 * self.Flow[l, i] <= self.FlowLim[l]

    def DummyFlow1(self, l, i):
        """Dummy flow constraint 1.

        Parameters:
        l (int): Index of the transmission line
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the dummy flow constraint 1
        """
        return self.DummyFlow[l, i] >= self.Flow[l, i]

    def DummyFlow2(self, l, i):
        """Dummy flow constraint 2.

        Parameters:
        l (int): Index of the transmission line
        i (int): Index of the hour

        Returns:
        pyo.Constraint: Constraint object for the dummy flow constraint 2
        """
        return self.DummyFlow[l, i] >= self.Flow[l, i] * -1
