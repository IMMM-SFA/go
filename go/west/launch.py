import logging
import os
from typing import Union

import cloudpickle
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.core import Constraint, Var
from pyomo.opt import SolverStatus, TerminationCondition

from go import configuration
from go.solvers import GoSolver
from go.west.linear import model_west_linear_multi
from go.utilities import write_solver_parameters, write_restart_file, get_restart_file


def west_linear_multi(
    config_file: str,
    solver_name: str = "appsi_highs",
    solver_params: Union[None, dict] = None,
    n_days: int = 365,
    restart_day: Union[None, int] = None,
    save_restart_file: bool = True,
    break_run: bool = False,
    restart_write_frequency: int = 10,
    warmstart: bool = False,
    warmstart_file: Union[str, None] = None,
    record_results_for_warmstart: bool = False,
    **kwargs
):
    """
    This function runs the West Linear Multi Model.

    :param config_file:         The configuration file to use. If None, the default configuration is used.
    :type config_file:          Union[str, None]

    :param solver_name:         The solver to use.  Options are 'appsi_highs', 'gurobi', and 'cplex'
                                Default: 'appsi_highs'
    :type solver_name:          str

    :param solver_params:       Parameter dictionary for the chosen solver to set options for the solver natively.
                                Default None
    :type solver_params:        Union[None, dict]

    :param n_days:              The number of the day in the calendar year to process through.
                                Default 365
    :type n_days:               int

    :param restart_day:         Day of the restart file to use. There must be a restart file for that day available.
                                If None, uses the largest day available among restart files or starts from day 1 if none exist. 
                                Default None
    :type restart_day:          Union[None, int]

    :param save_restart_file:   If True, save a restart file after ever timestep. 
                                Default True
    :type save_restart_file:    bool

    :param break_run:           If True, run will break after one day iteration.  This is only called if the 
                                SOLVER RETRY MODE is initiated.
                                Default False
    :type break_run:            bool

    :param warmstart:           Designates if solver warm start is enabled or not. 
                                Default False
    :type warmstart:            bool

    :param warmstart_file:      Path to the warm start file where values for each decision variable and corresponding indices are stored.
    :type warmstart_file:       str

    :param record_results_for_warmstart:        Saves model results in a .pkl file compatible with warm start procedure afterwards.  
                                                Default False
    :type record_results_for_warmstart:         bool

    :param restart_write_frequency: Write a restart file after this many days.
    :type restart_write_frequency:  int

    """

    logger = logging.getLogger(__name__)

    logger.info("Prepare simulation")

    # read in config file
    config = configuration.generate_config(config_file=config_file, **kwargs)

    # read in input files to data frames
    df_generators = pd.read_csv(config.generator_parameters_file, header=0)
    df_thermal = pd.read_csv(config.thermal_generators_file, header=0)
    df_loss_dict = np.load(config.generator_outage_file, allow_pickle=True).item()
    df_losses = pd.read_csv(config.lost_capacity_file, header=0, index_col=0)

    # extract nuclear
    nucs = df_thermal[df_thermal['Fuel'] == 'NUC (Nuclear)'].copy()

    # instantiate go solver
    opt = GoSolver(
        solver_name=solver_name,
        solver_params=solver_params
    ).go_solver

    # get restart file
    restart_file = get_restart_file(
        dir=config.restart_file_directory,
        day=restart_day,
    )

    # start from scratch if no restart file has been provided or previously created
    if restart_file is None:

        start_day = 1

        # instantiate model
        go_model = model_west_linear_multi()
        instance = go_model.create_instance(config.dat_file)
        instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # Total number of hours in the operating horizon
        n_horizon_hours = instance.HorizonHours

        # Generator for hours 1..n
        horizon_hours_series = range(1,  n_horizon_hours + 1)

        # store outputs
        mwh = []
        on = []
        switch = []
        flow = []
        slack = []
        vlt_angle = []
        duals = []
        charge=[]
        discharge=[]
        SoC=[]

        # storing solver parameters
        solver_parameters = {}

    else:

        logger.info(f"Initializing with the following restart file: {restart_file}")

        with open(restart_file, "rb") as f:
            restart_data = cloudpickle.load(f)

        instance = restart_data["model"]
        mwh = restart_data["mwh"]
        on = restart_data["on"]
        switch = restart_data["switch"]
        flow = restart_data["flow"]
        slack = restart_data["slack"]
        vlt_angle = restart_data["vlt_angle"]
        duals = restart_data["duals"]
        charge = restart_data["charge"]
        discharge = restart_data["discharge"]
        SoC = restart_data["SoC"]
        solver_parameters = restart_data["solver_parameters"]

        # make the start day one day ahead of the last day to solve
        start_day = restart_data["day"] + 1

        # Total number of hours in the operating horizon
        n_horizon_hours = instance.HorizonHours

        # Generator for hours 1..n
        horizon_hours_series = range(1,  n_horizon_hours + 1)

    # n_days is the number of the day in the calendar year to process through.  If the start day
    # -- is greater than the calendar day being processed, the will be no days to process.
    if n_days < start_day:
        msg = (
            f"n_days setting ({n_days}) must be >= to the start day ({start_day}). " + 
            "n_days represents the number of the day in the calendar year to process through."
        )
        raise AssertionError(msg)
    
    # If warm start is enabled, reading the file containing warm start information
    if warmstart:
        logger.info(f"Warm start is active. Loading initial values...")

        with open(warmstart_file, 'rb') as f:
            initial_var_values = cloudpickle.load(f)

    # If record_results_for_warmstart enabled, creating a dictionary to store results for future warm starts
    if record_results_for_warmstart:
        logger.info(f'Recording for warm start is active.')
        outputs_for_warmstart = {i: {} for i in range(1, n_days+1)}

    # max here can be (1, 365)
    for day in range(start_day, n_days + 1):

        logger.info(f"Day {day}: Set up optimization")

        # store the solver parameters for the current day
        solver_parameters[day] = solver_params

        for z in instance.buses:
            # load Demand and Reserve time series data
            for i in horizon_hours_series:
                instance.HorizonDemand[z, i] = instance.SimDemand[z, (day - 1) * 24 + i]

        for z in instance.Hydro:
            # load Hydropower time series data
            instance.HorizonHydro_MAX[z] = instance.SimHydro_MAX[z, day]
            instance.HorizonHydro_MIN[z] = instance.SimHydro_MIN[z, day]
            instance.HorizonHydro_TOTAL[z] = instance.SimHydro_TOTAL[z, day]

        for z in instance.Solar:
            # load Solar time series data
            for i in horizon_hours_series:
                instance.HorizonSolar[z, i] = instance.SimSolar[z, (day - 1) * 24 + i]

        for z in instance.Wind:
            # load Wind time series data
            for i in horizon_hours_series:
                instance.HorizonWind[z, i] = instance.SimWind[z, (day - 1) * 24 + i]

        for z in instance.OffshoreWind:
            # load OffshoreWind time series data
            for i in horizon_hours_series:
                instance.HorizonOffshoreWind[z, i] = instance.SimOffshoreWind[z, (day - 1) * 24 + i]

        for z in instance.Thermal:
            # load fuel prices for thermal generators
            instance.FuelPrice[z] = instance.SimFuelPrice[z, day]

        # Organizing outage data
        # load gen and mustrun capacity time series data
        for z in instance.Outage:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = instance.SimGenLimit[z, (day - 1) * 24 + i]

        for z in instance.buses:
            for i in horizon_hours_series:
                instance.HorizonMustrunLimit[z, i] = instance.SimMustrunLimit[z, (day - 1) * 24 + i]

        # subtract real or historical capacity losses
        for z in instance.Gas_below_50:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_below_50'] / len(df_loss_dict['Gas_below_50']))
        for z in instance.Gas_50_100:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_50_100'] / len(df_loss_dict['Gas_50_100']))
        for z in instance.Gas_100_200:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_100_200'] / len(df_loss_dict['Gas_100_200']))
        for z in instance.Gas_200_300:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_200_300'] / len(df_loss_dict['Gas_200_300']))
        for z in instance.Gas_300_400:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_300_400'] / len(df_loss_dict['Gas_300_400']))
        for z in instance.Gas_400_600:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_400_600'] / len(df_loss_dict['Gas_400_600']))
        for z in instance.Gas_600_800:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_600_800'] / len(df_loss_dict['Gas_600_800']))
        for z in instance.Gas_800_1000:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_800_1000'] / len(df_loss_dict['Gas_800_1000']))
        for z in instance.Gas_ovr_1000:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_ovr_1000'] / len(df_loss_dict['Gas_ovr_1000']))
        for z in instance.Gas_All_n_0_100:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_All_n_0_100'] / len(df_loss_dict['Gas_All_n_0_100']))
        for z in instance.Gas_All_n_100_200:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_All_n_100_200'] / len(df_loss_dict['Gas_All_n_100_200']))
        for z in instance.Gas_All_n_ovr_200:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Gas_All_n_ovr_200'] / len(df_loss_dict['Gas_All_n_ovr_200']))
        for z in instance.Coal_below_50:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_below_50'] / len(df_loss_dict['Coal_below_50']))
        for z in instance.Coal_50_100:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_50_100'] / len(df_loss_dict['Coal_50_100']))
        for z in instance.Coal_100_200:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_100_200'] / len(df_loss_dict['Coal_100_200']))
        for z in instance.Coal_200_300:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_200_300'] / len(df_loss_dict['Coal_200_300']))
        for z in instance.Coal_300_400:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_300_400'] / len(df_loss_dict['Coal_300_400']))
        for z in instance.Coal_400_600:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_400_600'] / len(df_loss_dict['Coal_400_600']))
        for z in instance.Coal_600_800:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_600_800'] / len(df_loss_dict['Coal_600_800']))
        for z in instance.Coal_800_1000:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_800_1000'] / len(df_loss_dict['Coal_800_1000']))
        for z in instance.Coal_ovr_1000:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_ovr_1000'] / len(df_loss_dict['Coal_ovr_1000']))
        for z in instance.Coal_All_n_0_100:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_All_n_0_100'] / len(df_loss_dict['Coal_All_n_0_100']))
        for z in instance.Coal_All_n_100_200:
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_All_n_100_200'] / len(df_loss_dict['Coal_All_n_100_200']))
            for i in horizon_hours_series:
                instance.HorizonGenLimit[z, i] = max(0, instance.HorizonGenLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Coal_All_n_ovr_200'] / len(df_loss_dict['Coal_All_n_ovr_200']))

        for z in instance.buses:
            for i in horizon_hours_series:
                instance.HorizonMustrunLimit[z, i] = max(0, instance.HorizonMustrunLimit[z, i].value - df_losses.loc[
                    (day - 1) * 24 + i, 'Nuclear_ovr_1000'] / len(nucs))

        # Initializing the state of charge of storage facilities as minimum state of charge, if it's the first day of simulation     
        if day == 1:
            for j in instance.Storage:
                instance.SoC[j, 0] = instance.min_SoC[j]
                instance.SoC[j, 0].fixed = True
        else:
            pass

        
        # If warm start is enabled, loading the initial values for each variable
        if warmstart:

            daily_init_vals = initial_var_values[day]

            # # Set initial values for variables
            # for var in instance.component_objects(Var, active=True):
            #     for index in var:
            #         if index is None:
            #             var_name = var.name
            #             if var_name in daily_init_vals.keys():
            #                 var.set_value(daily_init_vals[var_name])
            #         else:
            #             var_name = f"{var.name}[{index}]"
            #             if var_name in daily_init_vals.keys():
            #                 var[index].set_value(daily_init_vals[var_name])

            ########################################################################
            for v in instance.component_objects(Var, active=True):
                a = str(v)
                varobject = getattr(instance, a)

                for index in varobject:
                    if (index[1] > 0) and (index[1] < 25):
                        varobject[(index[0], index[1])].set_value(daily_init_vals[f"{a}[{(index[0], index[1])}]"])
            ########################################################################
                    
        logger.info(f"Day {day}: Start optimization")
        
        result = opt.solve(
            instance,
            tee=True,
            symbolic_solver_labels=True,
            load_solutions=False,
            warmstart=warmstart
        )

        # ensure that the solver termination condition is optimal
        if (result.solver.termination_condition != pyo.TerminationCondition.optimal) or (result.solver.status != SolverStatus.ok):
            logger.error(f"Day {day}: Optimization did not converge to an optimal solution. Termination condition: {result.solver.termination_condition}. Solver status: {result.solver.status}.")
            
            if save_restart_file:

                # write restart file for previous day
                new_restart_file_path = write_restart_file(
                    dir=config.restart_file_directory,
                    day=day - 1,
                    restart_data=restart_data,
                )
                
                if new_restart_file_path is not None:
                    logger.info(f'Day {day - 1}: Restart file written to {new_restart_file_path}.')

            raise RuntimeError(f"Optimization failed on day {day} with termination condition: {result.solver.termination_condition}")

        logger.info(f"Day {day}: Finished optimization")

        logger.info(f"Day {day}: Processing optimization result")
        instance.solutions.load_from(result)

        # check for negative generations, since HiGHs sometimes allows extreme out of bounds generations
        # if found, trigger retry logic
        logger.info(f"Day {day}: Checking for negative generation")
        has_negative_generation = False
        for v in instance.component_objects(Var, active=True):
            a = str(v)
            if a == 'mwh':
                varobject = getattr(instance, str(v))
                for index in varobject:
                    if (index[1] > 0) and (index[1] < 25):
                        if varobject[index].value < -1e-3:
                            has_negative_generation = True
                            logger.error(f"Day {day}: Generator {index[0]} has negative generation {varobject[index].value} at hour {index[1]}.")
        
        if has_negative_generation:
            if save_restart_file:

                # write restart file for previous day
                new_restart_file_path = write_restart_file(
                    dir=config.restart_file_directory,
                    day=day - 1,
                    restart_data=restart_data,
                )
                
                if new_restart_file_path is not None:
                    logger.info(f'Day {day - 1}: Restart file written to {new_restart_file_path}.')

            raise RuntimeError(f"Optimization failed on day {day} due to negative generation.")
                

        for c in instance.component_objects(Constraint, active=True):
            cobject = getattr(instance, str(c))
            if str(c) in ['Node_Constraint']:
                for index in cobject:
                    if int(index[1] > 0 and index[1] < 25):
                        try:
                            duals.append((index[0], index[1] + ((day - 1) * 24), instance.dual[cobject[index]]))
                        except KeyError:
                            duals.append((index[0], index[1] + ((day - 1) * 24), -999))

        for v in instance.component_objects(Var, active=True):
            varobject = getattr(instance, str(v))
            a = str(v)

            if a == 'Theta':
                for index in varobject:
                    if int(index[1] > 0 and index[1] < 25):
                        if index[0] in instance.buses:
                            vlt_angle.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

            if a == 'mwh':
                for index in varobject:

                    gen_name = index[0]
                    gen_heatrate = df_generators[df_generators['name'] == gen_name]['heat_rate'].values[0]

                    if int(index[1] > 0 and index[1] < 25):

                        # fuel_price = instance.FuelPrice[z].value

                        if index[0] in instance.Gas:
                            # marginal_cost = gen_heatrate*fuel_price
                            mwh.append((index[0], 'Gas', index[1] + ((day - 1) * 24), varobject[index].value))
                        elif index[0] in instance.Coal:
                            # marginal_cost = gen_heatrate*fuel_price
                            mwh.append((index[0], 'Coal', index[1] + ((day - 1) * 24), varobject[index].value))
                        elif index[0] in instance.Oil:
                            # marginal_cost = 0
                            mwh.append((index[0], 'Oil', index[1] + ((day - 1) * 24), varobject[index].value))
                        elif index[0] in instance.Hydro:
                            # marginal_cost = 0
                            mwh.append((index[0], 'Hydro', index[1] + ((day - 1) * 24), varobject[index].value))
                        elif index[0] in instance.Solar:
                            # marginal_cost = 0
                            mwh.append((index[0], 'Solar', index[1] + ((day - 1) * 24), varobject[index].value))
                        elif index[0] in instance.Wind:
                            # marginal_cost = 0
                            mwh.append((index[0], 'Wind', index[1] + ((day - 1) * 24), varobject[index].value))
                        elif index[0] in instance.OffshoreWind:
                            # marginal_cost = 0
                            mwh.append((index[0], 'OffshoreWind', index[1] + ((day - 1) * 24), varobject[index].value))
                        elif index[0] in instance.Biomass:
                            # marginal_cost = gen_heatrate*fuel_price
                            mwh.append((index[0], 'Biomass', index[1] + ((day - 1) * 24), varobject[index].value))
                        elif index[0] in instance.Geothermal:
                            # marginal_cost = gen_heatrate*fuel_price
                            mwh.append((index[0], 'Geothermal', index[1] + ((day - 1) * 24), varobject[index].value))

            if a == 'on':
                for index in varobject:
                    if int(index[1] > 0 and index[1] < 25):
                        on.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

            if a == 'switch':
                for index in varobject:
                    if int(index[1] > 0 and index[1] < 25):
                        switch.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

            if a == 'S':
                for index in varobject:
                    if index[0] in instance.buses:
                        slack.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

            if a == 'Flow':
                for index in varobject:
                    if int(index[1] > 0 and index[1] < 25):
                        flow.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

            if a == 'SoC':
                for index in varobject:
                    if int(index[1] > 0 and index[1] < 25):
                        SoC.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

            if a == 'Charge':
                for index in varobject:
                    if int(index[1] > 0 and index[1] < 25):
                        charge.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))

            if a == 'Discharge':
                for index in varobject:
                    if int(index[1] > 0 and index[1] < 25):
                        discharge.append((index[0], index[1] + ((day - 1) * 24), varobject[index].value))  

            # Passing last hour of generation for each generator to the first hour of next day
            for j in instance.Dispatchable:
                if instance.mwh[j, 24].value <= 0 and instance.mwh[j, 24].value >= -0.0001:
                    newval_1 = 0
                else:
                    newval_1 = instance.mwh[j, 24].value
                instance.mwh[j, 0] = newval_1
                instance.mwh[j, 0].fixed = True

            # Passing last hour of state of charge for each storage facility to the first hour of next day
            for j in instance.Storage:
                newval_2 = instance.SoC[j, 24].value
                instance.SoC[j, 0] = newval_2
                instance.SoC[j, 0].fixed = True

        if save_restart_file:

            restart_data = {
                "model": instance,
                "mwh": mwh,
                "on": on,
                "switch": switch,
                "flow": flow,
                "charge": charge,
                "discharge": discharge,
                "SoC": SoC,
                "slack": slack,
                "vlt_angle": vlt_angle,
                "duals": duals,
                "day": day,
                "solver_parameters": solver_parameters
            }

            if day % restart_write_frequency == 0:
                # write restart file for day
                new_restart_file_path = write_restart_file(
                    dir=config.restart_file_directory,
                    day=day,
                    restart_data=restart_data,
                )
                if new_restart_file_path is not None:
                    logger.info(f'Day {day}: Restart file written to {new_restart_file_path}.')

        #Recording results for warm starting afterwards
        if record_results_for_warmstart:
            logger.info(f'Saving results from day {day} for future warm starts...')

            for v in instance.component_objects(Var, active=True):
                a = str(v)
                varobject = getattr(instance, a)

                for index in varobject:
                    if (index[1] > 0) and (index[1] < 25):
                        outputs_for_warmstart[day][f"{a}[{(index[0], index[1])}]"] = varobject[(index[0], index[1])].value

    
        logger.info(f'Day {day} completed.')

        # if only one iteration is desired break the loop
        # -- this is only set to True when the model cannot solve and 
        # -- the SOLVER RETRY MODE is activated.
        if break_run:
            break

    vlt_angle_pd = pd.DataFrame(vlt_angle, columns=('Node', 'Time', 'Value'))
    mwh_pd = pd.DataFrame(mwh, columns=('Generator', 'Type', 'Time', 'Value'))
    slack_pd = pd.DataFrame(slack, columns=('Node','Time','Value'))
    flow_pd = pd.DataFrame(flow, columns=('Line', 'Time', 'Value'))
    duals_pd = pd.DataFrame(duals, columns=('Bus', 'Time', 'Value'))
    SoC_pd = pd.DataFrame(SoC, columns=('Storage','Time','Value'))
    discharge_pd = pd.DataFrame(discharge, columns=('Storage','Time','Value'))
    charge_pd = pd.DataFrame(charge, columns=('Storage','Time','Value'))

    # to save outputs
    vlt_angle_pd.to_parquet(config.vlt_angle_file, index=False)
    mwh_pd.to_parquet(config.mwh_file, index=False)
    slack_pd.to_parquet(config.slack_file, index=False)
    flow_pd.to_parquet(config.flow_file, index=False)
    duals_pd.to_parquet(config.duals_file, index=False)
    SoC_pd.to_parquet(config.SoC_file, index=False)
    discharge_pd.to_parquet(config.discharge_file, index=False)
    charge_pd.to_parquet(config.charge_file, index=False)

    # Saving warmstart data
    if record_results_for_warmstart:
        with open('warm_start_data.pkl', 'wb') as f:
            cloudpickle.dump(outputs_for_warmstart, f)

    # write out the solver parameters as a JSON file
    write_solver_parameters(
        solver_parameter_dictionary=solver_parameters,
        solver_parameter_file=os.path.join(config.restart_file_directory, "solver_parameters.json")
    )

    return day
