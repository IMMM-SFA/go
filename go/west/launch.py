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
from go.utilities import write_solver_parameters


def west_linear_multi(
    config_file: str,
    solver_name: str = "appsi_highs",
    solver_params: Union[None, dict] = None,
    n_days: int = 365,
    restart_file: Union[None, str] = None,
    save_restart_file: bool = True,
    break_run: bool = False,
    reset_restart_file: bool = False,
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

    :param restart_file:        Full path to cloudpickled restart file.  If no file is provided, the model will search for one
                                in the restart_file_directory specified by the user in the configuration file.
                                Default None
    :type restart_file:         Union[None, str]

    :param save_restart_file:   If True, save a restart file after ever timestep. 
                                Default True
    :type save_restart_file:    bool

    :param break_run:           If True, run will break after one day iteration.  This is only called if the 
                                SOLVER RETRY MODE is initiated.
                                Default False
    :type break_run:            bool

    :param reset_restart_file:  If True, any existing restart file will be deleted.  This is usualy used if the user wants
                                to start from day 1 but has already done a few runs and thus generated a restart file.
                                Default False
    :type reset_restart_file:   bool

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

    # Where the new restart file will saved to; this will not overwrite the restart file
    # -- passed in by the user unless they are the same path.  This gives the user the 
    # -- ability to pass in a restart file from another model if needed.
    local_restart_file = os.path.join(
        config.restart_file_directory,
        f"model_restart_file.pkl"
    )

    if reset_restart_file:
        try:
            os.remove(local_restart_file)
            logger.info(f"Deleted existing restart file {local_restart_file}")
        except PermissionError:
            logger.error(f"Permission denied: Unable to delete the restart file {local_restart_file}")
            raise PermissionError(f"Permission denied: Unable to delete the restart file {local_restart_file}")

    # if a restart file is provided or exists then use it
    if restart_file is None and os.path.exists(local_restart_file):
        restart_file = local_restart_file
    elif restart_file is None and os.path.exists(local_restart_file) is False:
        restart_file = None 
    else:
        restart_file = restart_file

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

    # max here can be (1, 365)
    restart_data = None
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

        logger.info(f"Day {day}: Start optimization")
        
        result = opt.solve(
            instance,
            tee=True,
            symbolic_solver_labels=True,
            load_solutions=False
        )

        # ensure that the solver termination condition is optimal
        if (result.solver.termination_condition != pyo.TerminationCondition.optimal) or (result.solver.status != SolverStatus.ok):
            logger.error(f"Day {day}: Optimization did not converge to an optimal solution. Termination condition: {result.solver.termination_condition}. Solver status: {result.solver.status}.")
            
            if save_restart_file and (restart_data is not None):

                logger.info(f"Day {day}: Writing restart file")

                with open(local_restart_file, "wb") as f:
                    cloudpickle.dump(restart_data, f)

                logger.info(f'Day {restart_data["day"]}: Restart file written to {local_restart_file}.')

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
            if save_restart_file and (restart_data is not None):

                logger.info(f"Day {day}: Writing restart file")

                with open(local_restart_file, "wb") as f:
                    cloudpickle.dump(restart_data, f)

                logger.info(f'Day {restart_data["day"]}: Restart file written to {local_restart_file}.')

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

        logger.info(f'Day {day} completed.')

        # if only one iteration is desired break the loop
        # -- this is only set to True when the model cannot solve and 
        # -- the SOLVER RETRY MODE is activated.
        if break_run:
            if save_restart_file and (restart_data is not None):

                logger.info(f"Day {day}: Writing restart file")

                with open(local_restart_file, "wb") as f:
                    cloudpickle.dump(restart_data, f)

                logger.info(f'Day {restart_data["day"]}: Restart file written to {local_restart_file}.')
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

    # write out the solver parameters as a JSON file
    write_solver_parameters(
        solver_parameter_dictionary=solver_parameters,
        solver_parameter_file=os.path.join(config.restart_file_directory, "solver_parameters.json")
    )

    return day
