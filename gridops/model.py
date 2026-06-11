from __future__ import annotations
import itertools
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
from gridops import configuration
from gridops.utilities import get_prior_restart_file_day


# ---------------------------------------------------------------------------
# Default retry option values for HiGHS solver
# ---------------------------------------------------------------------------

#: Default search grid for HiGHS retry logic.  Each key maps to a list of
#: values to try; the retry method builds a Cartesian product per algorithm.
HIGHS_RETRY_DEFAULTS: Dict[str, Any] = {
    # Algorithms to sweep
    "solver_list": ["simplex", "ipm"],
    # Tolerances
    "dual_feasibility_tolerance":   [1e-7, 1e-6],
    "primal_feasibility_tolerance": [1e-7, 1e-6],
    "ipm_optimality_tolerance":     [1e-8, 1e-7, 1e-6],
    # Simplex-specific
    "simplex_strategy":                      [0],
    "simplex_scale_strategy":                [1],
    "simplex_dual_edge_weight_strategy":     [-1],
    "simplex_primal_edge_weight_strategy":   [-1]
}


class Model:
    """gridops production cost model.

    Parameters
    ----------
    region:
        Geographic region. Supported: ``"west"``.  Placeholders: ``"east"``,
        ``"ercot"``.
    problem:
        Problem type. Supported: ``"linear"`` (LP - economic dispatch).
        Placeholder: ``"mixed_integer"`` (MILP - unit commitment ).
    solver_name:
        Solver interface name.  One of ``"appsi_highs"`` (open-source), 
        ``"appsi_gurobi"`` (commercial) or ``"gurobi"`` (commercial).  Default ``"appsi_highs"``.
    solver_params:
        Optional dict of solver option name → value pairs.  No defaults are
        imposed by gridops; pass every required option explicitly.
    """

    #: Maximum random seed value accepted by HiGHS.
    MAX_RANDOM_SEED_HIGHS = 2147483647
    MAX_RANDOM_SEED_GUROBI = 2000000000

    def __init__(
        self,
        region: str,
        problem: str,
        solver_name: str = "appsi_highs",
        solver_params: Optional[Dict[str, Any]] = None,
        warmstart: bool = False,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        # ------------------------------------------------------------------
        # Select the launch function for the requested configuration
        # ------------------------------------------------------------------
        if region == "west" and problem == "linear":
            from gridops.west.launch import west_linear
            self._launch_fn = west_linear
            self.logger.info(f"Configuration: {region}_{problem}")

        elif region in ("east", "ercot"):
            raise NotImplementedError(
                f"Region '{region}' is not yet implemented. "
                "Only 'west' is currently available."
            )
        elif problem == "mixed_integer":
            raise NotImplementedError(
                "Mixed integer (unit commitment) problem type is not yet "
                "implemented.  Use problem='linear' for economic dispatch."
            )
        else:
            raise AssertionError(
                f"Configuration '{region}_{problem}' is not supported."
            )

        # ------------------------------------------------------------------
        # Validate solver
        # ------------------------------------------------------------------
        valid_solvers = ("appsi_highs", "appsi_gurobi", "gurobi")
        if solver_name not in valid_solvers:
            raise AssertionError(
                f"Solver '{solver_name}' is not supported. "
                f"Choose from: {valid_solvers}"
            )
        self.solver_name   = solver_name
        self.solver_params = solver_params if solver_params is not None else {}
        self.warmstart     = warmstart

    # ======================================================================
    # Run function with retry logic
    # ======================================================================

    def run(
        self,
        config_file: Optional[str] = None,
        restart_day: Optional[int] = None,
        n_days: int = 365,
        horizon_hours: int = 24,
        warmstart: Optional[bool] = None,
        allow_retry: bool = True,
        max_retries: int = 30,
        retry_n_seeds: int = 1,
        retry_options: Optional[Dict[str, Any]] = None,
        retry_backoff: bool = True,
        restart_write_frequency: int = 30,
        fresh_start: bool = False,
        clear_restart: bool = True,
        **kwargs,
    ) -> None:
        """Run the production-cost model.

        Parameters
        ----------
        config_file:
            Path to the YAML configuration file.
        restart_day:
            Day whose restart file to use as starting point. ``None`` =
            resume from the latest available restart file (or start fresh).
        n_days:
            Total number of days to simulate. Default 365.
        horizon_hours:
            Hours for optimization horizon. Default 24.
        allow_retry:
            If ``True`` and a horizon fails, automatically attempt the retry
            logic before raising an error. Default ``True``.
        max_retries:
            Maximum number of retry attempts before aborting. Applies to
            both HiGHS (algorithm and tolerance sweep) and Gurobi (random-seed
            sweep). Default 30.
        retry_n_seeds:
            Number of random seeds to try per solver algorithm in retry mode.
        retry_options:
            Dict overriding specific keys in :data:`HIGHS_RETRY_DEFAULTS`.
            For example ``{"solver_list": ["simplex"], "simplex_strategy": [0, 1]}``
            to only sweep simplex with two strategy values. ``None`` = use
            all defaults.
        retry_backoff:
            If ``True`` and all retry trials fail, rewind to the prior restart
            file and retry from there. Default ``True``.
        restart_write_frequency:
            Write a restart file every this many solved horizons. Default 30.
        fresh_start:
            If ``True``, delete all existing restart files and start the
            simulation from scratch. Default ``False``.
        clear_restart:
            If ``True``, delete all restart files after all *n_days* have been
            solved successfully. Files are kept if the simulation fails
            before completion. Default ``True``.
        **kwargs:
            Forwarded to :func:`~gridops.configuration.generate_config`.
        """
        # Merge user overrides on top of defaults
        _retry_opts = dict(HIGHS_RETRY_DEFAULTS)
        if retry_options is not None:
            _retry_opts.update(retry_options)

        # Resolve warmstart: run-level overrides instance-level
        _warmstart = warmstart if warmstart is not None else self.warmstart

        # Resolve effective solver name/params based on warmstart
        _solver_name, _solver_params = self._resolve_warmstart(
            _warmstart,
        )

        try:
            self._launch_fn(
                config_file=config_file,
                solver_name=_solver_name,
                solver_params=_solver_params,
                warmstart=_warmstart,
                n_days=n_days,
                horizon_hours=horizon_hours,
                restart_day=restart_day,
                restart_write_frequency=restart_write_frequency,
                fresh_start=fresh_start,
                clear_restart=clear_restart,
                **kwargs,
            )
            self.logger.info("All horizons completed successfully.")

        except Exception as exc:
            if allow_retry:
                self.retry(
                    config_file=config_file,
                    solver_name=_solver_name,
                    n_days=n_days,
                    horizon_hours=horizon_hours,
                    max_retries=max_retries,
                    n_seeds=retry_n_seeds,
                    retry_options=_retry_opts,
                    retry_backoff=retry_backoff,
                    restart_write_frequency=restart_write_frequency,
                    **kwargs,
                )
            else:
                raise RuntimeError(
                    "Simulation failed and retry is disabled. Aborting...")

    # ======================================================================
    # Resolve warmstart settings
    # ======================================================================

    def _resolve_warmstart(
        self,
        warmstart: bool,
    ) -> tuple:
        """Return (solver_name, solver_params) adjusted for *warmstart*.

        Applies the warm-start policy documented in Model.__init__:

        * **appsi_highs, warmstart=False** (default):
          Uses IPM solver with crossover=on; clearSolver is handled in
          launch.py.  User-supplied ``solver`` and ``run_crossover`` are
          respected if present.
        * **appsi_highs, warmstart=True**:
          Forces solver=simplex, run_crossover=off so that the APPSI
          persistent interface reuses the previous horizon's basis.
        * **appsi_gurobi, warmstart=False**:
          Sets LPWarmStart=0 (disables warm start).
        * **appsi_gurobi, warmstart=True**:
          Sets LPWarmStart=1 (without presolve) unless user requested 2 (with presolve).
          Overrides user LPWarmStart=0 with a warning.
        * **gurobi, warmstart=True**:
          Switches to appsi_gurobi with LPWarmStart=1 and warns.
        * **gurobi, warmstart=False**:
          No changes (interface rebuilds each horizon).
        """
        solver_name = self.solver_name
        solver_params = dict(self.solver_params)

        if solver_name == "appsi_highs":
            if warmstart:
                solver_params["solver"] = "simplex"
                solver_params["run_crossover"] = "off"
                self.logger.info(
                    "Warm start enabled: HiGHS solver set to simplex, "
                    "crossover disabled."
                )
            else:
                # Default: IPM + crossover for best cold-start performance
                solver_params.setdefault("solver", "ipm")
                solver_params.setdefault("run_crossover", "on")

            # Simplex does not need crossover; always disable it.
            if solver_params.get("solver") == "simplex":
                solver_params["run_crossover"] = "off"

        elif solver_name == "appsi_gurobi":
            if warmstart:
                user_ws = solver_params.get("LPWarmStart")
                if user_ws == 0:
                    self.logger.warning(
                        "Solver parameter LPWarmStart is set to 1 since "
                        "warmstart is True. Please set LPWarmStart to "
                        "either 1 or 2 when warmstart is True."
                    )
                    solver_params["LPWarmStart"] = 1
                elif user_ws == 2:
                    pass  # Keep user's choice
                else:
                    solver_params["LPWarmStart"] = 1
            else:
                solver_params["LPWarmStart"] = 0 # Disable warm start when not requested
                solver_params.setdefault("PreDual", 1) # Enable dual presolve for better cold-start performance

        elif solver_name == "gurobi":
            if warmstart:
                solver_name = "appsi_gurobi"
                self.logger.warning(
                    "Solver is set to appsi_gurobi since gurobi solver interface "
                    "does not support warm starts."
                )
                user_ws = solver_params.get("LPWarmStart")
                if user_ws == 2:
                    pass  # Keep user's choice
                else:
                    self.logger.warning(
                        "Solver parameter LPWarmStart is set to 1 since "
                        "warmstart is True. Please set LPWarmStart to "
                        "either 1 or 2 when warmstart is True."
                    )
                    solver_params["LPWarmStart"] = 1
            else:
                solver_params.setdefault("PreDual", 1) # Enable dual presolve for better cold-start performance

        return solver_name, solver_params

    # ======================================================================
    # Retry logic
    # ======================================================================

    def retry(
        self,
        config_file: Optional[str] = None,
        solver_name: str = "appsi_highs",
        n_days: int = 365,
        horizon_hours: int = 24,
        max_retries: int = 30,
        n_seeds: int = 1,
        retry_options: Optional[Dict[str, Any]] = None,
        retry_backoff: bool = True,
        restart_write_frequency: int = 30,
        **kwargs,
    ) -> None:
        """Try alternative solver configurations after a failed horizon.

        For **HiGHS**, builds a Cartesian product of algorithm choices,
        tolerance settings, and random seeds, then caps the total number of
        trials at *max_retries*.

        For **Gurobi**, only varies the random seed (Gurobi's algorithm
        selection is automatic). Generates up to *max_retries* random seeds.

        After a successful retry the model reverts to the original solver
        settings and continues from the next horizon.

        If all retry trials fail and *retry_backoff* is ``True``, the model
        rewinds to the prior available restart file and tries again.

        Parameters
        ----------
        config_file:
            Path to the YAML configuration file.
        solver_name:
            Solver interface name.
        n_days:
            Total number of days to simulate. Default 365.
        horizon_hours:
            Hours for optimization horizon. Default 24.
        max_retries:
            Maximum number of retry attempts before aborting. Applies to
            both HiGHS (algorithm and tolerance sweep) and Gurobi (random-seed
            sweep). Default 30.
        n_seeds:
            Number of random seeds to try per algorithm in the HiGHS
            sweep. The Cartesian product is still capped at *max_retries*.
        retry_options:
            Dict with retry search grid values. Uses
            :data:`HIGHS_RETRY_DEFAULTS` for any missing keys.
        retry_backoff:
            If ``True`` and all retry trials fail, rewind to the prior restart
            file and retry from there. Default ``True``.
        restart_write_frequency:
            Write a restart file every this many solved horizons. Default 30.
        **kwargs:
            Forwarded to the launch function.
        """
        # Merge defaults
        opts = dict(HIGHS_RETRY_DEFAULTS)
        if retry_options is not None:
            opts.update(retry_options)

        solver_list                              = opts["solver_list"]
        dual_feasibility_tolerance_list          = opts["dual_feasibility_tolerance"]
        primal_feasibility_tolerance_list        = opts["primal_feasibility_tolerance"]
        ipm_optimality_tolerance_list            = opts["ipm_optimality_tolerance"]
        simplex_strategy_list                    = opts["simplex_strategy"]
        simplex_scale_strategy_list              = opts["simplex_scale_strategy"]
        simplex_dual_edge_weight_strategy_list   = opts["simplex_dual_edge_weight_strategy"]
        simplex_primal_edge_weight_strategy_list = opts["simplex_primal_edge_weight_strategy"]
        
        self.logger.warning("[RETRY MODE] Initiated.")

        # Build the trial list depending on solver type
        trial_list: List[Dict[str, Any]] = []

        if solver_name in ("appsi_gurobi", "gurobi"):
            # For Gurobi, only vary the random seed
            seeds = [
                int(np.random.randint(0, Model.MAX_RANDOM_SEED_GUROBI + 1))
                for _ in range(max_retries)
            ]
            for seed in seeds:
                trial_list.append({"Seed": seed})

        else:
            # For HiGHS, sweep over algorithm and tolerance combinations
            seeds = [
                int(np.random.randint(0, Model.MAX_RANDOM_SEED_HIGHS + 1))
                for _ in range(n_seeds)
            ]
            for algo in solver_list:
                if algo == "simplex":
                    combos = list(itertools.product(
                        seeds,
                        simplex_strategy_list,
                        simplex_scale_strategy_list,
                        simplex_dual_edge_weight_strategy_list,
                        simplex_primal_edge_weight_strategy_list,
                        dual_feasibility_tolerance_list,
                        primal_feasibility_tolerance_list,
                    ))
                    for c in combos:
                        trial_list.append({
                            "solver":                          algo,
                            "random_seed":                     c[0],
                            "simplex_strategy":                c[1],
                            "simplex_scale_strategy":          c[2],
                            "simplex_dual_edge_weight_strategy":   c[3],
                            "simplex_primal_edge_weight_strategy": c[4],
                            "dual_feasibility_tolerance":      c[5],
                            "primal_feasibility_tolerance":    c[6],
                        })
                
                elif algo == "ipm":
                    combos = list(itertools.product(
                        seeds,
                        dual_feasibility_tolerance_list,
                        primal_feasibility_tolerance_list,
                        ipm_optimality_tolerance_list,
                    ))
                    for c in combos:
                        trial_list.append({
                            "solver":                       algo,
                            "random_seed":                  c[0],
                            "dual_feasibility_tolerance":   c[1],
                            "primal_feasibility_tolerance": c[2],
                            "ipm_optimality_tolerance":     c[3],
                        })
                
                else:
                    self.logger.warning(
                        f"[RETRY MODE] Unknown algorithm '{algo}'; skipping..."
                    )

        # Cap the trial list at max_retries
        if len(trial_list) > max_retries:
            self.logger.info(
                f"[RETRY MODE] {len(trial_list)} candidates generated; capping at max_retries={max_retries}.",
            )
            trial_list = trial_list[:max_retries]

        self.logger.info(f"[RETRY MODE] {len(trial_list)} trial configurations to attempt.")

        success = False
        success_day: Optional[int] = None
        last_exc: Optional[Exception] = None

        for trial_idx, modification in enumerate(trial_list, 1):
            local_params = dict(self.solver_params)
            local_params.update(modification)

            self.logger.info(
                f"[RETRY MODE] Trial {trial_idx} / {len(trial_list)}: {modification}"
            )

            try:
                success_day = self._launch_fn(
                    config_file=config_file,
                    solver_name=solver_name,
                    solver_params=local_params,
                    n_days=n_days,
                    horizon_hours=horizon_hours,
                    restart_day=None,
                    break_run=True,
                    restart_write_frequency=restart_write_frequency,
                    **kwargs,
                )
                success = True
                break
            except Exception as exc:
                last_exc = exc

        if success:
            self.logger.info(
                "[RETRY MODE] Solution found. Reverting to original solver settings."
            )
            if success_day is not None and success_day < n_days:
                self.run(
                    config_file=config_file,
                    n_days=n_days,
                    horizon_hours=horizon_hours,
                    allow_retry=True,
                    max_retries=max_retries,
                    retry_n_seeds=n_seeds,
                    retry_options=opts,
                    retry_backoff=False,
                    restart_write_frequency=restart_write_frequency,
                    **kwargs,
                )
            else:
                self.logger.info("[RETRY MODE] All days completed.")
        else:
            if retry_backoff:
                config = configuration.generate_config(
                    config_file=config_file, **kwargs
                )
                prior_day = get_prior_restart_file_day(
                    config.restart_file_directory
                )
                if prior_day is None:
                    self.logger.error("[RETRY MODE] No prior restart file. Aborting...")
                    raise last_exc
                else:
                    # Vary seed and back off to previous restart
                    self.solver_params["random_seed"] = int(
                        np.random.randint(0, Model.MAX_RANDOM_SEED_HIGHS + 1)
                    )
                    self.run(
                        config_file=config_file,
                        restart_day=prior_day,
                        n_days=n_days,
                        horizon_hours=horizon_hours,
                        max_retries=max_retries,
                        retry_n_seeds=n_seeds,
                        retry_options=opts,
                        retry_backoff=False,
                        restart_write_frequency=restart_write_frequency,
                        **kwargs,
                    )
            else:
                self.logger.error("[RETRY MODE] All trials exhausted. Aborting...")
                raise last_exc
