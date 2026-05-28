# gridops

`gridops` is an open-source and customizable grid operations modeling framework for the three major U.S. interconnections (Western, Eastern, ERCOT). Production cost modeling in `gridops` utilizes linear DC Optimal Power Flow (DC-OPF) approximation to optimize hourly unit commitment (UC) and economic dispatch (ED) processes over a user-configurable rolling horizon.

## Purpose
`gridops` was created to:
  - Address weather and water dynamics and associated vulnerabilities in U.S. bulk power systems,
  - Link with a wide range of hydrometeorological datasets and energy system/machine learning models to explore future uncertainty in grid operations,
  - Analyze current and future grid stress and reliability trends under a wide range of weather conditions, generation mixes, and demand growth scenarios.
  - Create simple and user-tailored representations of the U.S. interconnections to find a balance between model accuracy and run time to answer research questions of interest.

## Installation

```bash
pip install gridops
```

## Quickstart

```python
import gridops as go

# Instantiate the model
model = go.Model(
    region="west",                     # U.S. Western Interconnection
    problem="linear",                  # Linear programming (economic dispatch only)
    solver_name="appsi_highs",         # Using open-source HiGHS solver
    solver_params=None,                # Optional – using default solver parameters
)

# Run for 7 days with a 24-hour rolling horizon
model.run(
    config_file="my_config_file.yml",  # Path to the model configuration file
    n_days=7,                          # Running for 7 days
    horizon_hours=24,                  # Optional – using default 24-hour horizon
)
```

## Configuration file

`gridops` needs a YAML file that maps each input CSV and NPY files to their file path and each
output Parquet to its destination. Check out below for a minimal example (see
`gridops/data/config.yml` for a full template):

```yaml
generator_parameters_file: Input/data_genparams.csv
line_parameters_file: Input/line_param.csv
# … all other input entries …
flow_file: Output/flow.parquet
mwh_file: Output/mwh.parquet
# … all other output entries …
restart_file_directory: Restart/
```

## Supported solvers

| Solver | Interface | Notes |
|--------|-----------|-------|
| HiGHS  | `appsi_highs`  | Open-source; install separately with `pip install highspy` |
| Gurobi | `appsi_gurobi` or `gurobi` | Commercial; requires a valid Gurobi license |

Users can pass any desired solver parameters via the `solver_params`:

```python
model = go.Model(
    solver_name="appsi_highs",
    solver_params={"presolve": "on", "solver": "ipm", "time_limit": 3600},
    ...
)
```

## License

BSD-2-Clause – See [LICENSE](LICENSE).

## Disclaimer

See [DISCLAIMER](DISCLAIMER).
