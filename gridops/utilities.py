from __future__ import annotations
import json
import logging
from glob import glob
from pathlib import Path
from typing import Any, Optional
import cloudpickle


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Solver-parameter helpers
# ---------------------------------------------------------------------------

def load_solver_parameters(solver_parameter_file: str) -> dict:
    """Load solver parameters from a JSON file.

    The JSON file is expected to have integer-string keys (produced by
    :func:`write_solver_parameters`); these are converted back to ``int``.

    Parameters
    ----------
    solver_parameter_file:
        Path to the JSON file.

    Returns
    -------
    dict
        ``{horizon_index: solver_params_dict, …}`` with integer keys.
    """
    with open(solver_parameter_file, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    # JSON keys are always strings; convert back to int for consistency.
    return {int(k): v for k, v in raw.items()}


def write_solver_parameters(
    solver_parameter_dictionary: dict,
    solver_parameter_file: str,
    indent: int = 4,
) -> None:
    """Write solver parameters to a JSON file.

    Parameters
    ----------
    solver_parameter_dictionary:
        Mapping of horizon index → solver options dict.
    solver_parameter_file:
        Destination file path.
    indent:
        JSON indentation level (default 4).
    """
    with open(solver_parameter_file, "w", encoding="utf-8") as fh:
        json.dump(solver_parameter_dictionary, fh, indent=indent)


# ---------------------------------------------------------------------------
# Restart-file helpers
# ---------------------------------------------------------------------------

def write_restart_file(
    dir: str,
    day: int,
    restart_data: Any,
) -> Optional[Path]:
    """Serialize *restart_data* to a versioned cloudpickle file.

    The file name encodes the day number and an auto-incremented version so
    that multiple restarts for the same day do not overwrite each other.
    Format: ``model_restart_file_day{DDD}_v{VVV}.pkl``

    Parameters
    ----------
    dir:
        Directory where the restart file will be written (must exist).
    day:
        Day index to encode in the file name.  Pass ``0`` to skip writing
        (returns ``None`` immediately).
    restart_data:
        Arbitrary Python object to serialize (typically a dict that includes
        the Pyomo model instance and accumulated result lists).

    Returns
    -------
    Path or None
        Path to the new file, or ``None`` if *day* is 0.

    Raises
    ------
    Exception
        If more than 101 versioned files already exist for *day*.
    """
    if day == 0:
        return None

    version = 0
    fp = Path(dir) / f"model_restart_file_day{str(day).zfill(3)}_v{str(version).zfill(3)}.pkl"

    # Increment version until we find a free filename (cap at 101).
    while fp.is_file() and version < 101:
        version += 1
        fp = Path(dir) / f"model_restart_file_day{str(day).zfill(3)}_v{str(version).zfill(3)}.pkl"

    if fp.is_file():
        raise Exception(
            "Too many restart files for the same day. Please clean up!"
        )

    with open(fp, "wb") as fh:
        cloudpickle.dump(restart_data, fh)

    logger.info(f"Restart file written: {fp}")
    return fp


def get_restart_file(
    dir: str,
    day: Optional[int] = None,
) -> Optional[str]:
    """Return the path to an existing restart file.

    Parameters
    ----------
    dir:
        Directory to search for restart files.
    day:
        If ``None``, return the latest restart file found
        (i.e., the most recently written one).  If an integer is provided,
        return the latest version for that specific day.

    Returns
    -------
    str or None
        File path string, or ``None`` if no restart files exist when *day* is
        ``None``.

    Raises
    ------
    Exception
        If *day* is specified but no restart file for that day can be found.
    """
    available = sorted(glob(str(Path(dir) / "model_restart_file_day*.pkl")))

    if day is None:
        # No restart files found → start fresh.
        if not available:
            return None
        return available[-1]

    # Find files matching the requested day.
    prefix = str(Path(dir) / f"model_restart_file_day{str(day).zfill(3)}")
    day_files = [fp for fp in available if fp.startswith(prefix)]

    if day_files:
        return day_files[-1]

    raise Exception(f"No restart file found for day {day}.")


def get_prior_restart_file_day(dir: str) -> Optional[int]:
    """Return the day number of the *second-to-last* restart file group.

    This is used by the retry back-off logic to rewind to the day before the
    one that failed.

    Parameters
    ----------
    dir:
        Directory containing restart files.

    Returns
    -------
    int or None
        Day number of the prior restart group, or ``None`` if there are fewer
        than two distinct day groups available.
    """
    available = sorted(glob(str(Path(dir) / "model_restart_file_day*.pkl")))

    if not available:
        return None

    # Identify the latest day directory prefix.
    latest_day_prefix = "_".join(available[-1].split("_")[:-1])

    # Files that belong to a *different* (earlier) day.
    prior_files = [fp for fp in available if latest_day_prefix not in fp]

    if prior_files:
        prior_file = prior_files[-1]
        # Extract day number from the file name.
        day_str = prior_file.split("_day")[-1].split("_v")[0]
        return int(day_str)

    return None


def clear_restart_files(dir: str) -> int:
    """Delete all restart pickle files in *dir*.

    Parameters
    ----------
    dir:
        Directory containing restart files.

    Returns
    -------
    int
        Number of files deleted.
    """
    files = glob(str(Path(dir) / "model_restart_file_day*.pkl"))
    for fp in files:
        Path(fp).unlink()
    if files:
        logger.info(f"Cleared {len(files)} restart file(s) from {dir}.")
    return len(files)

