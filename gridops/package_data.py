from __future__ import annotations
import pkg_resources


def get_data_directory() -> str:
    """Return the absolute path to the gridops package ``data/`` directory.

    Returns
    -------
    str
        Absolute path to ``gridops/data/``.
    """
    return pkg_resources.resource_filename("gridops", "data")
