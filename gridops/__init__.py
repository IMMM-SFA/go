"""
gridops - An open-source production cost model (PCM) for the U.S. interconnections and grid regions.
"""

import logging
import pyomo.environ

from .configuration import *
from .model import Model
from .package_data import *
from .solvers import GoSolver
from .utilities import *

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Package-level logger configuration
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)
