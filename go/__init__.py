import logging

import pyomo.environ

from .west import *
from .configuration import *
from .preprocessor import *
from .model import Model
from .utilities import *


__version__ = "0.1.0"

# instantiate logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# define handler and formatter
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# add formatter to handler
handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(handler)