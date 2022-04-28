import importlib
import sys

sys.path.append("submodules")

# Avoid import error.
sys.path.append("submodules/higher_hrnet/lib/")
from .api import higher_hrnet

sys.path.remove("submodules/higher_hrnet/lib/")
sys.path.append("submodules/hrnet/lib/")
from .api import hrnet

sys.path.remove("submodules/hrnet/lib/")
sys.path.append("submodules/unitrack/")
import utils
importlib.reload(utils)
from .api import unitrack

from . import dataset, extracter
