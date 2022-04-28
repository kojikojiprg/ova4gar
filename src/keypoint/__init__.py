import importlib
import sys

# hrnet and unitrack have "uitls" module.
# Avoid import error.
sys.path.append("submodules/higher-hrnet/lib/")
from .api import higher_hrnet

sys.path.remove("submodules/higher-hrnet/lib/")
sys.path.append("submodules/unitrack/")
import utils

importlib.reload(utils)
from . import dataset, extracter
from .api import unitrack
