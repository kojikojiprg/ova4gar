import importlib
import sys

# hrnet and unitrack have "uitls" module.
# Avoid import error.
sys.path.append("submodules/higher-hrnet/lib/")
from . import hrnet

sys.path.remove("submodules/higher-hrnet/lib/")
sys.path.append("submodules/unitrack/")
import utils

importlib.reload(utils)
from . import unitrack

from . import dataset, extracter
