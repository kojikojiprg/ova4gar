import importlib
import sys

# all submodules have "uitls" module.
# Avoid import error.
sys.path.append("submodules/higher-hrnet/lib/")
from .api import higher_hrnet

sys.path.remove("submodules/higher-hrnet/lib/")
sys.path.append("submodules/hrnet/lib/")
import utils
importlib.reload(utils)
from .api import hrnet

sys.path.remove("submodules/hrnet/lib/")
sys.path.append("submodules/unitrack/")
import utils
importlib.reload(utils)
from .api import unitrack

from . import dataset, extracter
