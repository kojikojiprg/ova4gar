import sys
import importlib

# hrnet and unitrack have "uitls" module.
# Avoid import error.
sys.path.append("submodules/hrnet/lib/")
from . import hrnet

sys.path.remove("submodules/hrnet/lib/")
sys.path.append("submodules/unitrack/")
import utils
importlib.reload(utils)
from . import unitrack

from . import dataset
from . import extracter
