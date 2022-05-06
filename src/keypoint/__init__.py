import importlib
import logging
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# disable matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
import matplotlib

del matplotlib

sys.path.append(os.path.join(root, "submodules"))

# Avoid import error.
sys.path.append(os.path.join(root, "submodules/higher_hrnet/lib/"))
from .api import higher_hrnet

sys.path.remove(os.path.join(root, "submodules/higher_hrnet/lib/"))
sys.path.append(os.path.join(root, "submodules/hrnet/lib/"))
import models

importlib.reload(models)
from .api import hrnet

sys.path.remove(os.path.join(root, "submodules/hrnet/lib/"))
sys.path.append(os.path.join(root, "submodules/unitrack/"))
import utils

importlib.reload(utils)
from .api import unitrack

sys.path.remove(os.path.join(root, "submodules/unitrack/"))
from . import dataset, extracter
