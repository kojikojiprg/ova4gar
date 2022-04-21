import json
import os
from json import JSONEncoder

import numpy as np


def load(json_path):
    data = {}
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def dump(data, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(data, f, cls=MyEncoder)


class MyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
