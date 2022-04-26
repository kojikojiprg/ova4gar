import os
from typing import Any, Dict

import numpy as np
from tqdm import tqdm
from utility import json_handler

from group.group import Group


class GroupAnalyzer:
    def __init__(self, **cfg):
        cfg = cfg["group"]
        self.defaults: Dict[str, Any] = {}
        for indicator_key, item in cfg["indicator"].items():
            self.defaults[indicator_key] = {}
            for key, val in item["default"].items():
                self.defaults[indicator_key][key] = val

    def analyze(self, data_dir: str, field: np.typing.NDArray, **karg):
        ind_json_path = os.path.join(data_dir, "json", "individual.json")
        individuals = json_handler.load(ind_json_path)

        group = Group(field, method)

        last_frame_num = individuals[-1]["frame"] + 1
        for frame_num in tqdm(range(last_frame_num)):
            ind_data_frame = [
                data for data in individuals if data["frame"] == frame_num
            ]

            group.calc_indicator(frame_num, ind_data_frame, **karg)

        # write json
        group_data = group.to_json()
        grp_json_path = os.path.join(data_dir, "json", "group.json")
        json_handler.dump(group_data, grp_json_path)
