import os
from types import SimpleNamespace
from typing import Any, Dict

import yaml
from tqdm import tqdm
from utility import json_handler
from utility.transform import Homography

from individual.individual import Individual


class Analyzer:
    def __init__(self, cfg_path: str):
        # load config
        with open(cfg_path) as f:
            args = yaml.safe_load(f)

        # read default values
        self.defaults: Dict[str, Dict[str, Any]] = {"indicator": {}, "keypoint": {}}
        for indicator_key, item in args["indicator"].items():
            self.defaults["indicator"][indicator_key] = {}
            for key, val in item["default"].items():
                self.defaults[indicator_key][key] = val
        for key, val in args["keypoints"].items():
            self.defaults["keypoint"][key] = val

    def analyze(self, data_dir: str, homo: Homography):
        kps_json_path = os.path.join(data_dir, "json", "keipoints.json")
        keypoints_data = json_handler.load(kps_json_path)

        individuals = {}
        json_data = []
        for data in tqdm(keypoints_data):
            frame_num = data["frame"]
            pid = data["person"]
            keypoints = data["keypoints"]

            # obtain individual
            if pid not in individuals:
                individuals[pid] = Individual(pid, homo, self.defaults)
            ind = individuals[pid]

            # calc indicators of individual
            ind.calc_indicator(frame_num, keypoints)

            # create and append json data
            output = ind.to_json(frame_num)
            if output is not None:
                json_data.append(output)

        # write json
        json_path = os.path.join(data_dir, "json", "individual.json")
        json_handler.dump(json_data, json_path)

        del keypoints_data, individuals, json_data  # release memory
