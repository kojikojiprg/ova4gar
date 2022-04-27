import os
from logging import Logger
from typing import Any, Dict

from tqdm import tqdm
from utility import json_handler
from utility.transform import Homography

from individual.individual import Individual


class IndividualAnalyzer:
    def __init__(self, cfg: dict, logger: Logger):
        # load config
        cfg = cfg["individual"]

        # read default values
        self._defaults: Dict[str, Dict[str, Any]] = {"indicator": {}, "keypoint": {}}
        for indicator_key, item in cfg["indicator"].items():
            self._defaults["indicator"][indicator_key] = {}
            for key, val in item["default"].items():
                self._defaults[indicator_key][key] = val
        for key, val in cfg["keypoints"].items():
            self._defaults["keypoint"][key] = val

        self._logger = logger

    def analyze(self, data_dir: str, homo: Homography):
        kps_json_path = os.path.join(data_dir, "json", "keipoints.json")
        self._logger.info(f"=> load keypoint data from {kps_json_path}")
        keypoints_data = json_handler.load(kps_json_path)

        individuals = {}
        json_data = []
        for data in tqdm(keypoints_data):
            frame_num = data["frame"]
            pid = data["person"]
            keypoints = data["keypoints"]

            # obtain individual
            if pid not in individuals:
                individuals[pid] = Individual(pid, homo, self._defaults)
            ind = individuals[pid]

            # calc indicators of individual
            ind.calc_indicator(frame_num, keypoints)

            # create and append json data
            output = ind.to_json(frame_num)
            if output is not None:
                json_data.append(output)

        # write json
        ind_json_path = os.path.join(data_dir, "json", "individual.json")
        self._logger.info(f"=> write individual data to {ind_json_path}")
        json_handler.dump(json_data, ind_json_path)

        del keypoints_data, individuals, json_data  # release memory
