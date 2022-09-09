import gc
import os
from logging import Logger
from typing import Any, Dict, List

from tqdm import tqdm
from utility import json_handler
from utility.transform import Homography

from individual.individual import Individual


class IndividualAnalyzer:
    def __init__(self, ind_cfg: dict, logger: Logger):
        # load config
        self._defaults: Dict[str, Dict[str, Any]] = self.load_default(ind_cfg)

        self._logger = logger

    @staticmethod
    def load_default(cfg: dict):
        defaults: Dict[str, Dict[str, Any]] = {"indicator": {}, "keypoint": {}}
        for indicator_key, item in cfg["indicator"].items():
            defaults["indicator"][indicator_key] = {}
            for key, val in item["default"].items():
                defaults["indicator"][indicator_key][key] = val
        for key, val in cfg["keypoint"]["default"].items():
            defaults["keypoint"][key] = val
        return defaults

    def analyze(
        self,
        data_dir: str,
        homo: Homography,
    ):
        # load keypoints data from json file
        kps_json_path = os.path.join(data_dir, ".json", "keypoints.json")
        self._logger.info(f"=> loading keypoint data from {kps_json_path}")
        keypoints_data = json_handler.load(kps_json_path)

        individuals: Dict[int, Individual] = {}
        json_data: List[Dict[str, Any]] = []
        pre_frame_num = 1
        for data in tqdm(keypoints_data):
            frame_num = data["frame"]
            pid = data["id"]
            keypoints = data["keypoints"]

            # obtain individual
            if pid not in individuals:
                individuals[pid] = Individual(pid, self._defaults)
            ind = individuals[pid]

            # calc indicators of individual
            ind.calc_indicator(frame_num, keypoints, homo)

            # create and append json data
            output = ind.to_dict(frame_num)
            json_data.append(output)

            # when frame next, write video frame
            if pre_frame_num < frame_num:
                pre_frame_num = frame_num  # update pre_frame_num

        # write json
        ind_json_path = os.path.join(data_dir, ".json", "individual.json")
        self._logger.info(f"=> writing individual data to {ind_json_path}")
        json_handler.dump(json_data, ind_json_path)

        # release memory
        del keypoints_data, individuals, json_data
        gc.collect()
