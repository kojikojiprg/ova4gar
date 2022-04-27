import os
from logging import Logger
from typing import Dict, Tuple

import numpy as np
from individual.individual import Individual
from individual.individual_analyzer import IndividualAnalyzer
from tqdm import tqdm
from utility import json_handler

from group.group import Group


class GroupAnalyzer:
    def __init__(self, cfg: dict, logger: Logger):
        self._cfg = cfg["group"]
        self._logger = logger

        self._ind_defs = IndividualAnalyzer.load_default(cfg["individual"])

    def analyze(self, data_dir: str, field: np.typing.NDArray):
        ind_json_path = os.path.join(data_dir, "json", "individual.json")
        self._logger.info(f"=> load individual data from {ind_json_path}")
        inds, last_frame_num = self._load_individuals(ind_json_path)

        self._logger.info(f"=> construct group activity model for {data_dir}")
        group = Group(self._cfg, field, self._logger)

        for frame_num in tqdm(range(last_frame_num)):
            inds_data_frame = [
                ind for ind in inds.values() if ind.exists_on_frame(frame_num)
            ]

            group.calc_indicator(frame_num, inds_data_frame)

        # write json
        group_data = group.to_json()
        grp_json_path = os.path.join(data_dir, "json", "group.json")
        self._logger.info(f"=> write group data to {grp_json_path}")
        json_handler.dump(group_data, grp_json_path)

        del inds, group, group_data  # release memory

    def _load_individuals(
        self, ind_json_path: str
    ) -> Tuple[Dict[int, Individual], int]:
        inds = {}
        last_frame_num = 0
        json_data = json_handler.load(ind_json_path)
        for item in json_data:
            frame_num = item["frame"]
            if item["id"] not in inds:
                inds[item["id"]] = Individual(item["id"], self._ind_defs)
            inds[item["id"]].from_json(item, frame_num)
            last_frame_num = frame_num if frame_num > last_frame_num else last_frame_num

        return inds, last_frame_num
