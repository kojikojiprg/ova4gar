import os
from logging import Logger

import numpy as np
from torch import isin
from tqdm import tqdm
from utility import json_handler

from group.group import Group


class GroupAnalyzer:
    def __init__(self, cfg: dict, logger: Logger):
        self._cfg = cfg["group"]
        self._logger = logger

    def analyze(self, data_dir: str, field: np.typing.NDArray):
        ind_json_path = os.path.join(data_dir, "json", "individual.json")
        self._logger.info(f"=> load individual data from {ind_json_path}")
        individuals = json_handler.load(ind_json_path)

        self._logger.info(f"=> construct group activity model for {data_dir}")
        group = Group(self._cfg, field, self._logger)

        last_frame_num = individuals[-1]["frame"] + 1
        for frame_num in tqdm(range(last_frame_num)):
            ind_data_frame = [
                data for data in individuals if data["frame"] == frame_num
            ]

            group.calc_indicator(frame_num, ind_data_frame)

        # write json
        group_data = group.to_json()
        grp_json_path = os.path.join(data_dir, "json", "group.json")
        self._logger.info(f"=> write group data to {grp_json_path}")
        json_handler.dump(group_data, grp_json_path)

        del individuals, group, group_data  # release memory
