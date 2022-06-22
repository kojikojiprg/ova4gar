import gc
import os
from logging import Logger

import torch
from numpy.typing import NDArray
from tqdm import tqdm
from utility import json_handler
from utility.activity_loader import load_individuals

from group.group import Group


class GroupAnalyzer:
    def __init__(self, grp_cfg: dict, ind_cfg: dict, logger: Logger, device: str):
        self._ind_cfg = ind_cfg
        self._grp_cfg = grp_cfg
        self._keys = list(self._grp_cfg.keys())
        self._logger = logger
        self._device = device

    def __del__(self):
        torch.cuda.empty_cache()
        gc.collect()

    def analyze(self, data_dir: str, field: NDArray):
        # load individual data from json file
        ind_json_path = os.path.join(data_dir, ".json", "individual.json")
        self._logger.info(f"=> load individual data from {ind_json_path}")
        inds = load_individuals(ind_json_path, self._ind_cfg)
        last_frame_num = max([ind.to_dict["keypoints"].keys() for ind in inds])

        # create group class
        self._logger.info(f"=> construct group activity model for {data_dir}")
        group = Group(self._grp_cfg, field, self._logger, self._device)

        for frame_num in tqdm(range(last_frame_num)):
            inds_per_frame = [
                ind for ind in inds.values() if ind.exists_on_frame(frame_num)
            ]

            group.calc_indicator(frame_num, inds_per_frame)

        # write json
        grp_json_path = os.path.join(data_dir, ".json", "group.json")
        self._logger.info(f"=> write group data to {grp_json_path}")
        json_handler.dump(group.to_dict(), grp_json_path)

        # release memory
        torch.cuda.empty_cache()
        del inds, group
        gc.collect()
