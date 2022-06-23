import os
from glob import glob
from logging import Logger
from typing import Dict, List

from group.group import Group
from individual.individual import Individual
from individual.individual_analyzer import IndividualAnalyzer
from numpy.typing import NDArray

from utility import json_handler


def get_data_dirs(room_num: str, surgery_num: str, expand_name: str = "") -> List[str]:
    data_dir = os.path.join("data", room_num, surgery_num, expand_name)
    data_dirs = sorted(glob(os.path.join(data_dir, "*")))
    for i in range(len(data_dirs)):
        if data_dirs[i].endswith("passing") or data_dirs[i].endswith("attention"):
            del data_dirs[i]

    return data_dirs


def load_individuals(json_path: str, cfg: dict) -> Dict[int, Individual]:
    defs = IndividualAnalyzer.load_default(cfg)
    inds = {}
    json_data = json_handler.load(json_path)
    for item in json_data:
        frame_num = item["frame"]
        if item["id"] not in inds:
            inds[item["id"]] = Individual(item["id"], defs)
        inds[item["id"]].from_json(item, frame_num)

    return inds


def load_group(
    json_path: str,
    cfg: dict,
    field: NDArray,
    logger: Logger,
    device: str = "cuda",
    only_data_loading: bool = False,
) -> Group:
    group = Group(cfg, field, logger, device, only_data_loading)
    json_data = json_handler.load(json_path)
    group.from_json(json_data)

    return group
