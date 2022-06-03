from logging import Logger
from typing import Dict, Tuple

from group.group import Group
from individual.individual import Individual
from individual.individual_analyzer import IndividualAnalyzer
from numpy.typing import NDArray

from utility import json_handler


def load_individuals(json_path: str, cfg: dict) -> Tuple[Dict[int, Individual], int]:
    defs = IndividualAnalyzer.load_default(cfg)
    inds = {}
    last_frame_num = 0
    json_data = json_handler.load(json_path)
    for item in json_data:
        frame_num = item["frame"]
        if item["id"] not in inds:
            inds[item["id"]] = Individual(item["id"], defs)
        inds[item["id"]].from_json(item, frame_num)
        last_frame_num = max(last_frame_num, frame_num)

    return inds, last_frame_num


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
