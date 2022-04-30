import argparse
import os
import sys
from glob import glob

import yaml

sys.path.append("src")
from group.group import Group
from group.passing.passing_detector import PassingDetector
from utility.activity_loader import load_individuals
from utility.logger import logger


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cfg_path", type=str, default="config/passing/pass_train.yaml"
    )

    return parser.parse_args()


def main():
    args = _setup_parser()
    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dirs_all = {}
    for room_num, date_items in cfg["dataset"].items():
        for date in date_items.keys():
            dirs = sorted(glob(os.path.join("data", room_num, date, "passing", "*")))
            data_dirs_all[f"{room_num}_{date}"] = dirs

    logger.info(f"=> loading individuals from {data_dirs_all}")
    inds = {}
    for key_prefix, dirs in data_dirs_all:
        for path in dirs:
            json_path = os.path.join(path, ".json", "individual.json")
            tmp_inds, _ = load_individuals(json_path, cfg["individual"])
            for pid, ind in tmp_inds.items():
                inds[f"{key_prefix}_{pid}"] = ind

    # create model
    grp_defs = Group.load_default(cfg["group"])
    model = PassingDetector(cfg["group"]["indicator"]["passing"]["cfg_path"], grp_defs)


if __name__ == "__main__":
    main()
