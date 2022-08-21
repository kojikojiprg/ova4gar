import argparse
import os
import sys
from glob import glob

import torch
import yaml
from tqdm import tqdm

sys.path.append("src")
from group.passing.dataset import make_data_loaders
from group.passing.train_api import parameter_tuning
from utility.activity_loader import load_individuals
from utility.logger import logger


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", required=True, type=int)
    parser.add_argument("-t", "--trial", type=int, default=100)
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument(
        "-c", "--cfg_path", type=str, default="config/passing/pass_train.yaml"
    )

    return parser.parse_args()


def main():
    args = _setup_parser()

    # load configs
    with open(args.cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)
    with open(train_cfg["config_path"]["individual"], "r") as f:
        ind_cfg = yaml.safe_load(f)
    with open(train_cfg["config_path"]["group"], "r") as f:
        grp_cfg = yaml.safe_load(f)

    # get data directories
    data_dirs_all = {}
    for room_num, surgery_data in train_cfg["dataset"]["setting"].items():
        for surgery_num in surgery_data.keys():
            dirs = sorted(
                glob(os.path.join("data", room_num, surgery_num, "passing", "*"))
            )
            data_dirs_all[f"{room_num}_{surgery_num}"] = dirs

    # load individual data
    logger.info(f"=> loading individuals from {data_dirs_all}")
    inds = {}
    for key_prefix, dirs in data_dirs_all.items():
        for path in tqdm(dirs):
            num = path.split("/")[-1]
            json_path = os.path.join(path, ".json", "individual.json")
            tmp_inds = load_individuals(json_path, ind_cfg)
            for pid, ind in tmp_inds.items():
                inds[f"{key_prefix}_{num}_{pid}"] = ind

    # create data loader
    dataset_cfg = train_cfg["dataset"]
    passing_defs = grp_cfg["passing"]["default"]
    train_loader, test_loader = make_data_loaders(
        inds, dataset_cfg, passing_defs, logger
    )

    # set cuda and device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # parameter tuning
    mdl_cfg = {
        "n_linears": 2,
        "hidden_dims": [16, 8],
        "rnn_dropout": 0.25,
        "n_classes": 2,
        "size": 4,
    }  # To Do save yaml
    logger.info("=> start parameter tuning")
    best_params = parameter_tuning(
        mdl_cfg, train_loader, test_loader, args.epoch, args.trial, device
    )
    print(best_params)


if __name__ == "__main__":
    main()
