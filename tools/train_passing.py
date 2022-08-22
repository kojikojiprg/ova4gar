import argparse
import os
import sys
from glob import glob

import torch
import yaml
from tqdm import tqdm

sys.path.append("src")
from group.passing.dataset import make_data_loaders
from group.passing.train_api import (
    init_loss,
    init_model,
    init_optimizer,
    init_scheduler,
    parameter_tuning,
    test,
    train,
)
from utility.activity_loader import load_individuals
from utility.logger import logger


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", required=True, type=int)
    parser.add_argument(
        "-t",
        "--trial",
        type=int,
        default=100,
        help="the number of trials of model parameter tuning",
    )
    parser.add_argument(
        "-p",
        "--parameter_tuning",
        default=False,
        action="store_true",
        help="if True, do model parameter tuning",
    )
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument(
        "-c",
        "--cfg_path",
        type=str,
        default="config/passing/pass_train.yaml",
        help="config path for model training",
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

    mdl_cfg_path = f"config/passing/pass_model_lstm_ep{args.epoch}.yaml"
    if os.path.exists(mdl_cfg_path):
        # load model config
        logger.info(f"=> loading model config from {mdl_cfg_path}")
        with open(mdl_cfg_path, "r") as f:
            mdl_cfg = yaml.safe_load(f)
    else:
        logger.info(f"=> model config {mdl_cfg_path} was not found")
        # initial model config
        mdl_cfg = train_cfg["default"]
        logger.info(f"=> model config is initialized {mdl_cfg}")

    if args.parameter_tuning:
        # parameter tuning
        logger.info("=> start parameter tuning")
        best_params = parameter_tuning(
            mdl_cfg, train_loader, test_loader, args.epoch, args.trial, device
        )
        mdl_cfg = mdl_cfg.update(best_params)

    # training
    logger.info("=> start training")
    model = init_model(mdl_cfg, device)
    criterion = init_loss(mdl_cfg["pos_weight"], device)
    optimizer = init_optimizer("Adam", mdl_cfg["lr"], mdl_cfg["weight_decay"], model)
    scheduler = init_scheduler(mdl_cfg["scheduler_rate"], optimizer)
    model = train(
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        args.epoch,
        device,
        val_loader=test_loader,
        logger=logger,
    )

    # test
    logger.info("=> testing")
    test(model, test_loader, device, logger)

    # save model
    model_path = f"models/passing/pass_model_lstm_ep{args.epoch}.pth"
    logger.info(f"=> saving model {model_path}")
    torch.save(model.state_dict(), model_path)
    logger.info(f"=> saving model config to {mdl_cfg_path}")
    mdl_cfg["pretrained_path"] = model_path
    with open(mdl_cfg_path, "w") as f:
        yaml.dump(mdl_cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()
