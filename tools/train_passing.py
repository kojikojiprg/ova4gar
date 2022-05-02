import argparse
import os
import sys
from glob import glob
from time import time

import numpy as np
import torch
import yaml
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch import nn, optim
from tqdm import tqdm

sys.path.append("src")
from group.passing.dataset import make_data_loaders
from group.passing.passing_detector import PassingDetector
from utility.activity_loader import load_individuals
from utility.logger import logger


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cfg_path", type=str, default="config/passing/pass_train.yaml"
    )

    return parser.parse_args()


def _init_model(cfg_path, defs, model=None):
    if isinstance(model, PassingDetector):
        del model
    model = PassingDetector(cfg_path, defs)
    return model


def _init_loss(pos_weight, device):
    pos_weight = torch.tensor(pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion


def _init_optim(model, lr, rate):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: rate**epoch
    )
    return optimizer, scheduler


def _scores(model, loader):
    preds, y_all = [], []
    for x, y in loader:
        pred = model(x)
        pred = pred.max(1)[1]
        pred = pred.cpu().numpy().tolist()
        y = y.cpu().numpy().T[1].astype(int).tolist()
        preds += pred
        y_all += y

    accuracy = accuracy_score(y_all, preds)
    precision = precision_score(y_all, preds)
    recall = recall_score(y_all, preds)
    f1 = f1_score(y_all, preds)
    print(
        "accuracy: {:.3f}".format(accuracy),
        "precision: {:.3f}".format(precision),
        "recall: {:.3f}".format(recall),
        "f1_score: {:.3f}".format(f1),
    )
    return accuracy, precision, recall, f1


def _train(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epoch_len, logger
):
    history = dict(train=[], val=[])
    for epoch in tqdm(range(1, epoch_len + 1)):
        ts = time.time()

        # train
        model = model.train()
        lr = optimizer.param_groups[0]["lr"]
        train_losses = []
        for x, y in train_loader:
            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred.requires_grad_(), y)
            loss.backward()
            train_losses.append(loss.item())

            optimizer.step()

        scheduler.step()

        # validate
        model = model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)

                loss = criterion(pred.requires_grad_(), y)
                val_losses.append(loss.item())

        te = time.time()
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        logger.info(
            f"Epoch[{epoch}/{(epoch_len)}] train loss: {train_loss:.5f},\
            val loss: {val_loss:.5f}, lr: {lr:.7f}, time: {te - ts:.2f}"
        )

    return model, epoch, history


def _testing(model, test_loader):
    model.eval()
    with torch.no_grad():
        print("train scores")
        _scores(model, test_loader)
        print("test scores")
        acc, pre, rcl, f1 = _scores(model, test_loader)

    return model, acc, pre, rcl, f1


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
        for model_path in dirs:
            num = model_path.split("/")[-1]
            json_path = os.path.join(model_path, ".json", "individual.json")
            tmp_inds, _ = load_individuals(json_path, cfg["individual"])
            for pid, ind in tmp_inds.items():
                inds[f"{key_prefix}_{num}_{pid}"] = ind

    # create model
    model_cfg_path = cfg["group"]["indicator"]["passing"]["cfg_path"]
    grp_defs = cfg["group"]
    model = _init_model(model_cfg_path, grp_defs)

    # create data loader
    train_loader, val_loader, test_loader = make_data_loaders(
        model, inds, cfg["dataset"]
    )

    # init optimizer
    criterion = _init_loss(cfg["optim"]["pos_weight"])
    optimizer, scheduler = _init_optim(cfg["optim"]["lr"], cfg["optim"]["lr_rate"])

    # train and test
    try:
        model, epoch, history = _train(
            model, train_loader, criterion, optimizer, scheduler
        )
    except KeyboardInterrupt:
        pass
    acc, pre, rcl, f1 = _testing(model, test_loader)
    logger.info(
        f"=> test score\naccuracy: {acc}\npresision: {pre}\nrecall: {rcl}\nf1: {f1}"
    )

    # save model
    model_path = os.path.join("models", "passing", f"pass_model_ep{epoch}.pth")
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
