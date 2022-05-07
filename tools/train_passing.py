import argparse
import os
import sys
import time
from glob import glob

import numpy as np
import torch
import yaml
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch import nn, optim

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


def init_model(cfg_path, defs, model=None):
    if isinstance(model, PassingDetector):
        del model
    model = PassingDetector(cfg_path, defs)
    return model


def init_loss(pos_weight, device):
    pos_weight = torch.tensor(pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion


def init_optim(model, lr, rate):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: rate**epoch
    )
    return optimizer, scheduler


def scores(model, loader):
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
    return accuracy, precision, recall, f1


def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epoch_len, logger
):
    logger.info("=> start training")
    history = dict(train=[], val=[])
    for epoch in range(1, epoch_len + 1):
        ts = time.time()

        # train
        model.train()
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
        model.eval()
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
            f"Epoch[{epoch}/{(epoch_len)}] train loss: {train_loss:.5f}, val loss: {val_loss:.5f}, lr: {lr:.7f}, time: {te - ts:.2f}"
        )
    logger.info("=> end training")

    logger.info("=> calculating train scores")
    acc, pre, rcl, f1 = scores(model, train_loader)
    logger.info(
        f"=> train score\naccuracy: {acc}\npresision: {pre}\nrecall: {rcl}\nf1: {f1}"
    )

    return model, epoch, history


def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        logger.info("=> calculating test scores")
        acc, pre, rcl, f1 = scores(model, test_loader)
        logger.info(
            f"=> test score\naccuracy: {acc}\npresision: {pre}\nrecall: {rcl}\nf1: {f1}"
        )

    return acc, pre, rcl, f1


def main():
    args = _setup_parser()
    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dirs_all = {}
    for room_num, date_items in cfg["dataset"]["frame_number"].items():
        for date in date_items.keys():
            dirs = sorted(glob(os.path.join("data", room_num, date, "passing", "*")))
            data_dirs_all[f"{room_num}_{date}"] = dirs

    logger.info(f"=> loading individuals from {data_dirs_all}")
    inds = {}
    for key_prefix, dirs in data_dirs_all.items():
        for model_path in dirs:
            num = model_path.split("/")[-1]
            json_path = os.path.join(model_path, ".json", "individual.json")
            tmp_inds, _ = load_individuals(json_path, cfg["individual"])
            for pid, ind in tmp_inds.items():
                inds[f"{key_prefix}_{num}_{pid}"] = ind

    # create model
    model_cfg_path = cfg["group"]["indicator"]["passing"]["cfg_path"]
    grp_defs = cfg["group"]["indicator"]["passing"]["default"]
    detector = init_model(model_cfg_path, grp_defs, model=None)

    # create data loader
    train_loader, val_loader, test_loader = make_data_loaders(
        detector, inds, cfg["dataset"], logger
    )

    # init optimizer
    criterion = init_loss(cfg["optim"]["pos_weight"], detector.device)
    optimizer, scheduler = init_optim(
        detector.model, cfg["optim"]["lr"], cfg["optim"]["lr_rate"]
    )

    # train and test
    epoch_len = cfg["optim"]["epoch"]
    try:
        detector.model, epoch, history = train(
            detector.model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            epoch_len,
            logger,
        )
    except KeyboardInterrupt:
        pass
    acc, pre, rcl, f1 = test(detector.model, test_loader)

    # save model
    model_path = os.path.join("models", "passing", f"pass_model_ep{epoch}.pth")
    logger.info(f"=> saving model params to {model_path}")
    torch.save(detector.model.state_dict(), model_path)


if __name__ == "__main__":
    main()
