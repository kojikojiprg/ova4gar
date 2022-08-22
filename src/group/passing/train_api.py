import time
from logging import Logger
from typing import Tuple, Union

import numpy as np
import optuna
import torch
from group.passing.lstm_model import LSTMModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn, optim


def init_model(model_cfg, device) -> LSTMModel:
    model = LSTMModel(**model_cfg).to(device)
    return model


def init_loss(pos_weight, device) -> nn.BCEWithLogitsLoss:
    pos_weight = torch.tensor(pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion


def init_optimizer(
    optimizer_name, lr, weight_decay, model
) -> Union[optim.Adam, optim.SGD, optim.RMSprop]:
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        return optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters())
    else:
        raise NameError


def init_scheduler(rate, optimizer) -> optim.lr_scheduler.LambdaLR:
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: rate**epoch
    )
    return scheduler


def train(
    model: LSTMModel,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: Union[optim.Adam, optim.SGD, optim.RMSprop],
    scheduler: optim.lr_scheduler.LambdaLR,
    epoch_len: int,
    device: str,
    val_loader: torch.utils.data.DataLoader = None,
    logger: Logger = None,
) -> LSTMModel:
    history: dict = dict(train=[], val=[])
    for epoch in range(1, epoch_len + 1):
        ts = time.time()

        # train
        model.train()
        train_losses = []
        lr = optimizer.param_groups[0]["lr"]
        for x, y in train_loader:
            optimizer.zero_grad()

            pred = model(x.to(device))

            loss = criterion(pred.requires_grad_(), y.to(device))
            loss.backward()
            train_losses.append(loss.item())

            optimizer.step()

        scheduler.step()

        # validate
        model.eval()
        val_losses = []
        if val_loader is not None:
            with torch.no_grad():
                for x, y in val_loader:
                    pred = model(x.to(device))

                    loss = criterion(pred.requires_grad_(), y.to(device))
                    val_losses.append(loss.item())
        else:
            val_losses.append(np.nan)

        te = time.time()
        if logger is not None:
            train_loss = np.mean(train_losses)
            history["train"].append(train_loss)
            val_loss = np.mean(val_losses)
            history["val"].append(val_loss)

            logger.info(
                f"Epoch[{epoch}/{(epoch_len)}] train loss: {train_loss:.5f}, "
                + f"val loss: {val_loss:.5f}, lr: {lr:.7f}, time: {te - ts:.2f}"
            )

    return model


def test(
    model: LSTMModel, test_loader: torch.utils.data.DataLoader, device: str, logger: Logger = None
) -> Tuple[float, float, float, float]:
    model.eval()
    preds, y_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device))
            pred = pred.max(1)[1]
            pred = pred.cpu().numpy().tolist()
            y = y.cpu().numpy().T[1].astype(int).tolist()
            preds += pred
            y_all += y

    try:
        acc = accuracy_score(y_all, preds)
        pre = precision_score(y_all, preds)
        rcl = recall_score(y_all, preds)
        f1 = f1_score(y_all, preds)
    except ZeroDivisionError:
        acc = np.nan
        pre = np.nan
        rcl = np.nan
        f1 = np.nan

    if logger is not None:
        logger.info(f"accuracy: {acc}")
        logger.info(f"precision: {pre}")
        logger.info(f"recall: {rcl}")
        logger.info(f"f1: {f1}")

    return acc, pre, rcl, f1


class Objective:
    def __init__(
        self,
        mdl_cfg: dict,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epoch: int,
        device: str,
    ):
        self._mdl_cfg = mdl_cfg
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._epoch = epoch
        self._device = device

    def __call__(self, trial: optuna.Trial):
        n_rnns = trial.suggest_int("n_rnns", 1, 3)
        rnn_hidden_dim = trial.suggest_categorical("rnn_hidden_dim", [8, 16, 32])
        rnn_dropout = trial.suggest_categorical("rnn_dropout", [0, 0.25, 0.5])

        self._mdl_cfg["n_rnns"] = n_rnns
        self._mdl_cfg["rnn_hidden_dim"] = rnn_hidden_dim
        self._mdl_cfg["rnn_dropout"] = rnn_dropout
        model = init_model(self._mdl_cfg, self._device)

        pos_weight = int(trial.suggest_discrete_uniform("pos_weight", 2, 8, 2))
        loss = init_loss(pos_weight, self._device)

        optimizer_name = "Adam"
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
        optimizer = init_optimizer(optimizer_name, lr, weight_decay, model)

        scheduler_rate = trial.suggest_uniform("scheduler_rate", 0.99, 1.0)
        scheduler = init_scheduler(scheduler_rate, optimizer)

        model = train(
            model,
            self._train_loader,
            loss,
            optimizer,
            scheduler,
            self._epoch,
            self._device,
        )
        _, _, _, f1 = test(model, self._test_loader, self._device)

        return 1 - f1  # return error rate


def parameter_tuning(mdl_cfg, train_loader, test_loader, epoch, trial_size, device):
    objective = Objective(mdl_cfg, train_loader, test_loader, epoch, device)
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    study.optimize(
        objective, n_trials=trial_size, gc_after_trial=True, show_progress_bar=True
    )

    return study.best_params
