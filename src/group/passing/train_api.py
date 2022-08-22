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
    lr = float(lr)
    weight_decay = float(weight_decay)

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
        optimizer, lr_lambda=lambda epoch: float(rate) ** epoch
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
    model: LSTMModel,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    logger: Logger = None,
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
        tuning_cfg: dict,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epoch: int,
        device: str,
    ):
        self._mdl_cfg = mdl_cfg
        self._tuning_cfg = tuning_cfg
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._epoch = epoch
        self._device = device

    def _set_trial(self, name: str, trial: optuna.Trial):
        typ = self._tuning_cfg[name][0]
        params = self._tuning_cfg[name][1]

        if typ == "categorical":
            return trial.suggest_categorical(name, params)
        elif typ == "discrete_uniform":
            return trial.suggest_discrete_uniform(
                name, float(params[0]), float(params[1]), float(params[2])
            )
        elif typ == "float":
            return trial.suggest_float(name, float(params[0]), float(params[1]))
        elif typ == "int":
            return trial.suggest_int(name, int(params[0]), int(params[1]))
        elif typ == "loguniform":
            return trial.suggest_loguniform(name, float(params[0]), float(params[1]))
        elif typ == "uniform":
            return trial.suggest_uniform(name, float(params[0]), float(params[1]))
        else:
            raise NameError

    def __call__(self, trial: optuna.Trial):
        n_rnns = self._set_trial("n_rnns", trial)
        rnn_hidden_dim = self._set_trial("rnn_hidden_dim", trial)
        rnn_dropout = self._set_trial("rnn_dropout", trial)

        self._mdl_cfg["n_rnns"] = n_rnns
        self._mdl_cfg["rnn_hidden_dim"] = rnn_hidden_dim
        self._mdl_cfg["rnn_dropout"] = rnn_dropout
        model = init_model(self._mdl_cfg, self._device)

        pos_weight = self._set_trial("pos_weight", trial)
        loss = init_loss(int(pos_weight), self._device)

        optimizer_name = "Adam"
        lr = self._set_trial("lr", trial)
        weight_decay = self._set_trial("weight_decay", trial)
        optimizer = init_optimizer(optimizer_name, lr, weight_decay, model)

        scheduler_rate = self._set_trial("scheduler_rate", trial)
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


def parameter_tuning(
    mdl_cfg: dict,
    tuning_cfg: dict,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epoch: int,
    trial_size: int,
    device: str,
    db_path: str,
):
    objective = Objective(mdl_cfg, tuning_cfg, train_loader, test_loader, epoch, device)
    study_name = f"passing_ep{epoch}"
    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        objective, n_trials=trial_size, gc_after_trial=True, show_progress_bar=True
    )

    return study.best_params
