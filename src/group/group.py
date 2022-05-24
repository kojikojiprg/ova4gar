import gc
from logging import Logger
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from individual.individual import Individual
from numpy.typing import NDArray

from group.indicator import attention as func_attention
from group.indicator import passing as func_passing
from group.passing.lstm_model import LSTMModel


class Group:
    def __init__(self, cfg: dict, field: NDArray, logger: Logger, device: str):
        self._keys = list(cfg.keys())
        self._funcs = {k: eval(f"func_{k}") for k in self._keys}
        self._defs: Dict[str, Any] = self.load_default(cfg)

        self._field = field
        self._logger = logger

        # load passing model
        pass_model_cfg_path = cfg["passing"]["cfg_path"]
        self._logger.info(f"=> load passing detector from {pass_model_cfg_path}")
        with open(pass_model_cfg_path) as f:
            model_cfg = yaml.safe_load(f)
        self._pass_model = LSTMModel(**model_cfg)
        param = torch.load(model_cfg["pretrained_path"])
        self._pass_model.load_state_dict(param)
        self._pass_model.to(device)
        self._pass_model.eval()
        self._device = device

        # dcreate indicator values
        self._idc_dict: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self._keys}
        self._idc_que: Dict[str, Any] = {
            "attention": [],
            "passing": {},
        }

    @staticmethod
    def load_default(cfg: dict) -> Dict[str, Any]:
        defs: Dict[str, Any] = {}
        for idc_key, item in cfg.items():
            defs[idc_key] = {}
            for key, val in item["default"].items():
                defs[idc_key][key] = val
        return defs

    def __del__(self):
        del self._field, self._logger
        del self._pass_model
        del self._idc_dict, self._idc_que
        gc.collect()

    def get(self, key):
        return self._idc_dict[key]

    @property
    def passing(self) -> Dict[str, List[int]]:
        data = self._idc_dict["passing"]

        data_dict: Dict[str, List[int]] = {}
        for row in data:
            frame_num = row["frame"]
            persons = row["persons"]

            pair_key = f"{persons[0]}_{persons[1]}"
            if pair_key not in data_dict:
                data_dict[pair_key] = []

            data_dict[pair_key].append(frame_num)

        return data_dict

    @property
    def attention(self) -> NDArray:
        all_data = self._idc_dict["attention"]

        shape = tuple(
            np.array(self._field.shape[1::-1]) // self._defs["attention"]["division"]
        )

        heatmap_lst = []
        max_frame_num = max([data["frame"] for data in all_data])
        for frame_num in range(max_frame_num):
            frame_data = [data for data in all_data if data["frame_nun"] == frame_num]

            # init heatmap
            heatmap = np.zeros(shape, dtype=np.float32)
            for data in frame_data:
                coor = tuple(
                    np.array(data["point"]) // self._defs["attention"]["division"]
                )
                value = data["value"]
                heatmap[coor] = value
            heatmap_lst.append(heatmap)

        return np.array(heatmap_lst)

    def calc_indicator(self, frame_num: int, individuals: List[Individual]):
        for key, func in self._funcs.items():
            if key == "passing":
                value, queue = func(
                    frame_num,
                    individuals,
                    self._idc_que[key],
                    self._defs["passing"],
                    self._pass_model,
                    self._device,
                )
            elif key == "attention":
                value, queue = func(
                    frame_num,
                    individuals,
                    self._idc_que[key],
                    self._field,
                    self._defs["attention"],
                )
            else:
                raise KeyError

            if value is not None:
                self._idc_dict[key] += value
            self._idc_que[key] = queue

    def to_dict(self):
        return self._idc_dict

    def from_json(self, json_data):
        self._idc_dict = json_data
