from logging import Logger
from typing import Any, Dict, List

import numpy as np
from individual.individual import Individual

from group.indicator import attention, passing
from group.passing_detector import PassingDetector


class Group:
    def __init__(self, cfg: dict, field: np.typing.NDArray, logger: Logger):
        self._keys = list(cfg["indicator"].keys())
        self._funcs = {k: eval(k) for k in self._keys}
        self._defs: Dict[str, Any] = {}
        for ind_key, item in cfg["indicator"].items():
            self._defs[ind_key] = {}
            for key, val in item["default"].items():
                self._defs[ind_key][key] = val

        self._field = field
        self._logger = logger

        pass_cfg_path = cfg["indicator"]["passing"]["cfg_path"]
        self._logger.info(f"=> load passing detector from {pass_cfg_path}")
        self._pass_clf = PassingDetector(pass_cfg_path, self._defs["passing"])
        self._pass_clf.eval()

        self._idc_dict: Dict[str, Any] = {k: [] for k in self._keys}
        self._idc_que: Dict[str, Any] = {
            "attention": [],
            "passing": {},
        }

    def calc_indicator(self, frame_num: int, individuals: List[Individual]):
        for key, func in self._keys:
            if key == self._keys[0]:
                # key == passing
                value, queue = func(
                    frame_num,
                    individuals,
                    self._idc_que[key],
                    self._pass_clf,
                )
            elif key == self._keys[1]:
                # key == attention
                value, queue = func(
                    frame_num,
                    individuals,
                    self._idc_que[key],
                    self._field,
                )
            else:
                raise KeyError

            self._idc_dict[key] += value
            self._idc_que[key] = queue

    def to_json(self):
        return self._idc_dict
