from logging import Logger
from typing import Any, Dict, List

from individual.individual import Individual
from numpy.typing import NDArray

from group.indicator import attention, passing
from group.passing.passing_detector import PassingDetector


class Group:
    def __init__(self, cfg: dict, field: NDArray, logger: Logger, device: str):
        self._keys = list(cfg["indicator"].keys())
        self._funcs = {k: eval(k) for k in self._keys}
        self._defs: Dict[str, Any] = self.load_default(cfg)

        self._field = field
        self._logger = logger

        pass_cfg_path = cfg["indicator"]["passing"]["cfg_path"]
        self._logger.info(f"=> load passing detector from {pass_cfg_path}")
        self._pass_clf = PassingDetector(pass_cfg_path, self._defs["passing"], device)
        self._pass_clf.eval()

        self._idc_dict: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self._keys}
        self._idc_que: Dict[str, Any] = {
            "attention": [],
            "passing": {},
        }

    @staticmethod
    def load_default(cfg: dict) -> Dict[str, Any]:
        defs: Dict[str, Any] = {}
        for ind_key, item in cfg["indicator"].items():
            defs[ind_key] = {}
            for key, val in item["default"].items():
                defs[ind_key][key] = val
        return defs

    def __del__(self):
        del self._field, self._logger
        del self._pass_clf
        del self._idc_dict, self._idc_que

    def get(self, key):
        return self._idc_dict[key]

    def calc_indicator(self, frame_num: int, individuals: List[Individual]):
        for key, func in self._funcs.items():
            if key == "passing":
                value, queue = func(
                    frame_num,
                    individuals,
                    self._idc_que[key],
                    self._pass_clf,
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
