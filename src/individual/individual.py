from ctypes import Union
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
from keypoint.keypoint import Keypoints, body
from utility.transform import Homography

from individual_que import KeypointQue, Que


class Individual:
    def __init__(self, pid: int, homo: Homography, defaults: SimpleNamespace):
        self.id = pid

        self._defs = defaults.indicator.__dict__
        self._keys = list(self._defs.keys())
        self._funcs = {k: eval(k) for k in self._keys}
        self._homo: np.ndarray = homo
        self._pre_frame_num: int = 0

        self._kps_dict: Dict[int, Keypoints] = {}
        self._kps_que: KeypointQue = KeypointQue(defaults.keypoints)
        self._idc_dict: Dict[str, Any] = {k: {} for k in self._keys}
        self._idc_que: Dict[str, Que] = {k: Que(self._defs[k]) for k in self._keys}

    def calc_indicator(self, frame_num: int, kps: Any):
        # calc keypoints
        if kps is None:
            return
        kps = Keypoints(kps)
        kps, self._kps_dict = self._kps_que.put_pop_kps(
            frame_num, self._pre_frame_num, kps, self._kps_dict
        )

        # calc indicators
        for k in self._keys:
            val = self._funcs[k](kps, self._homo, self._idc_que[k], **self._defs[k])

            self._idc_dict[k][frame_num] = val

        # update pre frame num
        self._pre_frame_num = frame_num

    def get_indicator(self, key: str, frame_num: int) -> Any:
        if key not in self._keys:
            raise KeyError

        if frame_num in self._idc_dict[key]:
            return self._idc_dict[key][frame_num]
        else:
            return None

    def get_keypoints(self, key: str, frame_num: int) -> Any:
        if key not in body:
            raise KeyError

        if frame_num in self._kps_dict:
            return self._kps_dict[frame_num][body[key]][:2]
        else:
            return None

    def to_json(self, frame_num: int) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        data["frame"] = frame_num
        data["id"] = self.id
        data["keypoints"] = self._kps_dict[frame_num]

        for k in self._keys:
            indicator = self._idc_dict[k][frame_num]
            if indicator is not None:
                data[k] = indicator.tolist()
            else:
                data[k] = None

        return data
