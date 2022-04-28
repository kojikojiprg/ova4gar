from typing import Any, Dict, Union

import numpy as np
from keypoint.keypoint import PARTS, Keypoints
from utility.transform import Homography

from individual.indicator import arm, body, face, position
from individual.individual_que import KeypointQue, Que


class Individual:
    def __init__(self, pid: int, defaults: dict):
        self.id = pid

        self._defs = defaults["indicator"]
        self._keys = list(self._defs.keys())
        self._funcs = {k: eval(k) for k in self._keys}
        self._pre_frame_num: int = 0

        self._kps_dict: Dict[int, Keypoints] = {}
        self._kps_que: KeypointQue = KeypointQue(defaults["keypoint"])
        self._idc_dict: Dict[str, Any] = {k: {} for k in self._keys}
        self._idc_que: Dict[str, Que] = {k: Que(self._defs[k]) for k in self._keys}

    def calc_indicator(self, frame_num: int, kps: Any, homo: Homography):
        # calc keypoints
        if kps is None:
            return
        kps = Keypoints(kps)
        kps, self._kps_dict = self._kps_que.put_pop_kps(
            frame_num, self._pre_frame_num, kps, self._kps_dict
        )

        # calc indicators
        for k in self._keys:
            val = self._funcs[k](kps, homo, self._idc_que[k], self._defs[k])

            self._idc_dict[k][frame_num] = val

        # update pre frame num
        self._pre_frame_num = frame_num

    def get_indicator(self, key: str, frame_num: int) -> Union[Any, None]:
        if key not in self._keys:
            raise KeyError

        if frame_num in self._idc_dict[key]:
            return self._idc_dict[key][frame_num]
        else:
            return None

    def get_keypoints(
        self, key: str, frame_num: int, ignore_confidence: bool = True
    ) -> Union[np.typing.NDArray, None]:
        if key not in PARTS:
            raise KeyError

        if frame_num in self._kps_dict:
            return self._kps_dict[frame_num].get(
                key, ignore_confidence=ignore_confidence
            )
        else:
            return None

    def exists_on_frame(self, frame_num: int):
        return frame_num in self._idc_dict["position"]

    def to_dict(self, frame_num: int) -> Union[Dict[str, Any], None]:
        data: Dict[str, Any] = {}
        data["frame"] = frame_num
        data["id"] = self.id

        if frame_num not in self._kps_dict:
            return None

        data["keypoints"] = self._kps_dict[frame_num]

        for k in self._keys:
            indicator = self._idc_dict[k][frame_num]
            if type(indicator) == np.array:
                data[k] = indicator.tolist()
            elif indicator is not None:
                data[k] = indicator
            else:
                data[k] = None

        return data

    def from_json(self, json_data: Dict[str, Any], frame_num: int):
        for k, v in json_data.items():
            if k in self._keys:
                self._idc_dict[k][frame_num] = v
            elif k == "keypoints":
                self._kps_dict[frame_num] = Keypoints(v)
