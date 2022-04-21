from common.json import IA_FORMAT, START_IDX
from common.keypoint import Keypoints, body

from individual_activity.indicator import (
    INDICATOR_DICT,
    calc_new_keypoints,
    fill_nan_keypoints,
)


class Individual:
    def __init__(self, activity_id, homo):
        self.id = activity_id
        self.keypoints = {}
        self.keypoints_que = []
        self.tracking_points = {}
        self.indicator_dict = {k: {} for k in INDICATOR_DICT.keys()}
        self.que_dict = {k: [] for k in INDICATOR_DICT.keys()}
        self.homo = homo

    def calc_indicator(self, frame_num, keypoints):
        # calc keypoints
        if keypoints is None:
            return
        keypoints = Keypoints(keypoints)
        pre_frame_num = 0
        if len(self.keypoints) > 0:
            pre_frame_num = list(self.keypoints.keys())[-1]

        self.keypoints[frame_num], self.keypoints_que = calc_new_keypoints(
            frame_num, pre_frame_num, keypoints, self.keypoints_que
        )
        self.keypoints, self.keypoints_que = fill_nan_keypoints(
            frame_num, pre_frame_num, self.keypoints, self.keypoints_que
        )
        keypoints = Keypoints(self.keypoints[frame_num])

        # calc tracking points
        self.tracking_points[frame_num] = keypoints.get_middle("Hip")

        # calc indicators
        for k in self.indicator_dict.keys():
            indicator, self.que_dict[k] = INDICATOR_DICT[k](
                keypoints, self.homo, self.que_dict[k]
            )

            self.indicator_dict[k][frame_num] = indicator

    def get_indicator(self, key, frame_num):
        if key not in IA_FORMAT:
            raise KeyError

        if frame_num in self.indicator_dict[key]:
            return self.indicator_dict[key][frame_num]
        else:
            return None

    def get_keypoints(self, key, frame_num):
        if key not in body:
            raise KeyError

        if frame_num in self.keypoints:
            return self.keypoints[frame_num][body[key]][:2]
        else:
            return None

    def to_json(self, frame_num):
        data = {}
        data[IA_FORMAT[0]] = self.id
        data[IA_FORMAT[1]] = frame_num
        data[IA_FORMAT[2]] = self.tracking_points[frame_num]
        data[IA_FORMAT[3]] = self.keypoints[frame_num]

        for k in IA_FORMAT[START_IDX:]:
            indicator = self.indicator_dict[k][frame_num]
            if indicator is not None:
                data[k] = indicator.tolist()
            else:
                data[k] = None

        return data
