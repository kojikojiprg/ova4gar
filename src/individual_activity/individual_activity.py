from common.keypoint import Keypoints
from common.json import IA_FORMAT
from individual_activity.indicator import INDICATOR_DICT


class IndividualActivity:
    def __init__(self, activity_id, homo=None):
        self.id = activity_id
        self.tracking_points = {}
        self.indicator_dict = {k: {} for k in INDICATOR_DICT.keys()}
        self.que_dict = {k: [] for k in INDICATOR_DICT.keys()}
        self.homo = homo

    def calc_indicator(self, frame_num, keypoints):
        if keypoints is None:
            return
        keypoints = Keypoints(keypoints)

        self.tracking_points[frame_num] = keypoints.get_middle('Hip')

        for k in self.indicator_dict.keys():
            indicator, self.que_dict[k] = INDICATOR_DICT[k](
                keypoints, self.homo, self.que_dict[k])

            self.indicator_dict[k][frame_num] = indicator

    def get_data(self, key, frame_num):
        if key not in IA_FORMAT:
            raise KeyError

        if frame_num in self.indicator_dict[key]:
            return self.indicator_dict[key][frame_num]
        else:
            return None

    def to_json(self, frame_num):
        data = {}
        data[IA_FORMAT[0]] = self.id
        data[IA_FORMAT[1]] = frame_num
        data[IA_FORMAT[2]] = self.tracking_points[frame_num]

        for k in IA_FORMAT[3:]:
            indicator = self.indicator_dict[k][frame_num]
            if indicator is not None:
                data[k] = indicator.tolist()
            else:
                data[k] = None

        return data
