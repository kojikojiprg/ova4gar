from common.keypoint import Keypoints, KeypointsList
from common.json import IA_FORMAT
from individual_activity.indicator import INDICATOR_DICT


class IndividualActivity:
    def __init__(self, activity_id, homo):
        self.id = activity_id
        self.keypoints_lst = KeypointsList()
        self.vector_lst = []
        self.average_lst = []
        self.indicator_dict = {k: {} for k in INDICATOR_DICT.keys()}
        self.que_dict = {k: [] for k in INDICATOR_DICT.keys()}
        self.homo = homo

    def calc_indicator(self, frame_num, keypoints, vector, average):
        if keypoints is None:
            self.keypoints_lst.append(None)
            self.vector_lst.append(None)
            self.average_lst.append(None)
            for v in self.indicator_dict.values():
                v.append(None)
            return

        self.keypoints_lst.append(Keypoints(keypoints))
        self.vector_lst.append(vector)
        self.average_lst.append(average)

        for k in self.indicator_dict.keys():
            indicator, self.que_dict[k] = INDICATOR_DICT[k](
                self.keypoints_lst[-1], self.homo, self.que_dict[k])

            self.indicator_dict[k][frame_num] = indicator

    def get_data(self, key, frame_num):
        if key not in IA_FORMAT:
            raise KeyError

        if key == 'keypoints':
            try:
                return self.keypoints_lst[frame_num]
            except IndexError:
                return None
        else:
            try:
                return self.indicator_dict[key][frame_num]
            except IndexError:
                return None

    def to_json(self, frame_num):
        data = {}
        data[IA_FORMAT[0]] = self.id
        data[IA_FORMAT[1]] = frame_num

        for k in IA_FORMAT[2:]:
            indicator = self.indicator_dict[k][frame_num]
            if indicator is not None:
                data[k] = indicator.tolist()
            else:
                data[k] = None

        return data
