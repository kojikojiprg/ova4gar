from common.keypoint import Keypoints, KeypointsList
from common.json import IA_FORMAT
from individual_activity.indicator import INDICATOR_DICT


class IndividualActivity:
    def __init__(self, activity_id, start_frame_num, homo):
        self.id = activity_id
        self.start_frame_num = start_frame_num
        self.keypoints_lst = KeypointsList()
        self.vector_lst = []
        self.average_lst = []
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}
        self.position_que = []
        self.body_que = []
        self.homo = homo

    def calc_indicator(self, keypoints, vector, average):
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
            if k == 'position':
                # position
                indicator, self.position_que = INDICATOR_DICT[k](
                    self.keypoints_lst[-1], self.homo, self.position_que)
            elif k == 'body_vector':
                # body_vector
                indicator, self.body_que = INDICATOR_DICT[k](
                    self.keypoints_lst[-1], self.homo, self.body_que)
            else:
                # face vector ~
                indicator = INDICATOR_DICT[k](
                    self.keypoints_lst[-1], self.homo)

            self.indicator_dict[k].append(indicator)

    def get_data(self, key, frame_num):
        if key not in IA_FORMAT:
            raise KeyError

        idx = frame_num - self.start_frame_num
        if key == 'keypoints':
            return self.keypoints_lst[idx]
        else:
            return self.indicator_dict[key][idx]

    def to_json(self, frame_num):
        idx = frame_num - self.start_frame_num
        if idx < 0 or len(self.keypoints_lst) <= idx:
            return None

        data = {}
        data[IA_FORMAT[0]] = self.id
        data[IA_FORMAT[1]] = frame_num
        if self.keypoints_lst[idx] is not None:
            data[IA_FORMAT[2]] = self.keypoints_lst[idx].to_json()
        else:
            data[IA_FORMAT[2]] = None

        for k in IA_FORMAT[3:]:
            indicator = self.indicator_dict[k][idx]
            if indicator is not None:
                data[k] = indicator.tolist()
            else:
                data[k] = None

        return data
