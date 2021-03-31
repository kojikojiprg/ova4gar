from common.keypoint import Keypoints, KeypointsList
from common.json import PERSON_FORMAT
from person.indicator import INDICATOR_DICT


class Person:
    def __init__(self, person_id, start_frame_num, homo):
        self.id = person_id
        self.start_frame_num = start_frame_num
        self.keypoints_lst = KeypointsList()
        self.vector_lst = []
        self.average_lst = []
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}
        self.position_que = []
        self.homo = homo

    def calc_indicator(self, keypoints, vector, average):
        if keypoints is None:
            return

        self.keypoints_lst.append(Keypoints(keypoints))
        self.vector_lst.append(vector)
        self.average_lst.append(average)

        for k in self.indicator_dict.keys():
            if k == 'position':
                # position
                indicator = INDICATOR_DICT[k](
                    self.keypoints_lst[-1], self.average_lst[-1], self.position_que, self.homo)
            else:
                # face vector ~
                indicator = INDICATOR_DICT[k](self.keypoints_lst[-1], self.homo)

            self.indicator_dict[k].append(indicator)

    def to_json(self, frame_num):
        idx = frame_num - self.start_frame_num
        if idx < 0 or len(self.keypoints_lst) <= idx:
            return None

        data = {}
        data[PERSON_FORMAT[0]] = self.id
        data[PERSON_FORMAT[1]] = frame_num
        data[PERSON_FORMAT[2]] = self.keypoints_lst[idx].to_json()
        for k in PERSON_FORMAT[3:]:
            indicator = self.indicator_dict[k][idx]
            if indicator is not None:
                data[k] = indicator.tolist()
            else:
                data[k] = None

        return data
