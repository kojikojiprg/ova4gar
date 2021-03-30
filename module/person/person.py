from common.keypoint import Keypoints, KeypointsList
from common.json import PERSON_FORMAT
from person.indicator import INDICATOR_DICT
import cv2


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
        self.vector_setting_lst = [
            # arrow_length, color, tip_length
            [20, (255, 0, 0), 1.0],
            [30, (0, 0, 255), 1.5],
        ]

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

    def display_tracking(self, frame_num, frame):
        idx = frame_num - self.start_frame_num
        if idx < 0:
            return frame

        keypoints = self.keypoints_lst[idx]
        if keypoints is None:
            return frame

        point = keypoints.get_middle('Hip')
        cv2.circle(frame, tuple(point), 7, (0, 0, 255), thickness=-1)
        cv2.putText(frame, str(self.id), tuple(point),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return frame

    def display_indicator(self, frame_num, field):
        idx = frame_num - self.start_frame_num
        if idx < 0:
            return field

        keypoints = self.keypoints_lst[idx]
        if keypoints is None:
            return field

        position = list(self.indicator_dict.values())[0][idx]

        # face vector, body vector
        for i, v in enumerate(list(self.indicator_dict.values())[:2]):
            data = v[idx]
            if data is None:
                continue

            arrow_length = self.vector_setting_lst[i][0]

            # 矢印の先端の座標を計算
            end = (position + (data * arrow_length)).astype(int)

            color = self.vector_setting_lst[i][1]
            tip_length = self.vector_setting_lst[i][2]
            cv2.arrowedLine(field, tuple(position), tuple(end), color, tipLength=tip_length)

        # wrist
        v = list(self.indicator_dict.values())[2]
        data = v[idx]
        if data is not None:
            lwrist = data[:3]
            rwrist = data[3:]

            radius = 3
            color = (0, 255, 0)
            cv2.circle(field, tuple(lwrist[:2].astype(int)), radius, color, thickness=-1)
            cv2.circle(field, tuple(rwrist[:2].astype(int)), radius, color, thickness=-1)

        return field
