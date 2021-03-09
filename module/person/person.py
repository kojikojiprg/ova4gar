from common import keypoint as kp
from person.indicator import INDICATOR_DICT
import numpy as np
import cv2


class Person:
    def __init__(self, person_id, start_frame_num, homo):
        self.id = person_id
        self.start_frame_num = start_frame_num
        self.keypoints_lst = kp.KeypointsList()
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}

        self.homo = homo
        self.setting_lst = [
            # arrow_length, color, tip_length
            [20, (255, 0, 0), 1.0],
            [30, (0, 0, 255), 1.5]
        ]

    def append_calc(self, keypoints):
        if keypoints is None:
            return

        self.keypoints_lst.append(keypoints)
        for k in self.indicator_dict.keys():
            if keypoints.shape == (17, 3):
                keypoints_tmp = kp.Keypoints(keypoints)
                self.indicator_dict[k].append(INDICATOR_DICT[k](keypoints_tmp, self.homo))
            else:
                self.indicator_dict[k].append(np.nan)

    def append_data(self, data):
        self.keypoints_lst.append(data[2])
        for i, k in enumerate(INDICATOR_DICT.keys()):
            self.indicator_dict[k].append(data[3 + i])

    def get_data(self, frame_num, is_keypoints_numpy=True):
        idx = frame_num - self.start_frame_num
        if idx < 0 or len(self.keypoints_lst) <= idx:
            return None

        if is_keypoints_numpy:
            data = [self.id, frame_num, np.array(self.keypoints_lst[idx])]
        else:
            data = [self.id, frame_num, self.keypoints_lst[idx]]
        for k in self.indicator_dict.keys():
            data.append(self.indicator_dict[k][idx])

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

    def display_vector(self, frame_num, field):
        idx = frame_num - self.start_frame_num
        if idx < 0:
            return field

        keypoints = self.keypoints_lst[idx]
        if keypoints is None:
            return field

        for i, v in enumerate(self.indicator_dict.values()):
            data = v[idx]
            if data is None:
                continue

            arrow_length = self.setting_lst[i][0]

            start = keypoints.get_middle('Ankle')

            # ホモグラフィ変換
            start = self.homo.transform_point(start)

            # 矢印の先端の座標を計算
            end = (start + (data * arrow_length)).astype(int)

            color = self.setting_lst[i][1]
            tip_length = self.setting_lst[i][2]
            cv2.arrowedLine(field, tuple(start), tuple(end), color, tipLength=tip_length)

        return field
