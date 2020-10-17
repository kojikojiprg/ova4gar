from common.keypoint import KeypointsList
from person.functions import FUNC_DICT
import numpy as np
import cv2


class Person:
    def __init__(self, person_id, frame_num):
        self.id = person_id
        self.start_frame_num = frame_num

        self.keypoints_lst = KeypointsList()
        self.indicators_dict = {k: [] for k in FUNC_DICT.keys()}
        self.setting_lst = [
            # arrow_length, color, tip_length
            [10, (255, 0, 0), 1.0],
            [15, (0, 0, 255), 1.5]
        ]

    def append(self, data):
        self.keypoints_lst.append(data[2])
        for i, k in enumerate(FUNC_DICT.keys()):
            self.indicators_dict[k].append(data[3 + i])

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

    def display_vector(self, frame_num, field, homo):
        idx = frame_num - self.start_frame_num
        if idx < 0:
            return field

        keypoints = self.keypoints_lst[idx]
        if keypoints is None:
            return field

        for i, v in enumerate(self.indicators_dict.values()):
            data = v[idx]
            if data is None:
                continue

            arrow_length = self.setting_lst[i][0]
            start = keypoints.get_middle('Ankle')
            x = np.cos(data)
            y = np.sin(data)
            end = start + np.array([x, y]) * arrow_length

            # ホモグラフィ変換
            start = homo.transform_point(start)
            end = homo.transform_point(end)

            # ホモグラフィ変換後の矢印の長さを揃える
            ratio = arrow_length / np.linalg.norm(end - start)
            end = start + ((end - start) * ratio).astype(int)

            color = self.setting_lst[i][1]
            tip_length = self.setting_lst[i][2]
            cv2.arrowedLine(field, tuple(start), tuple(end), color, tipLength=tip_length)

        return field
