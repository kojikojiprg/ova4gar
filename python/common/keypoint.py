import json
import numpy as np
from common import common

body = {
    "Nose": 0,
    "LEye": 1,
    "REye": 2,
    "LEar": 3,
    "REar": 4,
    "LShoulder": 5,
    "RShoulder": 6,
    "LElbow": 7,
    "RElbow": 8,
    "LWrist": 9,
    "RWrist": 10,
    "LHip": 11,
    "RHip": 12,
    "LKnee": 13,
    "RKnee": 14,
    "LAnkle": 15,
    "RAnkle": 16,
}

confidence_th = 0.2


class Keypoints(list):
    def __init__(self, keypoints):
        super().__init__([])
        for keypoint in keypoints:
            super().append(keypoint)

    def get(self, body_name, ignore_confidence=False):
        if ignore_confidence:
            return self[body[body_name]][:2]
        else:
            return self[body[body_name]]

    def get_middle(self, name):
        R = self.get('R' + name)
        L = self.get('L' + name)
        if R[2] < confidence_th:
            point = L
        elif L[2] < confidence_th:
            point = R
        elif R[2] < confidence_th and L[2] < confidence_th:
            return None
        else:
            point = (R + L) / 2
        return point[:2].astype(int)


class KeypointsList(list):
    def __init__(self):
        super().__init__([])

    def get_middle_points(self, name):
        points = []
        for keypoints in self:
            if keypoints is not None:
                points.append(keypoints.get_middle(name))
            else:
                points.append(None)

        return points

    def append(self, keypoints):
        if keypoints is not None:
            super().append(Keypoints(keypoints))
        else:
            super().append(None)


def read_json(json_path):
    return_lst = []
    with open(json_path) as f:
        dat = json.load(f)

        keypoints_lst = KeypointsList()
        pre_no = 0
        for item in dat:
            frame_no = int(item['image_id'].split('.')[0])

            if frame_no != pre_no:
                return_lst.append(keypoints_lst)
                keypoints_lst = KeypointsList()

            keypoints_lst.append(np.array(item['keypoints']).reshape(17, 3))
            pre_no = frame_no
        else:
            return_lst.append(keypoints_lst)

    return return_lst


def read_sql(db):
    datas = db.select(common.TRACKING_TABLE_NAME)

    persons = []
    frames = []
    for row in datas:
        person_id = row[0]
        frame_num = row[1]
        keypoints = row[2]

        if len(persons) == person_id:
            persons.append(KeypointsList())

        if len(frames) == frame_num:
            frames.append(KeypointsList())

        if keypoints.shape == (17, 3):
            keypoints = keypoints.flatten()
            persons[person_id].append(Keypoints(keypoints))
            frames[frame_num].append(Keypoints(keypoints))
        else:
            persons[person_id].append(None)
            frames[frame_num].append(None)

    return persons, frames
