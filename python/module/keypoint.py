import json
import numpy as np

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

confidence_th = 0.00001


class Keypoints(list):
    def __init__(self, keypoint):
        super().__init__([])
        for i in range(0, len(keypoint), 3):
            self.append([
                keypoint[i],
                keypoint[i + 1],
                keypoint[i + 2]])

    def get(self, body_name, ignore_confidence=False):
        if ignore_confidence:
            return np.array(self[body[body_name]])[:2]
        else:
            return np.array(self[body[body_name]])

    def get_middle(self, name):
        R = self.get('R' + name)
        L = self.get('L' + name)
        if R[2] < confidence_th:
            point = L
        elif L[2] < confidence_th:
            point = R
        else:
            point = (R + L) / 2
        return point[:2].astype(int)


class KeypointsList(list):
    def __init__(self, keypoints_lst=None):
        super().__init__([])
        if keypoints_lst is not None:
            for keypoints in keypoints_lst:
                self.append(Keypoints(keypoints))

    def get_middle_points(self, name):
        points = []
        for keypoints in self:
            if keypoints is not None:
                points.append(keypoints.get_middle(name))
            else:
                points.append(None)

        return points


class Frame(list):
    def __init__(self, json_path):
        super().__init__([])
        with open(json_path) as f:
            dat = json.load(f)

            keypoints_lst = []
            pre_no = 0
            for item in dat:
                frame_no = int(item['image_id'].split('.')[0])

                if frame_no != pre_no:
                    self.append(KeypointsList(keypoints_lst))
                    keypoints_lst = []

                keypoints_lst.append(item['keypoints'])
                pre_no = frame_no