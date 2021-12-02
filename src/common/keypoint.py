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

THRESHOLD_CONFIDENCE = 0.0


class Keypoints(list):
    def __init__(self, keypoints):
        super().__init__([])
        keypoints = np.array(keypoints)
        if keypoints.shape != (17, 3):
            keypoints = keypoints.reshape(17, 3)

        for keypoint in keypoints:
            super().append(keypoint)

    def get(self, body_name, ignore_confidence=False):
        if ignore_confidence:
            return self[body[body_name]][:2].copy()
        else:
            return self[body[body_name]].copy()

    def get_middle(self, name, th_conf=THRESHOLD_CONFIDENCE):
        R = self.get("R" + name)
        L = self.get("L" + name)
        if R[2] < th_conf and L[2] < th_conf:
            return None
        elif R[2] < th_conf:
            point = L
        elif L[2] < th_conf:
            point = R
        else:
            point = (R + L) / 2
        return point[:2].astype(int)

    def to_json(self):
        ret = []
        for keypoint in self:
            ret += keypoint.tolist()
        return ret


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
        if type(keypoints) == Keypoints:
            super().append(keypoints)
        elif keypoints is None:
            super().append(None)
        else:
            if keypoints.shape == (17, 3) or keypoints.shape == (51,):
                super().append(Keypoints(keypoints))
            else:
                super().append(None)
