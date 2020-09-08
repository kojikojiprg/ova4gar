from common.keypoint import KeypointsList
import numpy as np


class Person:
    def __init__(self, person_id, frame_num):
        self.id = person_id

        self.keypoints_lst = KeypointsList()
        self.vector_lst = []

        if frame_num > 0:
            self.keypoints_lst.append(np.array(np.nan))
            self.vector_lst.append(None)

    def append(self, keypoints, vector):
        self.keypoints_lst.append(keypoints)
        self.vector_lst.append(vector)
