from common.keypoint import KeypointsList


class Person:
    def __init__(self, person_id, frame_num):
        self.id = person_id
        self.start_frame_num = frame_num

        self.keypoints_lst = KeypointsList()
        self.vector_lst = []

    def append(self, keypoints, vector):
        self.keypoints_lst.append(keypoints)
        self.vector_lst.append(vector)
