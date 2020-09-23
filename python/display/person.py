from common.keypoint import KeypointsList


class Person:
    def __init__(self, person_id, frame_num):
        self.id = person_id
        self.start_frame_num = frame_num

        self.keypoints_lst = KeypointsList()
        self.average_lst = []
        self.vector_lst = []

    def append(self, tracking_data):
        i = 2
        self.keypoints_lst.append(tracking_data[i])
        self.average_lst.append(tracking_data[i + 1])
        self.vector_lst.append(tracking_data[i + 2])
