from common.keypoint import KeypointsList


class Frame:
    def __init__(self, frame_num):
        self.num = frame_num
        self.keypoints_lst = KeypointsList()

    def append(self, keypoints):
        self.keypoints_lst.append(keypoints)
