from common import keypoint as kp
import numpy as np


class Model:
    FUNC_DICT = {
        'face vector': eval('self.calc_face_vector'),
        'body vector': eval('self.calc_body_vector'),
    }

    def __init__(self, start_frame_num):
        self.start_frame_num = start_frame_num
        self.keypoints_lst = kp.KeypointsList()
        self.data_dict = {k: [] for k in self.FUNC_DICT.keys()}

    def append(self, keypoints):
        self.keypoints_lst.append(kp.Keypoints(keypoints))
        for k in self.data_dict.keys():
            if keypoints is not None:
                self.data_dict[k].append(self.FUNC_DICT[k](keypoints))
            else:
                self.data_dict[k].append(np.nan)

    def calc_face_vector(self, keypoints):
        nose = keypoints.get('Nose')
        lear = keypoints.get('LEar')
        rear = keypoints.get('REar')

        if lear[2] < kp.confidence_th:
            diff = nose - rear
            angle = np.arctan(diff[2])
        elif rear[2] < kp.confidence_th:
            diff = lear - nose
            angle = np.arctan(diff[2]) + np.pi
        elif lear[2] < kp.confidence_th and rear[2] < kp.confidence_th:
            angle = np.nan
        else:
            diff = lear - rear
            angle = np.arctan(diff[2])
            if diff[0] >= 0:
                angle += np.pi / 2
            else:
                angle -= np.pi / 2

        return angle

    def clac_body_vector(self, keypoints):
        lshoulder = keypoints.get('LShoulder')
        rshoulder = keypoints.get('RShoulder')

        if lshoulder[2] < kp.confidence_th or rshoulder[2] < kp.confidence_th:
            angle = np.nan
        else:
            diff = lshoulder - rshoulder
            angle = np.arctan(diff[2])
            if diff[0] >= 0:
                angle += np.pi / 2
            else:
                angle -= np.pi / 2

        return angle
