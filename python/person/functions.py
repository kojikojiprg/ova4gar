from common import keypoint as kp
import numpy as np


def calc_face_vector(keypoints):
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


def calc_body_vector(keypoints):
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


FUNC_DICT = {
    'face vector': calc_face_vector,
    'body vector': calc_body_vector,
}
