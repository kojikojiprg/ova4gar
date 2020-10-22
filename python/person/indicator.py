from common import keypoint as kp
import numpy as np


def calc_face_vector(keypoints):
    nose = keypoints.get('Nose')
    lear = keypoints.get('LEar')
    rear = keypoints.get('REar')

    if lear[2] < kp.confidence_th and rear[2] < kp.confidence_th:
        angle = np.nan
    elif lear[2] < kp.confidence_th:
        diff = nose - rear
        angle = np.arctan2(diff[1], diff[0])
    elif rear[2] < kp.confidence_th:
        diff = nose - lear
        angle = np.arctan2(diff[1], diff[0])
    else:
        diff = rear - lear
        angle = np.arctan2(diff[1], diff[0])
        if diff[0] <= 0:
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
        diff = rshoulder - lshoulder
        angle = np.arctan2(diff[1], diff[0])
        if diff[0] <= 0:
            angle += np.pi / 2
        else:
            angle -= np.pi / 2

    return angle


INDICATOR_DICT = {
    'face vector': calc_face_vector,
    'body vector': calc_body_vector,
}
