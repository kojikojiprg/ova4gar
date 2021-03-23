from common import keypoint as kp
import numpy as np


def normalize(vec):
    vec += 1e-10
    vec /= np.linalg.norm(vec)
    return vec


def rotation(vec, rad):
    R = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]])

    return np.dot(R, vec)


def calc_face_vector(keypoints, homo):
    nose = keypoints.get('Nose')
    lear = keypoints.get('LEar')
    rear = keypoints.get('REar')

    # ポイントを足元に反映
    diff = keypoints.get_middle('Ankle') - keypoints.get_middle('Ear')
    nose[1] += diff[1]
    lear[1] += diff[1]
    rear[1] += diff[1]

    # ホモグラフィ変換
    nose = np.append(homo.transform_point(nose[:2]), nose[2])
    lear = np.append(homo.transform_point(lear[:2]), lear[2])
    rear = np.append(homo.transform_point(rear[:2]), rear[2])

    if lear[2] < kp.confidence_th and nose[2] >= kp.confidence_th:
        vector = nose - rear
        vector = vector[:2]
        vector = normalize(vector)
    elif rear[2] < kp.confidence_th and nose[2] >= kp.confidence_th:
        vector = nose - lear
        vector = vector[:2]
        vector = normalize(vector)
    elif rear[2] >= kp.confidence_th and lear[2] >= kp.confidence_th:
        vector = lear - rear
        vector = vector[:2]
        vector = normalize(vector)
        vector = rotation(vector, np.pi / 2)
    else:
        vector = np.nan

    return vector


def calc_body_vector(keypoints, homo):
    lshoulder = keypoints.get('LShoulder')
    rshoulder = keypoints.get('RShoulder')

    # ポイントを足元に反映
    diff = keypoints.get_middle('Ankle') - keypoints.get_middle('Shoulder')
    lshoulder[1] += diff[1]
    rshoulder[1] += diff[1]

    # ホモグラフィ変換
    lshoulder = np.append(homo.transform_point(lshoulder[:2]), lshoulder[2])
    rshoulder = np.append(homo.transform_point(rshoulder[:2]), rshoulder[2])

    if lshoulder[2] >= kp.confidence_th and rshoulder[2] >= kp.confidence_th:
        vector = lshoulder - rshoulder
        vector = vector[:2]
        vector = normalize(vector)
        vector = rotation(vector, np.pi / 2)
    else:
        vector = np.nan

    return vector


def calc_wrist(keypoints, homo):
    def calc(keypoints, lr):
        wrist = keypoints.get(lr + 'Wrist')

        # ポイントを足元に反映
        diff = keypoints.get_middle('Ankle') - wrist[:2]
        wrist[1] += diff[1]

        # ホモグラフィ変換
        wrist = np.append(homo.transform_point(wrist[:2]), wrist[2])

        return wrist

    ret = np.append(calc(keypoints, 'L'), calc(keypoints, 'R'))
    return ret


INDICATOR_DICT = {
    'face vector': calc_face_vector,
    'body vector': calc_body_vector,
    'wrist': calc_wrist,
}
