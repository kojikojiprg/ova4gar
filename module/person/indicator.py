from common.json import PERSON_FORMAT
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


def calc_position(keypoints, average, position_que, homo, size=5):
    new_pos = keypoints.get_middle('Ankle')
    if new_pos is None:
        shoulder = keypoints.get_middle('Shoulder')
        diff = average - shoulder
        new_pos = (average + diff * 1.5).astype(int)

    new_pos = homo.transform_point(new_pos)
    position_que.append(new_pos)

    if len(position_que) <= size:
        pos = np.average(position_que[:len(position_que)])
    else:
        position_que = position_que[-size:]
        pos = np.average(position_que)

    return pos.astype(int)


def calc_face_vector(keypoints, homo):
    nose = keypoints.get('Nose')
    lear = keypoints.get('LEar')
    rear = keypoints.get('REar')

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


start_idx = 3
INDICATOR_DICT = {
    PERSON_FORMAT[start_idx + 0]: calc_position,
    PERSON_FORMAT[start_idx + 1]: calc_face_vector,
    PERSON_FORMAT[start_idx + 2]: calc_body_vector,
    PERSON_FORMAT[start_idx + 3]: calc_wrist,
}
