from common.json import PERSON_FORMAT
from common import keypoint as kp
from common.functions import normalize_vector, rotation
import numpy as np


def calc_position(keypoints, average, position_que, homo, size=10):
    new_pos = keypoints.get_middle('Ankle')
    if new_pos is None:
        if len(position_que) == 0:
            # 初期状態では何も追加しない
            return
        else:
            # 一つ前の値を追加する
            new_pos = position_que[-1]

    new_pos = homo.transform_point(new_pos)
    position_que.append(new_pos)

    if len(position_que) <= size:
        pos = np.average(position_que[:len(position_que)], axis=0)
    else:
        position_que = position_que[-size:]
        pos = np.average(position_que, axis=0)

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
        vector = normalize_vector(vector)
    elif rear[2] < kp.confidence_th and nose[2] >= kp.confidence_th:
        vector = nose - lear
        vector = vector[:2]
        vector = normalize_vector(vector)
    elif rear[2] >= kp.confidence_th and lear[2] >= kp.confidence_th:
        vector = lear - rear
        vector = vector[:2]
        vector = normalize_vector(vector)
        vector = rotation(vector, np.pi / 2)
    else:
        vector = None

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
        vector = normalize_vector(vector)
        vector = rotation(vector, np.pi / 2)
    else:
        vector = None

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
