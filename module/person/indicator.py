from common.default import POSITION_DEFAULT
from common.json import PERSON_FORMAT
from common import keypoint as kp
from common.functions import normalize_vector, cos_similarity, rotation
import numpy as np


def calc_position(
    keypoints, average, position_que, homo,
    size=POSITION_DEFAULT['size'], ratio=POSITION_DEFAULT['ratio']
):
    new_pos = keypoints.get_middle('Ankle')
    if new_pos is None:
        shoulder = keypoints.get_middle('Shoulder')
        hip = keypoints.get_middle('Hip')

        if shoulder is None or hip is None:
            return None
        else:
            body_line = hip - shoulder
            new_pos = hip + body_line * ratio

    new_pos = homo.transform_point(new_pos)
    position_que.append(new_pos)

    if len(position_que) <= size:
        pos = np.average(position_que, axis=0)
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

    if lear[2] < kp.THRESHOLD_CONFIDENCE and nose[2] >= kp.THRESHOLD_CONFIDENCE:
        vector = nose - rear
        vector = vector[:2]
        vector = normalize_vector(vector)
    elif rear[2] < kp.THRESHOLD_CONFIDENCE and nose[2] >= kp.THRESHOLD_CONFIDENCE:
        vector = nose - lear
        vector = vector[:2]
        vector = normalize_vector(vector)
    elif rear[2] >= kp.THRESHOLD_CONFIDENCE and lear[2] >= kp.THRESHOLD_CONFIDENCE:
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

    if lshoulder[2] >= kp.THRESHOLD_CONFIDENCE and rshoulder[2] >= kp.THRESHOLD_CONFIDENCE:
        vector = lshoulder - rshoulder
        vector = vector[:2]
        vector = normalize_vector(vector)
        vector = rotation(vector, np.pi / 2)
    else:
        vector = None

    return vector


def calc_arm_extention(keypoints, homo):
    def calc(keypoints, lr):
        shoulder = keypoints.get_middle('Shoulder')
        hip = keypoints.get_middle('Hip')

        if shoulder is None or hip is None:
            return None
        else:
            body_line = hip - shoulder
            arm = keypoints.get(lr + 'Wrist', ignore_confidence=True) \
                - keypoints.get(lr + 'Shoulder', ignore_confidence=True)

            return 1.0 - np.abs(cos_similarity(body_line, arm))  # cos to sin

    larm = calc(keypoints, 'L')
    rarm = calc(keypoints, 'R')
    if larm is None and rarm is None:
        return None
    elif larm is None and rarm is not None:
        return rarm
    elif larm is not None and rarm is None:
        return larm
    else:
        return np.max((larm, rarm))


start_idx = 3
INDICATOR_DICT = {
    PERSON_FORMAT[start_idx + 0]: calc_position,
    PERSON_FORMAT[start_idx + 1]: calc_face_vector,
    PERSON_FORMAT[start_idx + 2]: calc_body_vector,
    PERSON_FORMAT[start_idx + 3]: calc_arm_extention,
}
