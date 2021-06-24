from common.default import POSITION_DEFAULT, FACE_DEFAULT, BODY_DEFAULT, ARM_DEFAULT
from common.json import IA_FORMAT
from common import keypoint as kp
from common.functions import mahalanobis, normalize_vector, cos_similarity, rotation
import numpy as np


def calc_ma(que, std_th):
    que = np.array(que)

    if np.any(np.std(que, axis=0) < 1.0):
        # 分布の中身がほぼ同じのとき
        val = np.average(que, axis=0)
    else:
        if que.ndim < 2:
            # 各点の中心からの距離を求める
            mean = np.average(que)
            distances = np.abs(que - mean)
        else:
            # 各点の中心からのマハラノビス距離を求める
            distances = [mahalanobis(x, que) for x in que]

        # 中心からの距離の平均と分散を求める
        mean = np.average(distances)
        std = np.std(distances)

        # 外れ値を除去した平均値を値とする
        val = np.average(
            que[np.abs(distances - mean) < std * std_th], axis=0)

    return val


def calc_position(
    keypoints, homo, position_que,
    ankle_th=POSITION_DEFAULT['ankle_th'],
    size=POSITION_DEFAULT['size'], ratio=POSITION_DEFAULT['ratio'],
    std_th=POSITION_DEFAULT['std_th']
):
    new_pos = keypoints.get_middle('Ankle', th_conf=ankle_th)
    if new_pos is None:
        shoulder = keypoints.get_middle('Shoulder')
        hip = keypoints.get_middle('Hip')

        if shoulder is None or hip is None:
            return None, position_que
        else:
            body_line = hip - shoulder
            new_pos = hip + body_line * ratio

    new_pos = homo.transform_point(new_pos)
    position_que.append(new_pos)

    if len(position_que) < size:
        pos = np.average(position_que, axis=0)
    else:
        position_que = position_que[-size:]
        pos = calc_ma(position_que, std_th)

    return pos.astype(int), position_que


def calc_face_vector(
    keypoints, homo, face_que,
    size=FACE_DEFAULT['size'], ratio=FACE_DEFAULT['ratio'],
    std_th=FACE_DEFAULT['std_th']
):
    nose = keypoints.get('Nose')
    lear = keypoints.get('LEar')
    rear = keypoints.get('REar')

    # 足元にポイントを落とす
    ankle = keypoints.get_middle('Ankle')
    ear = keypoints.get_middle('Ear')
    if ankle is not None and ear is not None:
        diff = ankle[:2] - ear
    else:
        shoulder = keypoints.get_middle('Shoulder')
        hip = keypoints.get_middle('Hip')
        if shoulder is None or hip is None:
            return None, face_que

        diff = hip - shoulder
        diff = diff.astype(float) * ratio

    nose[:2] += diff
    lear[:2] += diff
    rear[:2] += diff

    # ホモグラフィ変換
    nose_homo = np.append(homo.transform_point(nose[:2]), nose[2])
    lear_homo = np.append(homo.transform_point(lear[:2]), lear[2])
    rear_homo = np.append(homo.transform_point(rear[:2]), rear[2])

    if lear[0] > rear[0]:
        x1 = rear[0]
        x2 = lear[0]
    else:
        x1 = lear[0]
        x2 = rear[0]

    if x1 < nose[0] and nose[0] < x2:
        new_vector = rear_homo - lear_homo
        new_vector = rotation(new_vector[:2], - np.pi / 2)
    else:
        center_ear = rear_homo + (lear_homo - rear_homo) / 2
        new_vector = nose_homo - center_ear
        new_vector = new_vector[:2]

    face_que.append(new_vector)
    if len(face_que) < size:
        vector = np.average(face_que, axis=0)
    else:
        face_que = face_que[-size:]
        vector = calc_ma(face_que, std_th)

    vector = normalize_vector(vector)
    return vector, face_que


def calc_body_vector(
    keypoints, homo, body_que,
    size=BODY_DEFAULT['size'], ratio=BODY_DEFAULT['ratio'],
    std_th=BODY_DEFAULT['std_th']
):
    lshoulder = keypoints.get('LShoulder')
    rshoulder = keypoints.get('RShoulder')
    if lshoulder[2] < kp.THRESHOLD_CONFIDENCE or rshoulder[2] < kp.THRESHOLD_CONFIDENCE:
        return None, body_que

    # 足元にポイントを落とす
    shoulder = keypoints.get_middle('Shoulder')
    hip = keypoints.get_middle('Hip')
    if shoulder is None or hip is None:
        return None, body_que

    diff = hip - shoulder
    diff = diff.astype(float) * ratio
    lshoulder[:2] += diff
    rshoulder[:2] += diff

    # ホモグラフィ変換
    lshoulder = homo.transform_point(lshoulder[:2])
    rshoulder = homo.transform_point(rshoulder[:2])

    new_vector = rshoulder - lshoulder
    new_vector = rotation(new_vector[:2], - np.pi / 2)

    body_que.append(new_vector)
    if len(body_que) < size:
        vector = np.average(body_que, axis=0)
    else:
        body_que = body_que[-size:]
        vector = calc_ma(body_que, std_th)

    vector = normalize_vector(vector)
    return vector, body_que


def calc_arm_extention(
    keypoints, homo, arm_que,
    size=ARM_DEFAULT['size'], std_th=ARM_DEFAULT['std_th']
):
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
        return None, arm_que
    elif larm is None and rarm is not None:
        new_arm = rarm
    elif larm is not None and rarm is None:
        new_arm = larm
    else:
        new_arm = np.max((larm, rarm))

    arm_que.append(new_arm)
    if len(arm_que) < size:
        arm = np.average(arm_que)
    else:
        arm_que = arm_que[-size:]
        arm = calc_ma(arm_que, std_th)

    return arm, arm_que


start_idx = 3
INDICATOR_DICT = {
    IA_FORMAT[start_idx + 0]: calc_position,
    IA_FORMAT[start_idx + 1]: calc_face_vector,
    IA_FORMAT[start_idx + 2]: calc_body_vector,
    IA_FORMAT[start_idx + 3]: calc_arm_extention,
}
