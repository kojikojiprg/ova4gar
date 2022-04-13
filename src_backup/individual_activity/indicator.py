import numpy as np
from common import keypoint as kp
from common.default import (
    ARM_DEFAULT,
    BODY_DEFAULT,
    FACE_DEFAULT,
    KEYPOINTS_DEFAULT,
    POSITION_DEFAULT,
)
from common.functions import cos_similarity, mahalanobis, normalize_vector, rotation
from common.json import IA_FORMAT, START_IDX


def calc_ma(que, std_th):
    que = np.array(que)

    if np.any(np.std(que, axis=0) < 1.0):
        # 分布の中身がほぼ同じのとき
        val = np.average(que, axis=0)
    else:
        if que.ndim < 2:
            # 各点の中心からの距離を求める
            mean = np.nanmean(que)
            distances = np.abs(que - mean)
        else:
            # 各点の中心からのマハラノビス距離を求める
            distances = [mahalanobis(x, que) for x in que]

        # 中心からの距離の平均と分散を求める
        mean = np.nanmean(distances)
        std = np.std(distances)

        # 外れ値を除去した平均値を値とする
        val = np.average(que[np.abs(distances - mean) < std * std_th], axis=0)

    return val


def calc_new_keypoints(
    frame_num,
    pre_frame_num,
    keypoints,
    que,
    que_size=KEYPOINTS_DEFAULT["size"],
    std_th=KEYPOINTS_DEFAULT["std_th"],
):
    # if confidnce score < THRESHOLD then [np.nan, np.nan, np.nan]
    keypoints = np.array(keypoints)
    mask = np.where(keypoints.T[2] < kp.THRESHOLD_CONFIDENCE)
    nan_array = np.full(keypoints.shape, np.nan)
    keypoints[mask] = nan_array[mask]

    # delete confidence scores
    # keypoints = np.delete(keypoints, 2, 1)

    if frame_num - pre_frame_num >= que_size:
        que = []
    que.append(keypoints)

    # calc mean of que
    if len(que) < que_size:
        new_keypoints = np.nanmean(que, axis=0)
    else:
        que = que[-que_size:]

        new_keypoints = []
        tmp_que = np.transpose(que, (1, 0, 2))
        for i in range(len(keypoints)):
            new_keypoints.append(calc_ma(tmp_que[i], std_th))
        new_keypoints = np.array(new_keypoints)

    return new_keypoints, que


def fill_nan_keypoints(
    frame_num,
    pre_frame_num,
    keypoints_dict,
    que,
    que_size=KEYPOINTS_DEFAULT["size"],
    window=KEYPOINTS_DEFAULT["window"],
):
    assert que_size > window, f"que_size:{que_size} > window:{window} is expected."
    if len(que) < 2:
        return keypoints_dict, que

    copy_kps_lst = []
    pre = np.array(keypoints_dict[pre_frame_num])
    for i in range(pre_frame_num, frame_num + 1):
        if i in keypoints_dict:
            # append current keypoints
            kps = np.array(keypoints_dict[i])
            copy_kps_lst.append(kps)
            pre = kps.copy()
        else:
            # copy and append pre keypoints
            copy_kps_lst.append(pre)

    # fill nan each keypoint
    ma_kps_lst = []
    for i in range(0, len(copy_kps_lst) - window + 1):
        # calc means of all keypoints
        means = np.nanmean(copy_kps_lst[i : i + window], axis=0)

        for kps in copy_kps_lst[i : i + window]:
            if True in np.isnan(kps):
                # fill nan if nan points are included
                kps = np.where(np.isnan(kps), means, kps).copy()
            ma_kps_lst.append(kps)

    ma_kps_dict = {}
    for i, kps in enumerate(ma_kps_lst):
        ma_kps_dict[pre_frame_num + i] = kps
    keypoints_dict.update(ma_kps_dict)

    if len(que) < len(ma_kps_lst):
        que = ma_kps_lst
    else:
        que = ma_kps_lst[-que_size:]

    return keypoints_dict, que


def calc_position(
    keypoints,
    homo,
    position_que,
    ankle_th=POSITION_DEFAULT["ankle_th"],
    size=POSITION_DEFAULT["size"],
    ratio=POSITION_DEFAULT["ratio"],
    std_th=POSITION_DEFAULT["std_th"],
):
    new_pos = keypoints.get_middle("Ankle", th_conf=ankle_th)
    if new_pos is None:
        shoulder = keypoints.get_middle("Shoulder")
        hip = keypoints.get_middle("Hip")

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
    keypoints,
    homo,
    face_que,
    size=FACE_DEFAULT["size"],
    ratio=FACE_DEFAULT["ratio"],
    std_th=FACE_DEFAULT["std_th"],
):
    nose = keypoints.get("Nose")
    leye = keypoints.get("LEye")
    reye = keypoints.get("REye")
    lear = keypoints.get("LEar")
    rear = keypoints.get("REar")

    # 足元にポイントを落とす
    ankle = keypoints.get_middle("Ankle")
    ear = keypoints.get_middle("Ear")
    if ankle is not None and ear is not None:
        diff = ankle[:2] - ear
    else:
        shoulder = keypoints.get_middle("Shoulder")
        hip = keypoints.get_middle("Hip")
        if shoulder is None or hip is None:
            return None, face_que

        diff = hip - shoulder
        diff = diff.astype(float) * ratio

    nose[:2] += diff
    leye[:2] += diff
    reye[:2] += diff
    lear[:2] += diff
    rear[:2] += diff

    # ホモグラフィ変換
    leye_homo = np.append(homo.transform_point(leye[:2]), leye[2])
    reye_homo = np.append(homo.transform_point(reye[:2]), reye[2])
    lear_homo = np.append(homo.transform_point(lear[:2]), lear[2])
    rear_homo = np.append(homo.transform_point(rear[:2]), rear[2])

    if lear[0] > rear[0]:
        x1 = rear[0]
        x2 = lear[0]
    else:
        x1 = lear[0]
        x2 = rear[0]

    if x1 < nose[0] and nose[0] < x2:
        # nose is between ears
        new_vector = rear_homo - lear_homo
        new_vector = rotation(new_vector[:2], -np.pi / 2)
    else:
        # face is turnning sideways
        center_eye = (leye_homo + reye_homo) / 2
        center_ear = (lear_homo + rear_homo) / 2
        new_vector = center_eye - center_ear
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
    keypoints,
    homo,
    body_que,
    size=BODY_DEFAULT["size"],
    ratio=BODY_DEFAULT["ratio"],
    std_th=BODY_DEFAULT["std_th"],
):
    lshoulder = keypoints.get("LShoulder")
    rshoulder = keypoints.get("RShoulder")
    if lshoulder[2] < kp.THRESHOLD_CONFIDENCE or rshoulder[2] < kp.THRESHOLD_CONFIDENCE:
        return None, body_que

    # 足元にポイントを落とす
    shoulder = keypoints.get_middle("Shoulder")
    hip = keypoints.get_middle("Hip")
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
    new_vector = rotation(new_vector[:2], -np.pi / 2)

    body_que.append(new_vector)
    if len(body_que) < size:
        vector = np.average(body_que, axis=0)
    else:
        body_que = body_que[-size:]
        vector = calc_ma(body_que, std_th)

    vector = normalize_vector(vector)
    return vector, body_que


def calc_arm_flexion(
    keypoints, homo, arm_que, size=ARM_DEFAULT["size"], std_th=ARM_DEFAULT["std_th"]
):
    def calc(keypoints, lr):
        shoulder = keypoints.get_middle("Shoulder")
        hip = keypoints.get_middle("Hip")

        if shoulder is None or hip is None:
            return None
        else:
            body_line = hip - shoulder
            arm = keypoints.get(lr + "Wrist", ignore_confidence=True) - keypoints.get(
                lr + "Shoulder", ignore_confidence=True
            )
            body_line = normalize_vector(body_line)
            arm = normalize_vector(arm)

            return 1.0 - np.abs(cos_similarity(body_line, arm))  # cos to sin

    larm = calc(keypoints, "L")
    rarm = calc(keypoints, "R")
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


INDICATOR_DICT = {
    IA_FORMAT[START_IDX + 0]: calc_position,
    IA_FORMAT[START_IDX + 1]: calc_face_vector,
    IA_FORMAT[START_IDX + 2]: calc_body_vector,
    IA_FORMAT[START_IDX + 3]: calc_arm_flexion,
}
