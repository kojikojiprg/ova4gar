import numpy as np
from keypoint.keypoint import Keypoints
from utility.functions import cos_similarity, normalize_vector, rotation
from utility.transform import Homography

from individual.individual_que import Que


def position(kps: Keypoints, homo: Homography, que: Que, **defs):
    # read default
    th_ankle = defs["th_ankle"]
    ratio = defs["ratio"]

    new_pos = kps.get_middle("Ankle", th_conf=th_ankle)
    if new_pos is None:
        shoulder = kps.get_middle("Shoulder")
        hip = kps.get_middle("Hip")

        if shoulder is None or hip is None:
            return None
        else:
            body_line = hip - shoulder
            new_pos = hip + body_line * ratio

    new_pos = homo.transform_point(new_pos)
    new_pos = que.put_pop(new_pos)

    return new_pos.astype(int)


def face(kps: Keypoints, homo: Homography, que: Que, **defs):
    ratio = defs["ratio"]

    nose = kps.get("Nose")
    leye = kps.get("LEye")
    reye = kps.get("REye")
    lear = kps.get("LEar")
    rear = kps.get("REar")

    # 足元にポイントを落とす
    ankle = kps.get_middle("Ankle")
    ear = kps.get_middle("Ear")
    if ankle is not None and ear is not None:
        diff = ankle[:2] - ear
    else:
        shoulder = kps.get_middle("Shoulder")
        hip = kps.get_middle("Hip")
        if shoulder is None or hip is None:
            return None

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

    new_vector = que.put_pop(new_vector)
    new_vector = normalize_vector(new_vector)
    return new_vector


def body(kps: Keypoints, homo: Homography, que: Que, **defs):
    ratio = defs["ratio"]

    lshoulder = kps.get("LShoulder")
    rshoulder = kps.get("RShoulder")
    if lshoulder is None or rshoulder is None:
        return None

    # 足元にポイントを落とす
    shoulder = kps.get_middle("Shoulder")
    hip = kps.get_middle("Hip")
    if shoulder is None or hip is None:
        return None

    diff = hip - shoulder
    diff = diff.astype(float) * ratio
    lshoulder[:2] += diff
    rshoulder[:2] += diff

    # ホモグラフィ変換
    lshoulder = homo.transform_point(lshoulder[:2])
    rshoulder = homo.transform_point(rshoulder[:2])

    new_vector = rshoulder - lshoulder
    new_vector = rotation(new_vector[:2], -np.pi / 2)

    new_vector = que.put_pop(new_vector)

    new_vector = normalize_vector(new_vector)
    return new_vector


def arm(kps: Keypoints, homo: Homography, que: Que, **defs):
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

    larm = calc(kps, "L")
    rarm = calc(kps, "R")
    if larm is None and rarm is None:
        return None
    elif larm is None and rarm is not None:
        new_arm = rarm
    elif larm is not None and rarm is None:
        new_arm = larm
    else:
        new_arm = np.max((larm, rarm))

    new_arm = que.put_pop(new_arm)

    return new_arm
