from common import common
import numpy as np


def calc_save(persons, db):
    datas = []
    for person in persons:
        frame_num = person.start_frame_num

        for keypoints in person.keypoints_lst:
            angle = calc(keypoints)
            datas.append((person.id, frame_num, angle))
            frame_num += 1

    # データベースに書き込み
    db.drop_table(common.MOVE_HAND_TABLE_NAME)
    db.create_table(common.MOVE_HAND_TABLE_NAME, common.MOVE_HAND_TABLE_COLS)
    db.insert_datas(
        common.MOVE_HAND_TABLE_NAME,
        list(common.MOVE_HAND_TABLE_COLS.keys()),
        datas)


def calc(keypoints):
    if keypoints is None:
        return None

    mid_shoulder = keypoints.get_middle('Shoulder')
    mid_hip = keypoints.get_middle('Hip')

    # 体軸ベクトルとノルム
    axis = mid_shoulder - mid_hip
    norm_axis = np.linalg.norm(axis, ord=2)

    angle = 0.
    for side in ('R', 'L'):
        elbow = keypoints.get(side + 'Elbow', ignore_confidence=True)
        wrist = keypoints.get(side + 'Wrist', ignore_confidence=True)

        # 前肢ベクトルとノルム
        vec = wrist - elbow
        norm = np.linalg.norm(vec, ord=2)

        # 体軸と前肢の角度(左右の大きい方を選択する)
        angle = max(
            angle,
            np.arccos(np.dot(axis, vec) / (norm_axis * norm + 1e-10)))

    return angle
