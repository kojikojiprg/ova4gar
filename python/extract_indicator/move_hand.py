from common import common
import numpy as np


def calc_save(persons, db):
    datas = []
    for person in persons:
        frame_num = person.start_frame_num

        for keypoints in person.keypoints_lst:
            if keypoints is not None:
                point = keypoints.get_middle('Ankle')
                angle = calc(keypoints)
            else:
                point = None
                angle = None

            datas.append((
                person.id,
                frame_num,
                point,
                angle))

            frame_num += 1

    # データベースに書き込み
    table = common.MOVE_HAND_TABLE
    db.drop_table(table.name)
    db.create_table(table.name, table.cols)
    db.insert_datas(
        table.name,
        list(table.cols.keys()),
        datas)


def calc(keypoints):
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
