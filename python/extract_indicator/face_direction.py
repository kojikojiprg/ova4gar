from common import database
from common import keypoint as kp
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
                point = np.nan
                angle = np.nan

            datas.append((
                person.id,
                frame_num,
                point,
                angle))

            frame_num += 1

    # データベースに書き込み
    table = database.FACE_DIRECTION
    db.drop_table(table.name)
    db.create_table(table.name, table.cols)
    db.insert_datas(
        table.name,
        list(table.cols.keys()),
        datas)


def calc(keypoints):
    nose = keypoints.get('Nose')
    lear = keypoints.get('LEar')
    rear = keypoints.get('REar')

    if lear[2] < kp.confidence_th:
        diff = nose - rear
        angle = np.arctan(diff[2])
    elif rear[2] < kp.confidence_th:
        diff = lear - nose
        angle = np.arctan(diff[2]) + np.pi
    elif lear[2] < kp.confidence_th and rear[2] < kp.confidence_th:
        angle = np.nan
    else:
        diff = lear - rear
        angle = np.arctan(diff[2])
        if diff[0] >= 0:
            angle += np.pi / 2
        else:
            angle -= np.pi / 2

    return angle
