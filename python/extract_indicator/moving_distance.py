from common import database
import numpy as np


def calc_save(persons, db, distance_th=5):
    datas = []
    for person in persons:
        frame_num = person.start_frame_num + 1
        keypoints = person.keypoints_lst
        for pre, now in zip(keypoints[:-1], keypoints[1:]):
            if pre is None or now is None:
                continue

            pre_hip = pre.get_middle('Hip')
            now_hip = now.get_middle('Hip')
            diff = np.linalg.norm(pre_hip - now_hip)

            if diff > distance_th:
                datas.append((
                    person.id,
                    frame_num,
                    pre.get_middle('Ankle'),
                    now.get_middle('Ankle'),
                    diff))

            frame_num += 1

    # データベースに書き込み
    table = database.MOVING_DISTANCE_TABLE
    db.drop_table(table.name)
    db.create_table(table.name, table.cols)
    db.insert_datas(
        table.name,
        list(table.cols.keys()),
        datas)
