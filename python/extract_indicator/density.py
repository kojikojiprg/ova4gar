from common import common
import numpy as np


def calc_save(frames, db):
    datas = []
    for frame in frames:
        points = []
        for keypoints in frame.keypoints_lst:
            if keypoints is not None:
                point = keypoints.get_middle('Ankle')
                points.append(point.tolist())

        datas.append((frame.num, np.array(points)))

    # データベースに書き込み
    table = common.DENSITY_TABLE
    db.drop_table(table.name)
    db.create_table(table.name, table.cols)
    db.insert_datas(
        table.name,
        list(table.cols.keys()),
        datas
    )
