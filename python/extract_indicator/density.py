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
    db.drop_table(common.DENSITY_TABLE_NAME)
    db.create_table(common.DENSITY_TABLE_NAME, common.DENSITY_TABLE_COLS)
    db.insert_datas(
        common.DENSITY_TABLE_NAME,
        list(common.DENSITY_TABLE_COLS.keys()),
        datas
    )
