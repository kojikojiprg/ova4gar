from common import database
import numpy as np
from pyclustering.cluster import gmeans


def calc_save(frames, db, k_init=3):
    datas = []
    for frame in frames:
        points = []
        for keypoints in frame.keypoints_lst:
            if keypoints is not None:
                point = keypoints.get_middle('Ankle')
                points.append(point)
        points = np.array(points)

        # g-means でクラスタリング
        gm = gmeans.gmeans(points, k_init=k_init)
        gm.process()
        for i, cluster in enumerate(gm.get_clusters()):
            datas.append((frame.num, i, points[cluster], len(cluster)))

    # データベースに書き込み
    table = database.DENSITY_TABLE
    db.drop_table(table.name)
    db.create_table(table.name, table.cols)
    db.insert_datas(
        table.name,
        list(table.cols.keys()),
        datas
    )
