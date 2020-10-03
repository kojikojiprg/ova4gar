from common import database
import numpy as np


def calc_save(frames, db, th=50):
    datas = []
    for frame in frames:
        points = []
        for keypoints in frame.keypoints_lst:
            if keypoints is not None:
                point = keypoints.get_middle('Ankle')
                points.append(Point(point))

        # 密度計算
        distribution = []
        max_len = 0
        for i in range(len(points)):
            cluster = []
            if not points[i].counted:
                # カウントされていないポイントを取得
                points[i].counted = True
                cluster.append(points[i].coor.tolist())

                # 異なるポイントをターゲットにする
                for j in range(len(points)):
                    if i != j and not points[j].counted:
                        # カウントされていないターゲットを取得
                        diff = points[i].coor - points[j].coor
                        norm = np.linalg.norm(diff)
                        if norm < th:
                            # 距離が近ければクラスターに加える
                            cluster.append(points[j].coor.tolist())
                            points[j].counted = True
            if len(cluster) > 0:
                max_len = max(max_len, len(cluster))
                distribution.append(cluster)

        # 要素数を揃えてデータベースのデータに変換する
        for i in range(len(distribution)):
            for _ in range(max_len - len(distribution[i])):
                distribution[i].append([np.nan, np.nan])
            datas.append((frame.num, i, np.array(distribution[i])))

    # データベースに書き込み
    table = database.DENSITY_TABLE
    db.drop_table(table.name)
    db.create_table(table.name, table.cols)
    db.insert_datas(
        table.name,
        list(table.cols.keys()),
        datas
    )


class Point:
    def __init__(self, coor):
        self.coor = coor
        self.counted = False
