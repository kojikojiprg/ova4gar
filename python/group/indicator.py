from common import database
import numpy as np
from pyclustering.cluster import gmeans


def calc_line(vector, point):
    a = vector[0]
    b = vector[1]
    c = -1 * np.dot(vector, point)
    return a, b, c


def calc_cross(l1, l2):
    a1 = l1[0]
    b1 = l1[1]
    c1 = l1[2]
    a2 = l2[0]
    b2 = l2[1]
    c2 = l2[2]

    ab = a1 * b2 - a2 * b1
    bc = b1 * c2 - b2 * c1
    ca = c1 * a2 - c2 * a1
    if abs(ab) < 1e-2:
        # 平行な直線
        return None
    else:
        x = bc / ab
        y = ca / ab

        return x, y


def calc_density(frame_num, person_datas, homo, k_init=3):
    points = []
    for data in person_datas:
        keypoints = data[database.PERSON_TABLE.index('Keypoints')]
        if keypoints is not None:
            point = keypoints.get_middle('Ankle')
            point = homo.transform_point(point)
            points.append(point)
    points = np.array(points)

    # g-means でクラスタリング
    gm = gmeans.gmeans(points, k_init=k_init)
    gm.process()
    datas = []
    for cluster in gm.get_clusters():
        datas.append((frame_num, points[cluster], len(cluster)))

    return datas


def calc_attension(frame_num, person_datas, homo, k_init=1):
    # 直線を求める
    linears = []
    for data in person_datas:
        keypoints = data[database.PERSON_TABLE.index('Keypoints')]
        face_vector = data[database.PERSON_TABLE.index('Face_Vector')]
        if keypoints is not None and face_vector is not None:
            point = keypoints.get_middle('Ankle')
            point = homo.transform_point(point)
            linear = calc_line(face_vector, point)
            linears.append(linear)

    # 直線の交点を求める
    cross_points = []
    for i in range(len(linears) - 1):
        for j in range(i + 1, len(linears)):
            cross_point = calc_cross(linears[i], linears[j])
            if cross_point is not None:
                cross_points.append(cross_point)
    cross_points = np.array(cross_points)

    # g-means でクラスタリング
    gm = gmeans.gmeans(cross_points, k_init=k_init)
    gm.process()
    datas = []
    for cluster in gm.get_clusters():
        if len(cluster) >= 3:
            # 3人以上の視線が集まっている箇所のみ抽出
            datas.append((frame_num, cross_points[cluster], len(cluster)))

    return datas


INDICATOR_DICT = {
    database.GROUP_TABLE_LIST[0].name: calc_density,
    database.GROUP_TABLE_LIST[1].name: calc_attension,
}
