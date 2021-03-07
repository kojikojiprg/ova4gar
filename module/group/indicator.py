from common import database
from halfline import HalfLine, calc_cross
import numpy as np
from pyclustering.cluster import gmeans


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
    datas = []
    if len(points) > 2:
        gm = gmeans.gmeans(points, k_init=k_init)
        gm.process()
        for cluster in gm.get_clusters():
            datas.append((frame_num, points[cluster], len(cluster)))
    else:
        for point in points:
            datas.append((frame_num, np.array([point]), 1))

    return datas


def calc_attension(frame_num, person_datas, homo, k_init=1):
    # 直線を求める
    lines = []
    for data in person_datas:
        keypoints = data[database.PERSON_TABLE.index('Keypoints')]
        face_vector = data[database.PERSON_TABLE.index('Face_Vector')]
        if keypoints is not None and face_vector is not None:
            point = keypoints.get_middle('Ankle')
            point = homo.transform_point(point)
            line = HalfLine(point, face_vector)
            lines.append(line)

    # 半直線の交点を求める
    cross_points = []
    for i in range(len(lines) - 1):
        for j in range(i + 1, len(lines)):
            cross_point = calc_cross(lines[i], lines[j])
            if cross_point is not None:
                if lines[i].is_online(cross_point) and lines[j].is_online(cross_point):
                    # 交点が半直線上にあれば追加する
                    cross_points.append(cross_point)
    cross_points = np.array(cross_points)

    datas = []

    # g-means でクラスタリング
    if len(cross_points) > 0:
        gm = gmeans.gmeans(cross_points, k_init=k_init)
        gm.process()
        for cluster in gm.get_clusters():
            datas.append((frame_num, cross_points[cluster], len(cluster)))
    else:
        datas.append((frame_num, None, 0))

    return datas


INDICATOR_DICT = {
    database.GROUP_TABLE_LIST[0].name: calc_density,
    database.GROUP_TABLE_LIST[1].name: calc_attension,
}
