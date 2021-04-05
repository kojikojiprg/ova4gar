from common.json import PERSON_FORMAT, GROUP_FORMAT
from group.halfline import HalfLine, calc_cross
import numpy as np
from pyclustering.cluster import gmeans


def calc_density(frame_num, person_datas, homo, k_init=3):
    json_format = GROUP_FORMAT['density']

    points = []
    for data in person_datas:
        position = data[PERSON_FORMAT[3]]
        if position is not None:
            points.append(position)
    points = np.array(points)

    # g-means でクラスタリング
    datas = []
    if len(points) > 2:
        gm = gmeans.gmeans(points, k_init=k_init)
        gm.process()
        for cluster in gm.get_clusters():
            datas.append({
                json_format[0]: frame_num,
                json_format[1]: points[cluster].tolist(),
                json_format[2]: len(cluster)})
    else:
        for point in points:
            datas.append({
                json_format[0]: frame_num,
                json_format[1]: [point.tolist()],
                json_format[2]: 1})

    return datas


def calc_attension(frame_num, person_datas, homo, k_init=1):
    json_format = GROUP_FORMAT['attention']

    # 直線を求める
    lines = []
    for data in person_datas:
        position = data[PERSON_FORMAT[3]]
        face_vector = data[PERSON_FORMAT[4]]
        if position is not None and face_vector is not None:
            line = HalfLine(position, face_vector)
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
            datas.append({
                json_format[0]: frame_num,
                json_format[1]: cross_points[cluster].tolist(),
                json_format[2]: len(cluster)})

    return datas


def calc_passing(frame_num, person_datas, homo, k_init=1):
    pass


keys = list(GROUP_FORMAT.keys())
INDICATOR_DICT = {
    keys[0]: calc_attension,
    keys[1]: calc_passing,
}
