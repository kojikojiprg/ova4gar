from common.json import PERSON_FORMAT, GROUP_FORMAT
from common.functions import cos_similarity, euclidean
from group.halfline import HalfLine, calc_cross
import inspect
import numpy as np
from pyclustering.cluster import gmeans


def calc_density(frame_num, person_datas, homo, k_init=3):
    key = inspect.currentframe().f_code.co_name.replace('calc_', '')
    json_format = GROUP_FORMAT[key]

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


def calc_attention(frame_num, person_datas, homo, k_init=1):
    key = inspect.currentframe().f_code.co_name.replace('calc_', '')
    json_format = GROUP_FORMAT[key]

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
                json_format[1]: cross_points[cluster].astype(int).tolist(),
                json_format[2]: len(cluster)})

    return datas


def calc_passing(frame_num, person_datas, homo, th_norm=300, th_shita=90):
    key = inspect.currentframe().f_code.co_name.replace('calc_', '')
    json_format = GROUP_FORMAT[key]

    datas = []
    for i in range(len(person_datas) - 1):
        for j in range(i + 1, len(person_datas)):
            p1 = person_datas[i]
            p2 = person_datas[j]

            # obtain datas
            p1_pos = p1[PERSON_FORMAT[3]]
            p2_pos = p2[PERSON_FORMAT[3]]
            p1_body = p1[PERSON_FORMAT[5]]
            p2_body = p2[PERSON_FORMAT[5]]

            if p1_pos is None or p2_pos is None or p1_body is None or p2_body is None:
                datas.append({
                    json_format[0]: frame_num,
                    json_format[1]: None,
                    json_format[2]: None})
                return datas

            p1_pos = np.array(p1_pos)
            p2_pos = np.array(p2_pos)
            p1_body = np.array(p1_body)
            p2_body = np.array(p2_body)

            # calc vector of each other
            p1p2 = p2_pos - p1_pos
            p2p1 = p1_pos - p2_pos

            # calc angle between p1 body and p1p2 vector
            shita1 = np.rad2deg(np.arccos(cos_similarity(p1_body, p1p2)))
            # calc angle between p2 body and p2p1 vector
            shita2 = np.rad2deg(np.arccos(cos_similarity(p2_body, p2p1)))

            norm = euclidean(p1_pos, p2_pos)
            if norm < th_norm and (
                (0 <= shita1 and shita1 <= th_shita) and
                (0 <= shita2 and shita2 <= th_shita)
            ):
                # 向き合っている度合い
                opposite = np.abs(cos_similarity(p1_body, p2_body))

                # 腕を伸ばしている度合い
                arm = np.average([p1[PERSON_FORMAT[6]], p2[PERSON_FORMAT[6]]])

                # 受け渡しをしている尤度
                likelifood = opposite * arm

                # 中心点
                center = np.average([p1_pos, p2_pos], axis=1)

                datas.append({
                    json_format[0]: frame_num,
                    json_format[1]: center.astype(int).tolist(),
                    json_format[2]: likelifood})
            else:
                datas.append({
                    json_format[0]: frame_num,
                    json_format[1]: None,
                    json_format[2]: None})

    return datas


keys = list(GROUP_FORMAT.keys())
INDICATOR_DICT = {
    keys[0]: calc_attention,
    keys[1]: calc_passing,
}
