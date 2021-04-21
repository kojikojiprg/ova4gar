from common.json import PERSON_FORMAT, GROUP_FORMAT
from common.functions import cos_similarity, euclidean, gauss
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


def calc_attention(frame_num, person_datas, homo, field, angle_range=np.pi / 18, division=5):
    key = inspect.currentframe().f_code.co_name.replace('calc_', '')
    json_format = GROUP_FORMAT[key]

    pixcel_datas = np.zeros((field.shape[1], field.shape[0]))
    for x in range(0, field.shape[1], division):
        for y in range(0, field.shape[0], division):
            point = np.array([x, y])
            for data in person_datas:
                pos = data[PERSON_FORMAT[3]]
                face_vector = data[PERSON_FORMAT[4]]
                if pos is None or face_vector is None:
                    continue

                diff = point - np.array(pos)
                shita = np.arctan2(diff[1], diff[0])

                face_shita = np.arctan2(face_vector[1], face_vector[0])

                if (
                    face_shita - angle_range <= shita and
                    shita <= face_shita + angle_range
                ):
                    pixcel_datas[x, y] += 1

    datas = []
    for x, row in enumerate(pixcel_datas):
        for y, data in enumerate(row):
            if data > 0:
                datas.append({
                    json_format[0]: frame_num,
                    json_format[1]: [x, y],
                    json_format[2]: data
                })

    return datas


def calc_passing(frame_num, person_datas, homo, th=0.5, th_shita=np.pi / 3):
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
            p1_arm = p1[PERSON_FORMAT[6]]
            p2_arm = p2[PERSON_FORMAT[6]]

            if (
                p1_pos is None or p2_pos is None or
                p1_body is None or p2_body is None or
                p1_arm is None or p2_arm is None
            ):
                continue

            p1_pos = np.array(p1_pos)
            p2_pos = np.array(p2_pos)
            p1_body = np.array(p1_body)
            p2_body = np.array(p2_body)

            # calc vector of each other
            p1p2 = p2_pos - p1_pos
            p2p1 = p1_pos - p2_pos

            # calc angle between p1 body and p1p2 vector
            shita1 = np.arccos(cos_similarity(p1_body, p1p2))
            # calc angle between p2 body and p2p1 vector
            shita2 = np.arccos(cos_similarity(p2_body, p2p1))

            if (
                (0 <= shita1 and shita1 <= th_shita) and
                (0 <= shita2 and shita2 <= th_shita)
            ):
                norm = euclidean(p1_pos, p2_pos)
                distance_prob = gauss(norm, mu=200, sigma=100)

                # 向き合っている度合い
                opposite = np.abs(cos_similarity(p1_body, p2_body))

                # 腕を伸ばしている度合い
                arm = np.average([p1_arm, p2_arm])

                # 受け渡しをしている尤度
                likelifood = np.average([opposite, arm]) * distance_prob

                # 中心点
                center = np.average([p1_pos, p2_pos], axis=0)

                if likelifood >= th:
                    datas.append({
                        json_format[0]: frame_num,
                        json_format[1]: center.astype(int).tolist(),
                        json_format[2]: likelifood})

    return datas


keys = list(GROUP_FORMAT.keys())
INDICATOR_DICT = {
    keys[0]: calc_attention,
    keys[1]: calc_passing,
}
