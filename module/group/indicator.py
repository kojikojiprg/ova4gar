from common import database
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
    gm = gmeans.gmeans(points, k_init=k_init)
    gm.process()
    datas = []
    for cluster in gm.get_clusters():
        datas.append((frame_num, points[cluster], len(cluster)))

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


class HalfLine:
    def __init__(self, point, vector):
        self.x0 = point[0]
        self.y0 = point[1]
        self.a = vector[1] / vector[0]              # 傾き
        self.b = -1 * self.a * self.x0 + self.y0    # 切片

        # 象限
        if vector[0] > 0 and vector[1] >= 0:
            self.quadrant = 1
        elif vector[0] <= 0 and vector[1] > 0:
            self.quadrant = 2
        elif vector[0] < 0 and vector[1] <= 0:
            self.quadrant = 3
        elif vector[0] >= 0 and vector[1] < 0:
            self.quadrant = 4
        else:
            self.quadrant = 0

    def func(self, x):
        y = self.a * x + self.b
        return y

    def limit(self, x, y):
        # 半直線の先端を判定
        diffx = x - self.x0
        diffy = y - self.y0
        if self.quadrant == 1:
            return diffx > 0 and diffy >= 0
        elif self.quadrant == 2:
            return diffx <= 0 and diffy > 0
        elif self.quadrant == 3:
            return diffx < 0 and diffy <= 0
        elif self.quadrant == 4:
            return diffx >= 0 and diffy < 0
        else:
            return False

    def is_online(self, point):
        x = point[0]
        y = point[1]

        if int(self.func(x) - y) == 0:
            return self.limit(x, y)
        else:
            return False


def calc_cross(l1, l2):
    a_c = l1.a - l2.a
    d_b = l2.b - l1.b
    ad_bc = l1.a * l2.b - l1.b * l2.a

    if abs(a_c) < 1e-5 or abs(d_b) > 1e+5 or abs(ad_bc) > 1e+5:
        return None
    else:
        x = d_b / a_c
        y = ad_bc / a_c
        return x, y


INDICATOR_DICT = {
    database.GROUP_TABLE_LIST[0].name: calc_density,
    database.GROUP_TABLE_LIST[1].name: calc_attension,
}
