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


def calc_attension(frame_num, person_datas, homo):
    points = []
    for data in person_datas:
        keypoints = data[database.PERSON_TABLE.index('Keypoints')]
        if keypoints is not None:
            point = keypoints.get_middle('Ankle')
            point = homo.transform_point(point)
            points.append(point)
    points = np.array(points)


INDICATOR_DICT = {
    database.GROUP_TABLE_LIST[0].name: calc_density,
}
