import numpy as np
from pyclustering.cluster import gmeans


def calc_density(person_datas, homo, k_init=3):
    points = []
    for data in person_datas:
        keypoints = data[2]
        if keypoints is not None:
            point = keypoints.get_middle('Ankle')
            point = homo.transform_point(point)
            points.append(point)
    points = np.array(points)

    # g-means でクラスタリング
    gm = gmeans.gmeans(points, k_init=k_init)
    gm.process()
    clusters = []
    for cluster in gm.get_clusters():
        clusters.append(points[cluster])

    return np.array(clusters)


INDICATOR_DICT = {
    'density': calc_density,
}
