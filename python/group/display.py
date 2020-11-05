from common import database
import numpy as np
import cv2


def display_density(datas, field, heatmap, min_r=8):
    for data in datas:
        point = np.average(data[1], axis=0).astype(int)
        r = min_r + len(data[1])
        color = heatmap.colormap(len(data[1]))
        cv2.circle(field, tuple(point), r, color, thickness=-1)
    return field


def display_attention(datas, field, heatmap, min_r=8):
    for data in datas:
        if data[1] is not None:
            for point in data[1]:
                cv2.circle(field, tuple(point.astype(int)), 5, (0, 0, 0), thickness=-1)
            point = np.average(data[1], axis=0).astype(int)
            r = min_r + len(data[1])
            color = heatmap.colormap(len(data[1]))
            cv2.circle(field, tuple(point), r, color, thickness=-1)
    return field


DISPLAY_DICT = {
    database.GROUP_TABLE_LIST[0].name: display_density,
    database.GROUP_TABLE_LIST[1].name: display_attention,
}

HEATMAP_SETTING_DICT = {
    # key: [is_heatmap, heatmap_data]
    database.GROUP_TABLE_LIST[0].name: [True, -1],
    database.GROUP_TABLE_LIST[1].name: [True, -1],
}
