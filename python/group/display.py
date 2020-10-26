from common import database
import numpy as np
import cv2


def display_density(datas, field, min_r=8):
    for data in datas:
        point = np.average(data[1], axis=0).astype(int)
        r = min_r + len(data[1])
        color = (255, 0, 0)
        cv2.circle(field, tuple(point), r, color, thickness=-1)
    return field


DISPLAY_DICT = {
    database.GROUP_TABLE_LIST[0].name: display_density,
}
