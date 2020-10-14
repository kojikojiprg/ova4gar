from common import database
import cv2
import numpy as np


def vector(frame_num, indicator, field, homo):
    datas = indicator.indicator_lst[frame_num]
    for data in datas:
        if data[2] is None or data[3] is None:
            continue

        start = data[2]
        end = data[2] + data[3]
        start = homo.transform_point(start)
        end = homo.transform_point(end)
        color = data[4]

        cv2.arrowedLine(field, tuple(start), tuple(end), color, tipLength=1.5)

    return field


def move_hand(frame_num, indicator, field, homo):
    datas = indicator.indicator_lst[frame_num]
    for data in datas:
        if data[2] is None:
            continue

        point = data[2]
        point = homo.transform_point(point)
        color = data[4]
        cv2.circle(field, tuple(point), 7, color, thickness=-1)
    return field


def face_direction(frame_num, indicator, field, homo, arrow_length=10):
    datas = indicator.indicator_lst[frame_num]
    for data in datas:
        if data[2] is None:
            continue

        start = data[2]
        x = np.cos(data[3])
        y = np.sin(data[3])
        end = start + np.array([x, y]) * arrow_length
        start = homo.transform_point(start)
        end = homo.transform_point(end)

        # ホモグラフィ変換後の矢印の長さを揃える
        ratio = arrow_length / np.linalg.norm(end - start)
        end = start + ((end - start) * ratio).astype(int)

        color = data[4]
        cv2.arrowedLine(field, tuple(start), tuple(end), color, tipLength=1.5)
    return field


def moving_distance(frame_num, indicator, field, homo):
    try:
        datas = indicator.indicator_lst[frame_num]
    except IndexError:
        return field

    for data in datas:
        diff = data[4]
        if diff != np.nan:
            start = data[2]
            end = data[3]
            start = homo.transform_point(start)
            end = homo.transform_point(end)
            color = data[5]
            cv2.line(field, tuple(start), tuple(end), color, thickness=3)
    return field


def density(frame_num, indicator, field, homo, min_r=8):
    datas = indicator.indicator_lst[frame_num]
    for data in datas:
        point = np.average(data[2], axis=0)
        point = homo.transform_point(point)
        r = min_r + data[3]
        color = data[4]
        cv2.circle(field, tuple(point), r, color, thickness=-1)
    return field


FUNC_DICT = {
    # [display_method, is_reset_display]
    database.VECTOR_TABLE.name: [vector, False],
    database.MOVE_HAND_TABLE.name: [move_hand, False],
    database.FACE_DIRECTION_TABLE.name: [face_direction, True],
    database.MOVING_DISTANCE_TABLE.name: [moving_distance, False],
    database.DENSITY_TABLE.name: [density, True],
}
