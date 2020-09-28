import cv2


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


def density(frame_num, indicator, field, homo):
    datas = indicator.indicator_lst[frame_num]
    for point in datas[0][1]:
        point = homo.transform_point(point)
        cv2.circle(field, tuple(point), 7, (255, 0, 0), thickness=-1)
    return field
