from common.json import IA_FORMAT, START_IDX, GA_FORMAT
import numpy as np
import cv2


LABEL_SETTING = [
    # size, color, thickness
    3,
    (20, 20, 20),
    2,
]

VECTOR_SETTING_LIST = {
    # arrow_length, color, tip_length
    IA_FORMAT[START_IDX + 1]: [25, (255, 0, 0), 1.0],  # face vector
    IA_FORMAT[START_IDX + 2]: [40, (0, 0, 255), 1.5],  # body vector
}


def disp_individual_activity(individual_activity_datas, field, method=None):
    if method == list(GA_FORMAT.keys())[0]:
        # attention
        field = disp_body_face(individual_activity_datas, field)
    else:
        field = disp_body_face(individual_activity_datas, field)
        # field = disp_arm_extention(individual_activity_datas, field)
    field = disp_label(individual_activity_datas, field)

    return field


def disp_label(individual_activity_datas, field):
    for data in individual_activity_datas:
        label = data[IA_FORMAT[0]]
        position = data[IA_FORMAT[START_IDX]]
        if position is not None:
            cv2.putText(
                field,
                str(label),
                tuple(position),
                cv2.FONT_HERSHEY_PLAIN,
                LABEL_SETTING[0],
                LABEL_SETTING[1],
                LABEL_SETTING[2],
            )
    return field


def disp_body_face(individual_activity_datas, field):
    def disp_arrow(key, data, field):
        position = data[IA_FORMAT[START_IDX]]
        vector = data[key]
        arrow_length = VECTOR_SETTING_LIST[key][0]

        if position is not None and vector is not None:
            # 矢印の先端の座標を計算
            end = (np.array(position) + (np.array(vector) * arrow_length)).astype(int)

            color = VECTOR_SETTING_LIST[key][1]
            tip_length = VECTOR_SETTING_LIST[key][2]
            cv2.arrowedLine(
                field,
                tuple(position),
                tuple(end),
                color,
                tipLength=tip_length,
                thickness=2,
            )

        return field

    for data in individual_activity_datas:
        # face vector
        field = disp_arrow(IA_FORMAT[START_IDX + 1], data, field)
        # body vector
        field = disp_arrow(IA_FORMAT[START_IDX + 2], data, field)

    return field


def disp_arm_extention(individual_activity_datas, field):
    for data in individual_activity_datas:
        position = data[IA_FORMAT[START_IDX]]
        arm_estention = data[IA_FORMAT[START_IDX + 3]]
        if arm_estention is not None:
            arm_estention = np.round(arm_estention, decimals=3)
            cv2.putText(
                field,
                str(arm_estention),
                tuple(position),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                2,
            )

    return field
