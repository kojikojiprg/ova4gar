
from common.json import PERSON_FORMAT
import numpy as np
import cv2


VECTOR_SETTING_LIST = {
    # arrow_length, color, tip_length
    PERSON_FORMAT[4]: [20, (255, 0, 0), 1.0],   # face vector
    PERSON_FORMAT[5]: [30, (0, 0, 255), 1.5],   # body vector
}


def disp_person(person_datas, field):
    field = disp_body_face(person_datas, field)
    field = disp_arm_extention(person_datas, field)
    field = disp_arm_extention2(person_datas, field)

    return field


def disp_body_face(person_datas, field):
    def disp_arrow(key, data, field):
        position = data[PERSON_FORMAT[3]]
        vector = data[key]
        arrow_length = VECTOR_SETTING_LIST[key][0]

        if position is not None and vector is not None:
            # 矢印の先端の座標を計算
            end = (np.array(position) + (np.array(vector) * arrow_length)).astype(int)

            color = VECTOR_SETTING_LIST[key][1]
            tip_length = VECTOR_SETTING_LIST[key][2]
            cv2.arrowedLine(field, tuple(position), tuple(end), color, tipLength=tip_length)

        return field

    for data in person_datas:
        # face vector
        field = disp_arrow(PERSON_FORMAT[4], data, field)
        # body vector
        field = disp_arrow(PERSON_FORMAT[5], data, field)

    return field


def disp_arm_extention(person_datas, field):
    for data in person_datas:
        position = data[PERSON_FORMAT[3]]
        arm_estention = data[PERSON_FORMAT[6]]
        if arm_estention is not None:
            arm_estention = np.round(arm_estention, decimals=3)
            cv2.putText(
                field, str(arm_estention), tuple(position), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    return field


def disp_arm_extention2(person_datas, field):
    for data in person_datas:
        position = data[PERSON_FORMAT[3]]
        arm_estention = data[PERSON_FORMAT[7]]
        if arm_estention is not None:
            arm_estention = np.round(arm_estention, decimals=3)
            position[1] += 20
            cv2.putText(
                field, str(arm_estention), tuple(position), cv2.FONT_HERSHEY_PLAIN, 2, (0, 230, 0), 2)

    return field
