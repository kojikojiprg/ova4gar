
from common.json import PERSON_FORMAT
import cv2


VECTOR_SETTING_LIST = {
    # arrow_length, color, tip_length
    PERSON_FORMAT[4]: [20, (255, 0, 0), 1.0],   # face vector
    PERSON_FORMAT[5]: [30, (0, 0, 255), 1.5],   # body vector
}


def disp_person(person_datas, field):
    field = disp_body_face(person_datas, field)

    return field


def disp_body_face(person_datas, field):
    def disp_arrow(key, data, field):
        position = data[PERSON_FORMAT[3]]
        vector = data[key]
        arrow_length = VECTOR_SETTING_LIST[key][0]

        # 矢印の先端の座標を計算
        end = (position + (vector * arrow_length)).astype(int)

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
