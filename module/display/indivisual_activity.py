from common.json import IA_FORMAT, GA_FORMAT
import numpy as np
import cv2


VECTOR_SETTING_LIST = {
    # arrow_length, color, tip_length
    IA_FORMAT[4]: [20, (255, 0, 0), 1.0],   # face vector
    IA_FORMAT[5]: [30, (0, 0, 255), 1.5],   # body vector
}


def disp_indivisual_activity(indivisual_activity_datas, field, method=None):
    if method == list(GA_FORMAT.keys())[0]:
        # attention
        field = disp_body_face(indivisual_activity_datas, field)
    else:
        field = disp_body_face(indivisual_activity_datas, field)
        field = disp_arm_extention(indivisual_activity_datas, field)

    return field


def disp_body_face(indivisual_activity_datas, field):
    def disp_arrow(key, data, field):
        position = data[IA_FORMAT[3]]
        vector = data[key]
        arrow_length = VECTOR_SETTING_LIST[key][0]

        if position is not None and vector is not None:
            # 矢印の先端の座標を計算
            end = (np.array(position) +
                   (np.array(vector) * arrow_length)).astype(int)

            color = VECTOR_SETTING_LIST[key][1]
            tip_length = VECTOR_SETTING_LIST[key][2]
            cv2.arrowedLine(
                field,
                tuple(position),
                tuple(end),
                color,
                tipLength=tip_length)

        return field

    for data in indivisual_activity_datas:
        # face vector
        field = disp_arrow(IA_FORMAT[4], data, field)
        # body vector
        field = disp_arrow(IA_FORMAT[5], data, field)

    return field


def disp_arm_extention(indivisual_activity_datas, field):
    for data in indivisual_activity_datas:
        position = data[IA_FORMAT[3]]
        arm_estention = data[IA_FORMAT[6]]
        if arm_estention is not None:
            arm_estention = np.round(arm_estention, decimals=3)
            cv2.putText(field, str(arm_estention), tuple(position),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    return field
