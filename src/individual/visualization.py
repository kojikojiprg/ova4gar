from typing import Any, Dict, List

import cv2
import numpy as np
from numpy.typing import NDArray

# size, color, thickness
ID_SETTING = (3, (20, 20, 20), 2)


VECTOR_SETTING_LIST = {
    # arrow_length, color, tip_length
    "face": [25, (255, 0, 0), 1.0],
    "body": [40, (0, 0, 255), 1.5],
}


def visualize(inds_data: List[Dict[str, Any]], field: NDArray):
    field = _vis_body_face(inds_data, field)
    field = _vis_arm(inds_data, field)
    field = _vis_id(inds_data, field)

    return field


def _vis_id(inds_data: List[Dict[str, Any]], field: NDArray):
    for data in inds_data:
        ind_id = data["id"]
        position = data["position"]
        if position is not None:
            cv2.putText(
                field,
                str(ind_id),
                tuple(position),
                cv2.FONT_HERSHEY_PLAIN,
                ID_SETTING[0],
                ID_SETTING[1],
                ID_SETTING[2],
            )
    return field


def __vis_arrow(key: str, data: Dict[str, Any], field: NDArray):
    position = data["position"]
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


def _vis_body_face(inds_data: List[Dict[str, Any]], field: NDArray):
    for data in inds_data:
        # face vector
        field = __vis_arrow("face", data, field)
        # body vector
        field = __vis_arrow("body", data, field)

    return field


def _vis_arm(inds_data: List[Dict[str, Any]], field: NDArray):
    for data in inds_data:
        position = data["position"]
        arm = data["arm"]
        if arm is not None:
            arm = np.round(arm, decimals=3)
            cv2.putText(
                field,
                str(arm),
                tuple(position),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                2,
            )

    return field
