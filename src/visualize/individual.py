import gc
import os
import sys
from typing import Any, Dict, List

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

sys.path.append("src")
from utility.json_handler import load
from utility.video import Capture, Writer, concat_field_with_frame

from visualize.keypoint import write_frame as kps_write_frame

# size, color, thickness
ID_SETTING = (3, (20, 20, 20), 2)


VECTOR_SETTING_LIST = {
    # arrow_length, color, tip_length
    "face": [25, (255, 0, 0), 1.0],
    "body": [40, (0, 0, 255), 1.5],
}


def write_video(data_dir, field, start_frame_num, end_frame_num):
    # create video capture
    kps_video_path = os.path.join(data_dir, "video", "keypoints.mp4")
    cap = Capture(kps_video_path)

    # create video writer
    cmb_img = concat_field_with_frame(cap.read()[1], field)
    size = cmb_img.shape[1::-1]
    out_path = os.path.join(data_dir, "video", "individual.mp4")
    wrt = Writer(out_path, cap.fps, size)

    # load json file
    json_path = os.path.join(data_dir, ".json", "keypoints.json")
    kps_data = load(json_path)
    json_path = os.path.join(data_dir, ".json", "individual.json")
    ind_data = load(json_path)

    cap.set_pos_frame_count(start_frame_num - 1)
    for frame_num in tqdm(range(start_frame_num, end_frame_num + 1)):
        _, frame = cap.read()
        frame = kps_write_frame(frame, kps_data, frame_num)
        frame = write_frame(ind_data, frame, field, frame_num)
        wrt.write(frame)

    # release memory
    del cap, wrt, kps_data, ind_data
    gc.collect()


def write_frame(
    data: List[Dict[str, Any]], frame: NDArray, field: NDArray, frame_num: int
):
    field_tmp = write_field(data, field.copy(), frame_num)
    frame = concat_field_with_frame(frame.copy(), field_tmp)
    return frame


def write_field(inds_data: List[Dict[str, Any]], field: NDArray, frame_num: int):
    for data in inds_data:
        if data["frame"] == frame_num:
            field = _vis_body_face(data, field)
            # field = _vis_arm(data, field)
            field = _vis_id(data, field)

    return field


def _vis_id(data: Dict[str, Any], field: NDArray):
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


def _vis_body_face(data: Dict[str, Any], field: NDArray):
    # face vector
    field = __vis_arrow("face", data, field)
    # body vector
    field = __vis_arrow("body", data, field)

    return field


def _vis_arm(data: Dict[str, Any], field: NDArray):
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
