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
from utility.video import Capture, Writer


def write_video(video_path, data_dir, start_frame_num, end_frame_num):
    # create video capture
    cap = Capture(video_path)

    # create video writer
    out_path = os.path.join(data_dir, "video", "keypoints.mp4")
    wrt = Writer(out_path, cap.fps, cap.size)

    # load json file
    json_path = os.path.join(data_dir, ".json", "keypoints.json")
    kps_data = load(json_path)

    # write video
    cap.set_pos_frame_count(start_frame_num - 1)
    for frame_num in tqdm(range(start_frame_num, end_frame_num + 1)):
        ret, frame = cap.read()
        frame = write_frame(wrt, frame, kps_data, frame_num)
        wrt.write(frame)

    del cap, wrt, kps_data
    gc.collect()


def write_frame(
    frame: NDArray,
    kps_data: List[Dict[str, Any]],
    frame_num: int,
    delete_height: int = 20,
):
    # add keypoints to image
    frame = _put_frame_num(frame, frame_num)
    for kps in kps_data:
        if kps["frame"] == frame_num:
            frame = _draw_skeleton(
                frame, kps["id"], np.array(kps["keypoints"]), delete_height
            )

    return frame


def _put_frame_num(img: NDArray, frame_num: int):
    return cv2.putText(
        img,
        "Frame:{}".format(frame_num),
        (10, 50),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 0),
    )


def _draw_skeleton(
    frame: NDArray, t_id: int, kp: NDArray, delete_height, vis_thresh: float = 0.2
):
    l_pair = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),  # Head
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),  # Body
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
    ]
    p_color = [
        # Nose, LEye, REye, LEar, REar
        (0, 255, 255),
        (0, 191, 255),
        (0, 255, 102),
        (0, 77, 255),
        (0, 255, 0),
        # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
        (77, 255, 255),
        (77, 255, 204),
        (77, 204, 255),
        (191, 255, 77),
        (77, 191, 255),
        (191, 255, 77),
        # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        (204, 77, 255),
        (77, 255, 204),
        (191, 77, 255),
        (77, 255, 191),
        (127, 77, 255),
        (77, 255, 127),
        (0, 255, 255),
    ]
    line_color = [
        (0, 215, 255),
        (0, 255, 204),
        (0, 134, 255),
        (0, 255, 50),
        (77, 255, 222),
        (77, 196, 255),
        (77, 135, 255),
        (191, 255, 77),
        (77, 255, 77),
        (77, 222, 255),
        (255, 156, 127),
        (0, 127, 255),
        (255, 127, 77),
        (0, 77, 255),
        (255, 77, 36),
    ]

    img = frame.copy()
    part_line = {}

    # draw keypoints
    for n in range(len(kp)):
        if kp[n, 2] <= vis_thresh:
            continue
        cor_x, cor_y = int(kp[n, 0]), int(kp[n, 1]) - delete_height
        part_line[n] = (cor_x, cor_y)
        cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)

    # draw limbs
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(
                img,
                start_xy,
                end_xy,
                line_color[i],
                2 * int(kp[start_p, 2] + kp[end_p, 2]) + 1,
            )

    # draw track id
    pt = np.mean([kp[5], kp[6], kp[11], kp[12]], axis=0).astype(int)[:2]
    img = cv2.putText(
        img,
        str(t_id),
        tuple(pt),
        cv2.FONT_HERSHEY_PLAIN,
        max(1, int(img.shape[1] / 500)),
        (255, 255, 0),
        max(1, int(img.shape[1] / 500)),
    )

    return img
