from typing import List

import cv2
import numpy as np
from numpy.typing import NDArray


def get_color(idx: int):
    idx = idx * 17
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def put_frame_num(img: NDArray, frame_num: int):
    return cv2.putText(
        img,
        "Frame:{}".format(frame_num),
        (10, 50),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 0),
    )


def draw_skeleton(frame: NDArray, t_id: int, kp: NDArray, vis_thresh: float = 0.2):
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
        cor_x, cor_y = int(kp[n, 0]), int(kp[n, 1])
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
