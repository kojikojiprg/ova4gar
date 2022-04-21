from typing import List
import cv2
import numpy as np
from numpy.typing import NDArray


def get_color(idx: int):
    idx = idx * 17
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def draw_skeleton(img: NDArray, t_id: int, kp: List[list]):
    skeleton = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ]

    color = get_color(t_id)

    # draw skeleton's stems
    for i, j in skeleton:
        if (
            kp[i - 1][0] >= 0
            and kp[i - 1][1] >= 0
            and kp[j - 1][0] >= 0
            and kp[j - 1][1] >= 0
            and (
                len(kp[i - 1]) <= 2
                or (len(kp[i - 1]) > 2 and kp[i - 1][2] > 0.1 and kp[j - 1][2] > 0.1)
            )
        ):
            st = (int(kp[i - 1][0]), int(kp[i - 1][1]))
            ed = (int(kp[j - 1][0]), int(kp[j - 1][1]))
            img = cv2.line(img, st, ed, color, max(1, int(img.shape[1] / 150.0)))

    # draw keypoints
    for j in range(len(kp)):
        if kp[j][0] >= 0 and kp[j][1] >= 0:
            pt = (int(kp[j][0]), int(kp[j][1]))
            if len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 1.1):
                img = cv2.circle(img, pt, 2, tuple((0, 0, 255)), 2)
            elif len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1):
                img = cv2.circle(img, pt, 2, tuple((255, 0, 0)), 2)

    # draw track id
    pt = np.mean([kp[5], kp[6], kp[11], kp[12]], axis=0).astype(int)[:2]
    img = cv2.putText(img, str(t_id), tuple(pt), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0))

    return img
