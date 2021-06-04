from common.keypoint import Keypoints
from common.json import IA_FORMAT
import cv2


def disp_tracking(individual_activity_datas, frame):
    for data in individual_activity_datas:
        individual_activity_id = data[IA_FORMAT[0]]
        keypoints = data[IA_FORMAT[2]]
        if keypoints is not None:
            point = Keypoints(keypoints).get_middle('Hip')
            if point is not None:
                cv2.circle(frame, tuple(point), 7, (0, 0, 255), thickness=-1)
                cv2.putText(frame, str(individual_activity_id), tuple(point),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    return frame
