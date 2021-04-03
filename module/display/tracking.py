from common.keypoint import Keypoints
from common.json import PERSON_FORMAT
import cv2


def disp_tracking(person_datas, frame):
    for data in person_datas:
        person_id = data[PERSON_FORMAT[0]]
        keypoints = data[PERSON_FORMAT[2]]
        if keypoints is not None:
            keypoints = Keypoints(keypoints)
            point = keypoints.get_middle('Hip')
            cv2.circle(frame, tuple(point), 7, (0, 0, 255), thickness=-1)
            cv2.putText(frame, str(person_id), tuple(point),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    return frame
