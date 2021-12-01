from common.json import IA_FORMAT
import cv2


def disp_tracking(individual_activity_datas, frame):
    for data in individual_activity_datas:
        individual_activity_id = data[IA_FORMAT[0]]
        point = data[IA_FORMAT[2]]
        if point is not None:
            cv2.circle(frame, tuple(point), 7, (0, 0, 255), thickness=-1)
            cv2.putText(frame, str(individual_activity_id), tuple(point),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    return frame
