from common import common, transform
from tracking.tracking import track
from extract_indicator.extract_indicator import extract_indicator
from display.display import display
import numpy as np
import cv2


IS_TRACKING = False
IS_INDICATOR = False


if __name__ == '__main__':
    video_path = common.data_dir + 'basketball/basketball_alphapose.mp4'
    out_dir = common.out_dir + 'basketball/'
    keypoints_path = common.data_dir + 'basketball/keypoints.json'
    tracking_db_path = common.db_dir + 'basketball/tracking.db'
    indicator_db_path = common.db_dir + 'basketball/indicator.db'
    court_path = common.data_dir + 'basketball/court.png'

    # homography
    court_raw = cv2.imread(court_path)
    p_video = np.float32([[210, 364], [1082, 362], [836, 488], [438, 489]])
    p_court = np.float32([[24, 24], [568, 24], [383, 232], [205, 232]])
    homo = transform.Homography(p_video, p_court, court_raw.shape)

    if IS_TRACKING:
        track(keypoints_path, tracking_db_path)

    if IS_INDICATOR:
        extract_indicator(tracking_db_path, indicator_db_path)

    display(video_path, out_dir, tracking_db_path, indicator_db_path, court_raw, homo)
