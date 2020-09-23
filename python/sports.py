from common import common, transform
from tracking.tracking import track
from extract_indicator.extract_indicator import extract_indicator
from display.display import display
import cv2


IS_TRACKING = False
IS_INDICATOR = False


if __name__ == '__main__':
    name = 'basketball'
    video_path = common.data_dir + '{0}/{0}_alphapose.mp4'.format(name)
    out_dir = common.out_dir + '{}/'.format(name)
    court_path = common.data_dir + '{}/court.png'.format(name)
    keypoints_path = common.data_dir + '{}/keypoints.json'.format(name)
    tracking_db_path = common.db_dir + '{}/tracking.db'.format(name)
    indicator_db_path = common.db_dir + '{}/indicator.db'.format(name)

    # homography
    court_raw = cv2.imread(court_path)
    p_video = common.homo['basketball'][0]
    p_court = common.homo['basketball'][1]
    homo = transform.Homography(p_video, p_court, court_raw.shape)

    if IS_TRACKING:
        track(keypoints_path, tracking_db_path)

    if IS_INDICATOR:
        extract_indicator(tracking_db_path, indicator_db_path)

    display(video_path, out_dir, tracking_db_path, indicator_db_path, court_raw, homo)
