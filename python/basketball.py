from common import common
from tracking.tracking import track
from extract_indicator.extract_indicator import extract_indicator


IS_TRACKING = True
IS_INDICATOR = False


if __name__ == '__main__':
    keypoints_path = common.data_dir + 'basketball/keypoints.json'
    tracking_db_path = common.db_dir + 'basketball/tracking.db'
    indicator_db_path = common.db_dir + 'basketball/indicator.db'

    if IS_TRACKING:
        track(keypoints_path, tracking_db_path)

    if IS_INDICATOR:
        extract_indicator(tracking_db_path, indicator_db_path)
