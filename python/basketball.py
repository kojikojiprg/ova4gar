from common import common
from tracking import track
from extract_indicator import extract_indicator


if __name__ == '__main__':
    keypoints_path = common.data_dir + 'basketball/keypoints.json'
    tracking_db_path = common.db_dir + 'basketball/tracking.db'
    indicator_db_path = common.db_dir + 'basketball/indicator.db'

    # track(keypoints_path, tracking_db_path)
    extract_indicator(tracking_db_path, indicator_db_path)
