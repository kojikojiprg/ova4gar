from common import common
from tracking import track


if __name__ == '__main__':
    keypoints_path = common.data_dir + 'basketball/keypoints.json'
    tracking_db_path = common.db_dir + 'basketball/tracking.db'

    track(keypoints_path, tracking_db_path)
