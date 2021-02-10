from common import common, transform
from tracking.tracking import track
from person import data as pd
from group import data as gd
from display.display import display
import cv2


IS_TRACKING = False
IS_PERSON = False
IS_GROUP = True
IS_DISPLAY = True


if __name__ == '__main__':
    name = 'basketball'
    video_path = common.data_dir + '{0}/{0}_alphapose.mp4'.format(name)
    out_dir = common.out_dir + '{}/'.format(name)
    court_path = common.data_dir + '{}/court.png'.format(name)
    keypoints_path = common.data_dir + '{}/keypoints.json'.format(name)
    tracking_db_path = common.db_dir + '{}/tracking.db'.format(name)
    person_db_path = common.db_dir + '{}/person.db'.format(name)
    group_db_path = common.db_dir + '{}/group.db'.format(name)

    # homography
    court_raw = cv2.imread(court_path)
    p_video = common.homo[name][0]
    p_court = common.homo[name][1]
    homo = transform.Homography(p_video, p_court, court_raw.shape)

    if IS_TRACKING:
        track(keypoints_path, tracking_db_path, name)

    if IS_PERSON:
        pd.make_database(tracking_db_path, person_db_path, homo)

    if IS_GROUP:
        gd.make_database(person_db_path, group_db_path, homo)

    if IS_DISPLAY:
        display(video_path, out_dir, person_db_path, group_db_path, court_raw, homo)
