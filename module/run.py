from common import common, transform
from tracking.tracking import track
from person import data as pd
from group import data as gd
from display.display import display
import os
import cv2


IS_TRACKING = True
IS_PERSON = True
IS_GROUP = True
IS_DISPLAY = True

ROOM_NUM = '09'
DATE = '20210304'
NAME = 'gaze1-1'

if __name__ == '__main__':
    video_path = os.path.join(common.data_dir, '{0}/{1}/{2}/AlphaPose_{2}.mp4'.format(ROOM_NUM, DATE, NAME))
    out_dir = os.path.join(common.out_dir, '{0}/{1}/{2}/'.format(ROOM_NUM, DATE, NAME))
    field_path = os.path.join(common.data_dir, 'field.png')
    keypoints_path = os.path.join(common.data_dir, '{0}/{1}/{2}/alphapose-results.json'.format(ROOM_NUM, DATE, NAME))
    tracking_db_path = os.path.join(common.db_dir, '{0}/{1}/{2}/tracking.db'.format(ROOM_NUM, DATE, NAME))
    person_db_path = os.path.join(common.db_dir, '{0}/{1}/{2}/person.db'.format(ROOM_NUM, DATE, NAME))
    group_db_path = os.path.join(common.db_dir, '{0}/{1}/{2}/group.db'.format(ROOM_NUM, DATE, NAME))

    # homography
    field_raw = cv2.imread(field_path)
    p_video = common.homo[ROOM_NUM][0]
    p_field = common.homo[ROOM_NUM][1]
    homo = transform.Homography(p_video, p_field, field_raw.shape)

    if IS_TRACKING:
        track(keypoints_path, tracking_db_path)

    if IS_PERSON:
        pd.make_database(tracking_db_path, person_db_path, homo)

    if IS_GROUP:
        gd.make_database(person_db_path, group_db_path, homo)

    if IS_DISPLAY:
        display(video_path, out_dir, person_db_path, group_db_path, field_raw, homo)
