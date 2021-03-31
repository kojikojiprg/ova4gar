from common import common, transform
from tracker import main as tr
from person import main as pd
from group import main as gd
from display.display import display
import os
import cv2


IS_TRACKING = True
IS_PERSON = True
IS_GROUP = True
IS_DISPLAY = True

ROOM_NUM = '09'
DATE = '20210304'
NAME = 'pass1'


if __name__ == '__main__':
    video_path = os.path.join(common.data_dir, '{0}/{1}/{2}/AlphaPose_{2}.mp4'.format(ROOM_NUM, DATE, NAME))
    out_dir = os.path.join(common.out_dir, '{0}/{1}/{2}/'.format(ROOM_NUM, DATE, NAME))
    field_path = os.path.join(common.data_dir, 'field.png')
    keypoints_path = os.path.join(common.data_dir, '{0}/{1}/{2}/alphapose-results.json'.format(ROOM_NUM, DATE, NAME))
    tracking_json_path = os.path.join(common.json_dir, '{0}/{1}/{2}/tracking.json'.format(ROOM_NUM, DATE, NAME))
    person_json_path = os.path.join(common.json_dir, '{0}/{1}/{2}/person.json'.format(ROOM_NUM, DATE, NAME))
    group_json_path = os.path.join(common.json_dir, '{0}/{1}/{2}/group.json'.format(ROOM_NUM, DATE, NAME))

    # homography
    field_raw = cv2.imread(field_path)
    p_video = common.homo[ROOM_NUM][0]
    p_field = common.homo[ROOM_NUM][1]
    homo = transform.Homography(p_video, p_field, field_raw.shape)

    if IS_TRACKING:
        tr.main(keypoints_path, tracking_json_path)

    if IS_PERSON:
        pd.main(tracking_json_path, person_json_path, homo)

    if IS_GROUP:
        gd.main(person_json_path, group_json_path, homo)

    if IS_DISPLAY:
        display(video_path, out_dir, person_json_path, group_json_path, field_raw)
