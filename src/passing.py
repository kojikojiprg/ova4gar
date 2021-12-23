import argparse
import os
from glob import glob

import cv2

from common import common, transform
from display.display import display
from group_activity import main as ga
from individual_activity import main as ia
from tracker import main as tr

# is_tracking = True
# is_individual_activity = True
# is_group_activity = True
# is_display = True

# room_num = '09'
# date = '20210304'
# name = 'pass2'


METHOD = "passing"


def main(
    room_num,
    date,
    name,
    is_tracking,
    is_individual_activity,
    is_group_activity,
    is_display,
):
    video_path = os.path.join(
        common.data_dir,
        "{0}/{1}/{2}/video/pose_{3}.mp4".format(
            room_num, date, name, common.split_path(name)[-1]
        ),
    )
    out_dir = os.path.join(
        common.data_dir, "{0}/{1}/{2}/out/".format(room_num, date, name)
    )
    field_path = os.path.join(common.data_dir, "{}/field.png".format(room_num))
    keypoints_path = os.path.join(
        common.data_dir,
        "{0}/{1}/{2}/json/pose-results.json".format(room_num, date, name),
    )
    tracking_json_path = os.path.join(
        common.data_dir, "{0}/{1}/{2}/json/tracking.json".format(room_num, date, name)
    )
    individual_activity_json_path = os.path.join(
        common.data_dir,
        "{0}/{1}/{2}/json/individual_activity.json".format(room_num, date, name),
    )
    group_activity_json_path = os.path.join(
        common.data_dir,
        "{0}/{1}/{2}/json/group_activity.json".format(room_num, date, name),
    )

    # homography
    field_raw = cv2.imread(field_path)
    p_video = transform.homo[room_num][0]
    p_field = transform.homo[room_num][1]
    homo = transform.Homography(p_video, p_field, field_raw.shape)

    if is_tracking:
        tr.main(keypoints_path, tracking_json_path)

    if is_individual_activity:
        ia.main(tracking_json_path, individual_activity_json_path, homo)

    if is_group_activity:
        ga.main(
            individual_activity_json_path, group_activity_json_path, field_raw, METHOD
        )

    if is_display:
        display(
            video_path,
            out_dir,
            individual_activity_json_path,
            group_activity_json_path,
            field_raw,
            METHOD,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("room_num", type=str)
    parser.add_argument("date", type=str)
    parser.add_argument("-n", "--name", default=None, type=str)
    parser.add_argument("-a", "--all", default=False, action="store_true")
    parser.add_argument("-t", "--tracking", default=False, action="store_true")
    parser.add_argument(
        "-ia", "--individual_activity", default=False, action="store_true"
    )
    parser.add_argument("-ga", "--group_activity", default=False, action="store_true")
    parser.add_argument("-d", "--display", default=False, action="store_true")

    args = parser.parse_args()
    room_num = args.room_num
    date = args.date
    name = args.name
    is_all = args.all
    is_tracking = args.tracking
    is_individual_activity = args.individual_activity
    is_group_activity = args.group_activity
    is_display = args.display

    if is_all:
        dirs = sorted(glob(f"{common.data_dir}/{room_num}/{date}/{METHOD}/*"))
        if dirs[-1].endswith("make_csv.csv"):
            dirs = dirs[:-1]  # del make_csv.csv
        print(dirs)
        for name in dirs:
            name = f"{METHOD}/" + common.split_path(name)[-1]
            main(
                room_num,
                date,
                name,
                is_tracking,
                is_individual_activity,
                is_group_activity,
                is_display,
            )
    else:
        main(
            room_num,
            date,
            name,
            is_tracking,
            is_individual_activity,
            is_group_activity,
            is_display,
        )
