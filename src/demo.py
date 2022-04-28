import argparse
import os
from glob import glob

import cv2
import yaml

from group.group_analyzer import GroupAnalyzer
from individual.individual_analyzer import IndividualAnalyzer
from keypoint.extracter import Extractor
from utility.logger import setup_logger
from utility.transform import Homography


def parser():
    parser = argparse.ArgumentParser()
    # requires
    parser.add_argument(
        "-rn",
        "--room_num",
        required=True,
        type=str,
        help="The room number of operating room",
    )
    parser.add_argument("-d", "--date", required=True, type=str, help="Date")

    # options
    parser.add_argument(
        "-c",
        "--cfg_path",
        type=str,
        default="config/config.yaml",
        help="Config file path.",
    )
    parser.add_argument(
        "-wk",
        "--without_keypoint",
        default=False,
        action="store_true",
        help="Without keypoint extraction.",
    )
    parser.add_argument(
        "-wi",
        "--without_individual",
        default=False,
        action="store_true",
        help="Without idividual analyzation.",
    )
    parser.add_argument(
        "-wg",
        "--without_group",
        default=False,
        action="store_true",
        help="Without group analyzation.",
    )
    return parser.parse_args()


def main():
    args = parser()
    room_num = args.room_num
    date = args.date
    cfg_path = args.cfg_path
    without_keypoint = args.without_keypoint
    without_individual = args.without_individual
    without_group = args.without_group

    video_dir = os.path.join("video", room_num, date)
    video_paths = glob(os.path.join(video_dir, "*.mp4"))
    data_dirs = []
    for video_path in video_paths:
        name = os.path.basename(video_path).replace(".mp4", "")
        data_dir = os.path.join("data", room_num, date, name)
        data_dirs.append(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    # open config file
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # homography
    homo_cfg = cfg["homography"]
    field = cv2.imread(homo_cfg["field_path"])

    p_video = homo_cfg[room_num]["video"]
    p_field = homo_cfg[room_num]["field"]
    homo = Homography(p_video, p_field, field.shape)

    # create logger
    log_dir = os.path.join(data_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    for video_path, data_dir in zip(video_paths, data_dirs):
        # extract keypoints
        if not without_keypoint:
            extractor = Extractor(cfg, logger)
            extractor.predict(video_path, data_dir)

        # individual actitivy
        if not without_individual:
            individual_anlyzer: IndividualAnalyzer = IndividualAnalyzer(cfg, logger)
            individual_anlyzer.analyze(data_dir, homo, field)

        # group actitivy
        if not without_group:
            group_anlyzer: GroupAnalyzer = GroupAnalyzer(cfg, logger)
            group_anlyzer.analyze(data_dir, field)


if __name__ == "__main__":
    main()
