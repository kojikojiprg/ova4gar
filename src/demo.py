import argparse
import imp
import os

import cv2
import yaml

from keypoint.extracter import Extractor
from individual.individual_analyzer import IndividualAnalyzer
from utility.logger import setup_logger
from utility.transform import Homography


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path of a raw video.")
    parser.add_argument(
        "data_dir", type=str, help="Path of data directory where results are saved."
    )
    parser.add_argument(
        "-c", "--cfg_path", type=str, default="config/config.yaml", help="Config file path."
    )
    parser.add_argument(
        "-rn", "--room_num", type=str, default="02", help="The room number of operating room"
    )
    return parser.parse_args()


def main():
    args = parser()
    video_path = args.video_path
    data_dir = args.data_dir
    cfg_path = args.cfg_path
    room_num = args.room_num

    # create data dir
    os.makedirs(data_dir, exist_ok=True)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # homography
    homo_cfg = cfg["homography"]
    field_raw = cv2.imread(homo_cfg["field_path"])
    p_video = homo_cfg[room_num]["video"]
    p_field = homo_cfg[room_num]["field"]
    homo = Homography(p_video, p_field, field_raw.shape)

    # create logger
    log_dir = os.path.join(data_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    # extract keypoints
    hrnet_cfg_path = "config/higher-hrnet/w32_512_adam_lr1e-3.yaml"
    unitrack_cfg_path = "config/unitrack/barlowtwins.yaml"
    extractor = Extractor(hrnet_cfg_path, unitrack_cfg_path, logger)
    extractor.predict(video_path, data_dir)

    # individual actitivy
    anlyzer: IndividualAnalyzer = IndividualAnalyzer(**cfg)
    anlyzer.analyze(data_dir, homo)


if __name__ == "__main__":
    main()
