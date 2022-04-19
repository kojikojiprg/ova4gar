import argparse
import os

from keypoints.hrnet import HRNetExtractor
from utils.logger import setup_logger


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path of a raw video.")
    parser.add_argument(
        "data_dir", type=str, help="Path of data directory where results are saved."
    )
    return parser.parse_args()


def main():
    args = parser()
    video_path = args.video_path
    data_dir = args.data_dir

    # create data dir
    os.makedirs(data_dir, exist_ok=True)

    # create logger
    log_dir = os.path.join(data_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    # extract keypoints
    hrnet_cfg_path = (
        "./submodules/hrnet/experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml"
    )
    hrnet_opts = [
        "TEST.MODEL_FILE",
        "models/hrnet/pose_higher_hrnet_w32_512.pth",
        "TEST.FLIP_TEST",
        "False",
    ]
    extractor = HRNetExtractor(hrnet_cfg_path, hrnet_opts, logger)
    extractor.predict(video_path, data_dir)


if __name__ == "__main__":
    main()
