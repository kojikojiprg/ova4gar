import argparse
import os

from keypoint.extracter import Extractor
from utility.logger import setup_logger


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
    hrnet_cfg_path = "config/higher-hrnet/w32_512_adam_lr1e-3.yaml"
    unitrack_cfg_path = "config/unitrack/barlowtwins.yaml"
    extractor = Extractor(hrnet_cfg_path, unitrack_cfg_path, logger)
    extractor.predict(video_path, data_dir)


if __name__ == "__main__":
    main()
