import os
import sys
from glob import glob

sys.path.append("src")
from demo_api.parser import surgery_parser
from demo_api.visualizer import Visalizer
from utility.logger import logger


def main():
    args = surgery_parser()

    video_dir = os.path.join("video", args.room_num, args.surgery_num, args.expand_name)
    video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))
    logger.info(f"=> video paths:\n{video_paths}")

    data_dirs = []
    for video_path in video_paths:
        name = os.path.basename(video_path).replace(".mp4", "")
        data_dir = os.path.join(
            "data", args.room_num, args.surgery_num, args.expand_name, name
        )
        data_dirs.append(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    viser = Visalizer(args, logger)

    for video_path, data_dir in zip(video_paths, data_dirs):
        logger.info(f"=> processing {video_path}")
        viser.write_video(video_path, data_dir)


if __name__ == "__main__":
    main()
