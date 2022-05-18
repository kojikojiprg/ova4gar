import os
import sys
from glob import glob

from api.inference import InferenceModel
from api.parser import surgery_parser

sys.path.append("src")
from utility.logger import logger


def main():
    args = surgery_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    video_dir = os.path.join("video", args.room_num, args.surgery_num)
    video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))
    logger.info(f"=> video paths:\n{video_paths}")

    data_dirs = []
    for video_path in video_paths:
        name = os.path.basename(video_path).replace(".mp4", "")
        data_dir = os.path.join("data", args.room_num, args.surgery_num, name)
        data_dirs.append(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    model = InferenceModel(args, logger)

    for video_path, data_dir in zip(video_paths, data_dirs):
        logger.info(f"=> processing {video_path}")
        model.inference(video_path, data_dir)


if __name__ == "__main__":
    main()
