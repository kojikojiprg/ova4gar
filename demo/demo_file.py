import os
import sys

from api.inference import InferenceModel
from api.parser import file_parser

sys.path.append("src")
from utility.logger import logger


def main():
    args = file_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # create data dir
    os.makedirs(args.data_dir, exist_ok=True)

    model = InferenceModel(args, logger)
    logger.info(f"=> processing {args.video_path}")
    model.inference(args.video_path, args.data_dir)


if __name__ == "__main__":
    main()
