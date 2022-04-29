import os

from api.inference import InferenceModel
from api.parser import file_parser


def main():
    args = file_parser()

    # create data dir
    os.makedirs(args.data_dir, exist_ok=True)

    model = InferenceModel(args)
    model.inference(args.video_path, args.data_dir)


if __name__ == "__main__":
    main()
